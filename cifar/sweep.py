from pathlib import Path
import pickle, traceback, json, os, logging

import hydra, optuna
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from optuna.pruners import HyperbandPruner
from optuna.integration import PyTorchLightningPruningCallback
from optuna.storages import RDBStorage
from lightning.pytorch.callbacks import ModelCheckpoint

from .train import main_train

GPU_LIST = [int(x) for x in os.getenv("GPU_LIST", "0,1,2,3,4,5,6,7,8").split(",")]

def pick_free_gpu() -> int:
    """Return a GPU from GPU_LIST with <200 MiB used, else round‑robin."""
    try:
        import subprocess
        lines = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=memory.used,index",
             "--format=csv,noheader,nounits"]
        ).decode().strip().splitlines()
        mem_idx = [(int(l.split(",")[0]), int(l.split(",")[1])) for l in lines]
        for mem, idx in sorted(mem_idx):
            if idx in GPU_LIST and mem < 200:
                return idx
    except Exception:
        pass
    # fallback → deterministic round‑robin over allowed list
    pick = pick_free_gpu.counter % len(GPU_LIST)
    pick_free_gpu.counter += 1
    return GPU_LIST[pick]

pick_free_gpu.counter = 0




_SQLITE_URL = "sqlite:///koopman_sweep.db"

def make_storage():
    return RDBStorage(
        url=_SQLITE_URL,
        engine_kwargs={"connect_args": {"timeout": 60,
                                        "check_same_thread": False}},
        heartbeat_interval=60,
    )

try:
    storage = make_storage()
except AssertionError:
    if Path("koopman_sweep.db").exists():
        print("⚠️  Corrupt Optuna DB detected → deleting and recreating.")
        Path("koopman_sweep.db").unlink()
        storage = make_storage()
    else:
        raise






# ─── create / attach to study ──────────────────────────────
STUDY_NAME = "koopman_full_sched_search"

pruner = HyperbandPruner(min_resource=1, max_resource=4, reduction_factor=3)
study = optuna.create_study(
    study_name=STUDY_NAME,
    storage=storage,
    direction="minimize",
    pruner=pruner,
    load_if_exists=True,
)

# ─── simple logger to collect failures ─────────────────────
logging.basicConfig(filename="sweep.log",
                    filemode="a",
                    level=logging.INFO,
                    format="%(asctime)s  %(message)s")

# ─── the sweep entry ‑ Hydra + Optuna objective ───────────
@hydra.main(config_path="../conf", config_name="sweep", version_base="1.3")
def sweep_entry(cfg: DictConfig):

    base_epochs = cfg.koopman.train.max_epochs  # 3 or 4 from YAML

    def objective(trial: optuna.Trial):

        print(f"\n▶️  NEW TRIAL  #{trial.number:03d}  {trial.params}\n", flush=True)
        # Pick an available gpu
        gpu = pick_free_gpu()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cfg.koopman.train.devices = [0]          # Lightning sees only that GPU

        # tune those
        cfg.koopman.train.autoencoder_lr = trial.suggest_float(
            "autoencoder_lr", 5e-4, 2e-3, log=True)
        cfg.koopman.train.lie_lr = trial.suggest_float(
            "lie_lr", 5e-5, 5e-4, log=True)
        cfg.koopman.train.weight_decay = trial.suggest_float(
            "weight_decay", 0.0, 0.1)
        cfg.model.dropout = trial.suggest_categorical(
            "dropout", [0.0, 0.1, 0.2])
        cfg.koopman.operator.init_std = trial.suggest_categorical(
            "operator.init_std", [1e-4, 1e-3, 1e-2])

        # tune scheduler and its hyperparams too
        scheduler_name = trial.suggest_categorical(
            "lr_scheduler", ["ReduceLROnPlateau", "LinearWarmupCosineAnnealingLR"])
        cfg.koopman.train.lr_scheduler = scheduler_name   # LightningModule reads this

        if scheduler_name == "LinearWarmupCosineAnnealingLR":
            multiplier   = trial.suggest_int("epochs_mult", 2, 4)
            total_epochs = base_epochs * multiplier
            warmup_frac  = trial.suggest_float("warmup_frac", 0.05, 0.10)
            eta_min_frac = trial.suggest_float("eta_min_frac", 0.05, 0.30)

            cfg.koopman.train.max_epochs    = total_epochs
            cfg.koopman.train.warmup_step = max(1, int(warmup_frac * total_epochs))
            cfg.koopman.train.eta_min_frac  = eta_min_frac
        else:
            cfg.koopman.train.max_epochs    = base_epochs
            cfg.koopman.train.plateau_factor = trial.suggest_float(
                "plateau_factor", 0.90, 0.999)

        # avoid a complete crash when OOM
        try:
            pruning_cb = PyTorchLightningPruningCallback(trial, monitor="fid_train")
            trainer = main_train(cfg, additional_cbs=[pruning_cb])
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # log parameters + error and tell Optuna this trial failed
                logging.info("OOM  " + json.dumps(trial.params))
                logging.info(traceback.format_exc())
                raise
            else:
                raise

        # pick best choice on fid, bookkeeping
        ckpt_fid = next(cb for cb in trainer.callbacks
                        if isinstance(cb, ModelCheckpoint) and cb.monitor == "fid_train")
        trial.set_user_attr("best_ckpt_path", ckpt_fid.best_model_path or "NONE")

        fid_final = trainer.callback_metrics["fid_train"].item()
        return fid_final

    # one worker per process
    study.optimize(objective, timeout=24*60*60, n_trials=None, show_progress_bar=True, 
                   callbacks=[
                       lambda study, trial: print(f"✓ Trial {trial.number} → FID {trial.value:.3f}")
                    ])

    # persist the entire study for inspection
    study.trials_dataframe().to_csv("study_results.csv", index=False)
    with open("study.pkl", "wb") as f:
        pickle.dump(study, f)

if __name__ == "__main__":
    sweep_entry()