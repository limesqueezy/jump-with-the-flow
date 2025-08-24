import optuna
from optuna.pruners import HyperbandPruner
from optuna.integration import PyTorchLightningPruningCallback
import hydra
from omegaconf import DictConfig
from .train import main_train

import pickle, gzip
from pathlib import Path
from hydra.core.hydra_config import HydraConfig
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

@hydra.main(config_path="../conf", config_name="sweep", version_base="1.3")
def sweep_entry(cfg: DictConfig):

    # Optuna objective --------------------------------------------------
    def objective(trial: optuna.Trial):
        # ---- hyper-parameter suggestions -----------------------------
        cfg.koopman.train.autoencoder_lr = trial.suggest_float(
            "autoencoder_lr", 5e-4, 2e-3, log=True)
        cfg.koopman.train.lie_lr = trial.suggest_float(
            "lie_lr", 5e-5, 5e-4, log=True)
        cfg.koopman.train.weight_decay = trial.suggest_float(
            "weight_decay", 0.0, 1e-4)
        cfg.model.dropout = trial.suggest_categorical(
            "dropout", [0.0, 0.1, 0.2])
        cfg.koopman.operator.init_std = trial.suggest_categorical(
            "operator.init_std", [1e-4, 1e-3, 1e-2])
        cfg.koopman.train.warmup_step = trial.suggest_categorical(
            "warmup_step", [0, 10, 50])

        # ---- each trial uses just two epochs -------------------------
        cfg.koopman.train.max_epochs = 2

        # ---- add pruning callback (monitors fid_train) --------------
        pruning_cb = PyTorchLightningPruningCallback(
            trial, monitor="fid_train")

        # run and get the Trainer (with your existing callbacks) back
        trainer = main_train(cfg, additional_cbs=[pruning_cb])

        # 1) record the Hydra run directory
        run_dir = Path(HydraConfig.get().runtime.output_dir)
        trial.set_user_attr("hydra_run_dir", str(run_dir))

        # 2) grab your two ModelCheckpoint callbacks by what they monitor
        ckpt_fid_train = next(
            cb for cb in trainer.callbacks # type: ignore[attr-defined]
            if isinstance(cb, ModelCheckpoint) and cb.monitor == "fid_train"
        )
        ckpt_cb = next(
            cb for cb in trainer.callbacks # type: ignore[attr-defined]
            if isinstance(cb, ModelCheckpoint) and cb.monitor == "total_loss"
        )

        # 3) record the best‐of‐those checkpoint paths
        best_ckpt = ckpt_fid_train.best_model_path or ckpt_cb.best_model_path
        trial.set_user_attr("best_ckpt_path", best_ckpt)

        # 4) fetch the final fid_train metric and return it
        fid_final = trainer.callback_metrics["fid_train"].item()
        trial.set_user_attr("fid_train_final", fid_final)
        return fid_final


    # ----------------- Optuna study & Hyperband -----------------------
    pruner = HyperbandPruner(min_resource=1,
                             max_resource=2,
                             reduction_factor=3)

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, timeout=24 * 60 * 60)   # 24 h budget

    print("Best params:", study.best_params)
    print("Best fid_train:", study.best_value)

    # 3) Persist the entire study so you can resume or inspect later
    study.trials_dataframe().to_csv("study_results.csv", index=False)
    with open("study.pkl", "wb") as f:
        pickle.dump(study, f)
