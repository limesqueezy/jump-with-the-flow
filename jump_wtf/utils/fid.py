import torch, torchvision
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from pathlib import Path
import torch.nn.functional as F
from jump_wtf.utils.sampling import sample_efficient
from lightning.pytorch.callbacks import Callback

class FIDTrainCallback(Callback):
    def __init__(self, every_n_steps=500,
                 fake_batches=8, bs=256, rollout_steps=1):
        self.every, self.fake_batches, self.bs = every_n_steps, fake_batches, bs
        self.rollout_steps = rollout_steps
        self._step_counter = 0

    # signature with *args **kwargs swallows extra params can get richer signatures outputs, batch, batch_idx blabla
    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs): # design choice to run the after optimizer step hook there's on_train_batch_start too
        self._step_counter += 1
        if self._step_counter % self.every:
            return                                # not time yet

        fid = pl_module.fid_train
        fid.reset()                               # clear fake stats only

        pl_module.eval()
        with torch.no_grad():
            for _ in range(self.fake_batches):
                fake = sample_efficient(
                    pl_module, n_iter=self.rollout_steps, n_samples=self.bs
                )
                fid.update(to_fid(fake), real=False)
        pl_module.train()

        score = fid.compute()
        pl_module.log(
            "fid_train",
            score,
            on_step=True,       # step‐wise metric, requried!
            on_epoch=False,
            prog_bar=True,
            sync_dist=True
        )

class FIDValCallback(Callback):
    def __init__(self, fake_batches=12, bs=256, rollout_steps=100):
        self.fake_batches, self.bs, self.rollout_steps = fake_batches, bs, rollout_steps

    def on_validation_epoch_end(self, trainer, pl_module):
        fid = pl_module.fid_val
        fid.reset()

        pl_module.eval()
        with torch.no_grad():
            for _ in range(self.fake_batches):
                fake = sample_efficient(
                    pl_module, n_iter=self.rollout_steps, n_samples=self.bs
                )
                fid.update(to_fid(fake), real=False)
        score = fid.compute()
        pl_module.log("fid_val", score, prog_bar=True, sync_dist=True)

def to_fid(img: torch.Tensor) -> torch.Tensor:
    """
    Resize to 299×299 and ensure a 3-channel uint8 image:
      - Grayscale (C=1)
      - RGB (C=3)
    Expects floats in [–1, 1], returns uint8 on the same device.
    """
    # 1) resize & scale to [0,255]
    img = torch.nn.functional.interpolate(
        img, size=299, mode="bilinear", antialias=True
    )
    img = (img.clamp(-1, 1) * 127.5 + 127.5).to(torch.uint8)
    # ensure 3 channels
    C = img.shape[1]
    if C == 1:
        img = img.repeat(1, 3, 1, 1)
    elif C == 3:
        pass  # already RGB
    else:
        raise ValueError(f"Unexpected number of channels for FID: got {C}")
    return img

@torch.no_grad()
def compute_real_stats(
    data,
    out,
    bs=256,
    device="cuda",
    num_workers=4,
):
    """
    One‑shot over your real data, dump full FID.state_dict()
    Skips work if `out` already exists.

    Args:
      data: either a torch.utils.data.Dataset or a DataLoader
      out:  where to write the .pth
      bs:   batch-size (ignored if you passed in a DataLoader)
    """
    p = Path(out)
    if p.exists():
        return str(p)

    fid = FrechetInceptionDistance(2048, normalize=True).to(device)

    # if they passed a Dataset, wrap it
    if not isinstance(data, DataLoader):
        loader = DataLoader(data, batch_size=bs,
                            num_workers=num_workers, shuffle=False)
    else:
        loader = data

    for imgs, _ in loader:
        imgs = imgs.to(device)
        fid.update(to_fid(imgs), real=True)

    state = fid.metric_state

    torch.save(state, p)
    return str(p)

def make_fid_metric(state_path: str, device: str = "cuda"):
    """
    Returns a FrechetInceptionDistance metric whose real‐image statistics
    are loaded from a pre‐computed state dict. Fake‐image accumulators
    are zeroed so you can directly call `.update(…, real=False)` on your
    generator outputs.
    """
    state = torch.load(state_path, map_location="cpu")

    mu_sum    = state["real_features_sum"].to(device)
    cov_sum   = state["real_features_cov_sum"].to(device)
    num_real  = int(state["real_features_num_samples"].item())

    # rebuild FID object (don't re‐compute real features)
    m = FrechetInceptionDistance(
        feature=2048,
        normalize=True,
        reset_real_features=False
    )
    
    m.to(device)

    # 3) Overwrite its buffers to match exactly what we saw
    m.real_features_sum.copy_(mu_sum)
    m.real_features_cov_sum.copy_(cov_sum)
    m.real_features_num_samples.fill_(num_real)

    # 4) Clear the fake side so you can accumulate new fakes
    m.fake_features_sum.zero_()
    m.fake_features_cov_sum.zero_()
    m.fake_features_num_samples.zero_()

    return m
