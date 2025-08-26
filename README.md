# Jump with the flow

Under train/train.py there's a two-stage pipeline. 
* The CFM stage, learns a velocity field $v_\theta$ and generate trajectories $(t, x_t, v_t, x_1, \Delta t)$ based on the underlying data (MNIST, FMNIST, TFD ..).
* The Koopman stage learns an encoder φ and koopman operator K.

```
conf/               # Hydra configs
train/train.py      # single entry point: runs CFM first, then Koopman
jump_wtf/           # core code
  data/             # datasets + datamodule
  models/           # Autoencoder, Lightning Module (Koopman)
  operators/        # Koopman operator
  utils/            # FID, sampling, plotting, logging
evaluation/         # dataset-specific sampling, FID sweeps, figures
spectral_*.ipynb    # spectral analysis notebooks (per dataset)
logs/               # TensorBoard (symlink)
assets/             # FID stats, cached data (symlink)
```

Create a conda environment,

```
conda env create -f environment.yml
```

If you want access to the assets (dynamics, past weights etc),

```
ln -s /mnt/disk6/ari/koopy_assets assets
```

To train CFM / Koopman on MNIST overwriting some of Hydra's preset values,

```
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=8 \
python -m train.train \
	datadim=mnist \
	model=unet_mnist \
	model@wrapper=unet_mnist_wrapper \
	datadim.dim=[1,28,28] \
	datadim.tag="full" \
	cfm.train.total_steps=20_000 \
	cfm.train.batch_size=256 \
	koopman.train.batch_size=64 \
	koopman.autoencoder.bottleneck=false \
	koopman.train.max_epochs=8 \
	koopman.autoencoder.attention_resolutions=\'14,7\' 
```