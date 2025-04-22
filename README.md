Add Hydra asap

koopman/
│   __init__.py
│
├── data/
│   ├── __init__.py
│   ├── datamodule.py      # Lightning‑style data module
│   └── samplers.py        # TimeGroupedSampler, etc.
│
├── models/
│   ├── __init__.py
│   ├── unet_wrapper.py    # UNetWrapperKoopman + encoder wrapper
│   ├── autoencoder.py     # AutoencoderUNet
│   └── system.py          # End‑to‑end nn.Module that binds all pieces
│
├── operators/
│   ├── __init__.py
│   ├── generic.py         # GenericOperatorState
│   └── utils.py
│
├── losses/
│   ├── __init__.py
│   ├── koopman_loss.py    # grad_phase, decode_predict, reg terms
│   └── multistep.py       # optional multi‑step rollout loss NA
│
├── utils/
│   ├── __init__.py
│   ├── logging.py
|   ├── sampling.py
|   ├── fid.py
│   └── plot.py
│
├── configs/
│   ├── flow_mnist.yaml
│   └── koopman_mnist.yaml
│
└── scripts/
    ├── train_flow.py      # trains flow‑matching UNet; saves ckpt
    ├── train_koopman.py   # trains deep‑Koopman using saved flow
    └── sample.py          # rolls out & saves images/metrics