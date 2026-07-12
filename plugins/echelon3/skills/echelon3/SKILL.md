---
name: echelon3
description: "Use echelon3, a config-driven PyTorch training framework - write YAML configs and run echelon3 train/finetune/evaluate/export/run. Use when the repo uses echelon3, or the user mentions echelon3, its CLI, or training configs."
---

# Using echelon3

echelon3 is a config-driven PyTorch training framework: you describe a run in YAML
(network, data, losses, metrics, optimizer, trainer) and echelon3 assembles and runs
it. This is the shared reference; each agent tool wraps this content in its own format.

## Install

```bash
pip install echelon3
# extras: echelon3[export] (ONNX), echelon3[smp], echelon3[detection], echelon3[test]
```

## Rules agents get wrong — read first

1. **Every component is three keys `module` + `type` + `config`** — NOT a single
   dotted `name:`. echelon3 does `getattr(import_module(module), type)(**config)`.
   ```yaml
   net: { module: my_pkg.nets.my_net, type: MyNet, config: { channels: 32 } }   # RIGHT
   # net: { name: my_pkg.nets.my_net.MyNet, config: {} }                          # WRONG
   ```
2. **The CLI takes `--config-dir`/`-cd` + `--config-name`/`-cn`, not a YAML path:**
   ```bash
   echelon3 train -cd . -cn my_experiment device=cpu    # RIGHT
   # echelon3 train my_experiment.yaml --device cpu      # WRONG
   ```
3. `loss:` and `metrics:` are **lists** of `- name: { module, type, config, weight }`.
   `target:` is `{ path: ..., checkpoints_to_keep: ... }` (NOT under a `config:` block).
   `trainer:` also needs `module` + `type` + `config` (`echelon3.trainers.baseline` /
   `Trainer`).

## CLI

One executable, `echelon3`, with a subcommand per task. Run it from your repo root
(the current dir is added to `sys.path`, so configs can reference your own packages
via `module: my_pkg.nets.foo`).

```bash
echelon3 train    --config-dir configs --config-name my_experiment [overrides...]
echelon3 finetune --config-dir configs --config-name my_experiment [overrides...]
echelon3 run      --config-dir configs --config-name my_experiment [overrides...]
echelon3 evaluate --config-dir configs --config-name my_experiment [overrides...]
echelon3 export   --config-dir configs --config-name my_experiment [overrides...]
```

- `--config-dir` / `-cd`: directory of the config (default: current dir).
- `--config-name` / `-cn`: config file name, with or without `.yaml` (required).

## Config anatomy

YAML. Most sections describe a **component** as `module` + `type` + `config`
(constructor kwargs); echelon3 imports `module`, takes `type`, calls it with `config`.

```yaml
net:  { module: echelon3.nets.segmenter, type: Segmenter, config: { ... } }
data:
  train: { module: ..., type: ..., config: { ... } }
  test:  { module: ..., type: ..., config: { ... } }   # or a dict of named test sets
transform:                     # optional: augment (albumentations) + preprocess
  train: { augment: {...}, preprocess: {...} }
  test:  { augment: {...}, preprocess: {...} }
dataloaders:
  train: { module: torch.utils.data, type: DataLoader, config: { batch_size: 64, ... } }
  test:  { module: torch.utils.data, type: DataLoader, config: { batch_size: 128, ... } }
loss:                          # list of named losses (optional weight, default 1.0)
  - ce: { module: torch.nn, type: CrossEntropyLoss, config: {}, weight: 1.0 }
metrics:                       # optional
  - iou: { module: echelon3.metrics.segmentation, type: IoU, config: { num_classes: 3 } }
optimizer: { module: torch.optim, type: AdamW, config: { lr: 3e-4 } }
scheduler: { module: torch.optim.lr_scheduler, type: LinearLR, config: { ... } }  # optional
trainer:
  module: echelon3.trainers.baseline
  type: Trainer
  config:
    epochs: 100
    keep_best_on: { iou: { mode: directional, value: high } }   # optional; multi-metric = AND (see Gotchas)
    precision: auto            # auto|bf16|fp16|fp32 (auto = bf16 on capable GPUs)
    compile: false             # torch.compile
    times_to_validate_per_epoch: 1
target: { path: ${oc.env:OUT,./targets/run}, checkpoints_to_keep: 2 }
gpus: [0]                      # optional; default = all visible GPUs (multi -> DDP)
device: cuda                   # cuda|cpu. gpus wins for the index.
```

Optional sections: `transform`, `metrics`, `scheduler`, `mlops`, `gpus`,
`keep_best_on`. Required: `data`, `dataloaders`, `net`, `loss`, `optimizer`,
`trainer`, `target`. Every component needs its `config:` block (`config: {}` if the
constructor takes no args).

## Complete minimal example (verified end-to-end — copy and adapt)

Three files in one directory train a FashionMNIST classifier. Copy the exact shapes.

`fmnist_net.py`
```python
import torch.nn as nn
class FashionMNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(32, 10))
    def forward(self, x):
        return self.layers(x)
```

`fmnist_data.py`  (a dataset is a component; __init__ MUST accept augment/preprocess)
```python
import numpy as np, torch
from torch.utils.data import Dataset
from torchvision.datasets import FashionMNIST
class FashionMNISTData(Dataset):
    def __init__(self, split="train", limit=None, augment=None, preprocess=None):
        self.ds = FashionMNIST(root="./mnist", download=True, train=(split == "train"))
        self.n = len(self.ds) if limit is None else min(limit, len(self.ds))
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        img, label = self.ds[i]
        return torch.from_numpy(np.array(img)).float().unsqueeze(0) / 255.0, int(label)
```

`fmnist.yaml`  (note the exact shape of loss=list, trainer/target wrappers)
```yaml
net: { module: fmnist_net, type: FashionMNISTNet, config: {} }
data:
  train: { module: fmnist_data, type: FashionMNISTData, config: { split: train, limit: 512 } }
  test:  { module: fmnist_data, type: FashionMNISTData, config: { split: test, limit: 256 } }
dataloaders:
  train: { module: torch.utils.data, type: DataLoader, config: { batch_size: 64, num_workers: 0, shuffle: true, drop_last: true } }
  test:  { module: torch.utils.data, type: DataLoader, config: { batch_size: 64, num_workers: 0 } }
loss:
  - ce: { module: torch.nn, type: CrossEntropyLoss, config: {} }
optimizer: { module: torch.optim, type: AdamW, config: { lr: 0.001 } }
trainer: { module: echelon3.trainers.baseline, type: Trainer, config: { epochs: 1 } }
target: { path: ./out, checkpoints_to_keep: 1 }
```

Run from that directory:
```bash
echelon3 train -cd . -cn fmnist device=cpu
```

## CLI overrides

`key=value` positionals override config values (dotted paths, OmegaConf-typed):

```bash
# key=value overrides a value; +key=value adds one (key= also adds); ~key deletes one.
echelon3 train -cd configs -cn my_experiment \
    trainer.config.epochs=200 optimizer.config.lr=5e-5 \
    dataloaders.train.config.batch_size=256 \
    +trainer.config.compile=true \
    ~scheduler
```

Types infer: `epochs=200`→int, `lr=5e-5`→float, `compile=true`→bool, `x=null`→None,
`gpus=[0,1,2,3]`→list, `mode=reduce-overhead`→str, `path=${oc.env:OUT,/tmp}`→env
interpolation. `hydra.*` overrides are ignored.

## Config composition (`defaults:`)

```yaml
defaults:
  - base_experiment    # merge base_experiment.yaml at the root
  - net: resnet50      # merge net/resnet50.yaml under the `net:` key (config group)
  - _self_             # this file's own keys (wins; implicit-last if omitted)
```

Merged left-to-right; base configs may compose recursively.

## Multi-GPU, precision, compile

- **DDP**: `gpus=[0,1,2,3]` spawns one process per GPU (built-in launcher, no
  `torchrun`). `dataloaders.*.config.batch_size` is the **global** batch (split across
  ranks). `num_workers`/`prefetch_factor` are **per-rank** — watch host RAM.
- **Single GPU**: `gpus=[1]` runs on physical GPU 1; `device=cpu` forces CPU.
- **Precision**: bf16 autocast by default on capable GPUs; `trainer.config.precision:
  fp32` disables AMP; `fp16` uses a GradScaler.
- **compile**: `trainer.config.compile: true` (+ `compile_mode`) for launch-bound nets.

## Common recipes

- **Resume (continue a run):** automatic — `echelon3 train` continues if `target.path`
  already has checkpoints (optimizer + scheduler + epoch state are restored).
- **Start a fresh run from existing weights (plain `train`, NOT finetune):** set
  `net.weights: <ckpt>` plus a `weights_loader` component and run `echelon3 train`.
  Only the weights are loaded; optimizer and scheduler start fresh, so LR warmup /
  schedule run normally. This is the usual "init from a checkpoint and train" path.
  ```yaml
  net:
    module: my_pkg.nets
    type: MyNet
    config: { ... }
    weights: ./runs/base/checkpoint-40.tar   # triggers weight load in `echelon3 train`
  weights_loader:                            # strict full load
    module: echelon3.weightloaders.basic
    type: WeightsLoader
    config: {}
    # tolerant partial load (name+shape match, strict=False):
    # module: echelon3.weightloaders.partial
    # type: PartialWeightsLoader
    # config: { strip_prefix: "module." }    # optional: strip a key prefix
  ```
- **Finetune (only when you need freeze / per-layer LRs):** `echelon3 finetune` +
  `init_from.checkpoint` (warm-start), `finetune.freeze_patterns` (regex freeze),
  `finetune.head_only` / `finetune.param_groups`. If you just want to start from
  weights, use plain `train` above.
- **Evaluate**: `echelon3 evaluate` loads the latest checkpoint, runs `evaluator.metric`
  over `data.test`.
- **Export ONNX**: `echelon3 export` runs the `export` section (preprocess→net→
  postprocess into one graph). Needs `echelon3[export]`.

## Extending with your own code

Write a normal `nn.Module` / `Dataset` / loss in your repo and reference it by import
path — no registration:

```yaml
net: { module: my_pkg.nets.my_net, type: MyNet, config: { channels: 32 } }
```

Run `echelon3 <cmd>` from the repo root so `my_pkg` imports.

## Gotchas

- Every component block needs a `config:` (empty `{}` is fine).
- `gpus` sets the GPU index for single-GPU too; `device: cpu` overrides.
- `Ctrl-C` stops cleanly (exit 130, no traceback), including under DDP.
- Under DDP, custom `echelon3.metrics.base.Metric` subclasses only see rank 0's shard
  for keep-best unless they implement `dist_reduce()` (SUM all-reduce of their
  counters); torchmetrics aggregate automatically.
- `keep_best_on` with **several** metrics is combined with **AND**: a checkpoint is
  saved only when **every** listed metric improves on the same validation — there is no
  OR / weighted / priority mode, so multi-metric saves are rare. Each entry takes
  `mode: directional` (`value: high|low`) or `mode: tolerance` (`value:` + `direction:`);
  `metrics_on: { metric: test_set }` routes a metric to a specific test set.
- Config loading is OmegaConf (not Hydra); `${oc.env:VAR,default}` works; `defaults:`
  composes and `hydra:` blocks are ignored.
