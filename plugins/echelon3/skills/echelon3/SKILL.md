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

## The component model (how everything is built)

One rule builds **every** section: `getattr(import_module(module), type)(**config)`.

- `module` is a dotted import path **or** a path to a `.py` file (so zoo/project code
  drops in without installation). `type` is the attribute (class or function) in it.
- If `type` is a **class**, echelon3 calls `Class(**config)`. If it is a **function**,
  echelon3 uses `functools.partial(fn, **config)` (e.g. a metric or collate function).
- The `config:` block is **optional everywhere** — omit it and the constructor is
  called with no extra args (`config: {}` and no `config:` are equivalent).

Beyond your `config:`, echelon3 **injects** extra constructor kwargs per section. Your
class must accept them (they arrive by keyword), or you get a `TypeError` at build time:

| Section | echelon3 calls | Injected beyond your `config:` |
|---|---|---|
| `net` | `Net(**config)` | — |
| `loss` (list item) | `Loss(**config)` | paired with `weight` (default `1.0`) |
| `metrics` (list item) | `Metric(**config)` (class) | — — for `train` a metric must be an **object** with `.to()`/`update`/`compute`/`reset` (subclass `Metric` or a `torchmetrics.Metric`); a bare function/`partial` only works under `evaluate` |
| `optimizer` | `Opt(params=…, **config)` | model params; `config.trainable_only: true` keeps only `requires_grad` params |
| `scheduler` | `Sched(optimizer=…, **config)` | the optimizer |
| `data.train` / `data.test` | `Dataset(**config, augment=…, preprocess=…)` | **`augment` + `preprocess`** (built from `transform:`) |
| `dataloaders.*` | `DataLoader(dataset=…, **config)` | the dataset; under DDP also a `DistributedSampler`, split `batch_size`, and a `worker_init_fn` |
| `trainer` | `Trainer(**config, net=…, optimizer=…, …)` | net, optimizer, both dataloaders, losses, metrics, scheduler, ckpt_manager, mlops_logger, device |
| `target` | `CheckpointManager(**config)` | — (`path`, `checkpoints_to_keep` passed verbatim) |
| `evaluator` (`evaluate`) | `Ev(**config, net=…, dataloader=…, metric=…, preprocess=…, postprocess=…)` | net, test loader, the chosen metric, pre/postprocess |
| `export.exporters.*` (`export`) | `Ex(**config, net=…, preprocess=…, postprocess=…)` | net, pre/postprocess |
| `runner` (`run`) | `Runner(**config)` then `.process(model, preprocess, postprocess)` | — |

## Component contract (signatures echelon3 calls — mismatches fail at runtime)

- **Dataset** — `__init__` **must accept `augment=` and `preprocess=`** (always injected),
  alongside your `config:` kwargs. `__getitem__` returns a `(source, label)` pair. You are
  responsible for **applying** augment/preprocess (echelon3 only constructs and hands them
  in):
  ```python
  # augment: an albumentations Compose that ENDS with ToTensorV2 -> returns a CHW tensor.
  # preprocess: an nn.Sequential (or None) applied to that tensor.
  img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)   # HWC uint8 numpy
  if self.augment is not None:
      img = self.augment(image=img)["image"]                # classification
      # segmentation: t = self.augment(image=img, mask=mask); img, label = t["image"], t["mask"].long()
      # detection:    t = self.augment(image=img, bboxes=boxes); img, label = t["image"], t["bboxes"]
  if self.preprocess is not None:
      img = self.preprocess(img)
  return img, label
  ```
  (Simple datasets may ignore augment/preprocess and return an already-made tensor — but
  the two kwargs must still be in the signature. See the FashionMNIST example below.)
- **Net** — called `net(source) -> predictions`, where `source` is the dataset's first
  return value after collation + device move. For the baseline trainer `predictions` is a
  tensor; multi-head / pair trainers use dict / tuple shapes (see **Trainers**).
- **Loss** — each named loss is called `loss(predictions, labels) -> scalar tensor`;
  echelon3 sums them weighted by `weight`. `labels` are cast to float first if the trainer
  has `float_labels: true`.
- **Metric** — torchmetrics-style: `metric.update(predictions, labels)` per batch, then
  `metric.compute() -> scalar`, and `reset()` between validations. Subclass
  `echelon3.metrics.base.Metric` for custom ones (see the cookbook), or use any
  `torchmetrics.Metric`.

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
    metrics_on: { iou: test }  # optional: route a metric to a named test set
    precision: auto            # auto|bf16|fp16|fp32 (auto = bf16 on capable GPUs)
    compile: false             # torch.compile
    times_to_validate_per_epoch: 1
    float_labels: false        # cast labels to float before the loss
target: { path: ${oc.env:OUT,./targets/run}, checkpoints_to_keep: 2 }
gpus: [0]                      # optional; default = all visible GPUs (multi -> DDP)
device: cuda                   # cuda|cpu. gpus wins for the index.
```

Optional sections: `transform`, `metrics`, `scheduler`, `mlops`, `gpus`,
`keep_best_on`. Required: `data`, `dataloaders`, `net`, `loss`, `optimizer`,
`trainer`, `target`. Every component needs its `config:` block (`config: {}` if the
constructor takes no args).

**Multiple test sets:** `data.test` (and `dataloaders.test`) may be a dict of named
sets instead of one component; the trainer validates each and prints one line per set.
```yaml
data:
  train: { ... }
  test:
    incidents: { module: ..., type: ..., config: { ... } }
    control:   { module: ..., type: ..., config: { ... } }
```

## Transforms (`augment` + `preprocess`)

`transform:` has `train:` and `test:` purposes, each with optional `augment:` and
`preprocess:`. echelon3 builds them and injects them into the matching dataset.

```yaml
transform:
  train:
    augment:                 # albumentations ops, applied in listed order; ToTensorV2 is auto-appended
      flip:  { module: albumentations, type: HorizontalFlip, config: { p: 0.5 } }
      blur:  { module: albumentations, type: GaussianBlur,   config: { p: 0.2 } }
    preprocess:              # nn.Module chain applied AFTER augment; each entry needs a `name`
      norm: { module: torchvision.transforms, type: Normalize, name: norm,
              config: { mean: [0.485,0.456,0.406], std: [0.229,0.224,0.225] } }
  test:
    augment: {}              # empty -> just ToTensorV2
```

- `augment` → an albumentations `A.Compose([... , ToTensorV2()])`. For detection, put
  `bbox_params` under the purpose's `config`.
- **Dtype/scale:** the auto-appended `ToTensorV2` only transposes HWC→CHW — it does **not**
  divide by 255 or cast to float (unlike `torchvision.ToTensor`). For real images add an
  `A.Normalize` op to `augment` (it converts to float and scales) **before** `ToTensorV2`,
  or cast/scale yourself in `__getitem__`; feeding a uint8 tensor into a float net or a
  `Normalize` preprocess otherwise mis-scales or errors.
- `preprocess` → a `torch.nn.Sequential`; each entry is a normal component **plus** a
  `name:` field (used as the layer key). Omit `transform:` entirely and datasets get a
  bare `ToTensorV2` augment and no preprocess.

## Trainers

- `echelon3.trainers.baseline.Trainer` — the default. Single-tensor `predictions`;
  `loss(pred, label)`, `metric.update(pred, label)`.
- `echelon3.trainers.multihead.MultiHeadTrainer` — for **dict-shaped** predictions and
  labels (multi-head nets); losses/metrics are keyed per head.
- `echelon3.trainers.pair.PairTrainer` — for **two-image** ("pair" / image-in-image)
  inputs: the dataset returns `((base, query), gt)` and a pair collate keeps the two
  images together as the net input `(B_base, B_query)`.

Common `trainer.config` keys: `epochs`, `times_to_validate_per_epoch`,
`keep_best_on`, `metrics_on`, `precision` (`auto|bf16|fp16|fp32`), `compile`
(+`compile_mode`), `high_is_better`, `float_labels`.

At start echelon3 runs an **initial validation before training** (the step-0 / loaded
checkpoint baseline the run must beat), then trains; each validation prints
`--> Trained epoch N: …% …, lr=…, <losses>` and `--> Evaluated [set]: <losses, metrics>`.

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

## Custom components cookbook

Every one of these is referenced from YAML by `module`/`type`/`config` and needs no
registration. Run `echelon3` from the repo root so the import path resolves.

**Net** — a plain `nn.Module`; `forward(source) -> predictions`. `config:` = its
`__init__` kwargs.
```python
class MyNet(nn.Module):
    def __init__(self, channels=32): ...
    def forward(self, x): return self.head(self.body(x))
```

**Dataset** — accept `augment`/`preprocess`, return `(source, label)` (apply them as in
the contract above if you take raw images):
```python
class MyDataset(Dataset):
    def __init__(self, root, split="train", augment=None, preprocess=None):
        self.augment, self.preprocess = augment, preprocess
        ...
    def __len__(self): ...
    def __getitem__(self, i):
        img = cv2.cvtColor(cv2.imread(self.paths[i]), cv2.COLOR_BGR2RGB)
        if self.augment is not None:   img = self.augment(image=img)["image"]  # add A.Normalize to augment: ToTensorV2 keeps uint8
        if self.preprocess is not None: img = self.preprocess(img)
        return img, self.labels[i]
```

**Loss** — any callable `loss(pred, label) -> scalar tensor` (an `nn.Module` or a
function). Weighted-summed with the others.

**Metric** — subclass `echelon3.metrics.base.Metric`. Keep counters as **tensors on the
data's device** and, under DDP, SUM-reduce **all** of them (validation is sharded per
rank, so every counter is partial — reducing only some gives a wrong global value):
```python
from echelon3.metrics.base import Metric, all_reduce_sum_
class Accuracy(Metric):
    def __init__(self): self.reset()
    def reset(self):
        self.correct = torch.zeros((), dtype=torch.float64)
        self.total   = torch.zeros((), dtype=torch.float64)
    def update(self, predicted, target):
        dev = target.device                          # follow the batch onto its device
        self.correct = self.correct.to(dev) + (predicted.argmax(-1) == target).sum()
        self.total   = self.total.to(dev)   + target.numel()
    def compute(self): return (self.correct / self.total.clamp(min=1)).item()
    def dist_reduce(self):                            # DDP: sum BOTH shard counters
        all_reduce_sum_(self.correct, self.total)    # (NCCL needs CUDA tensors — hence .to(dev))
```
(Any `torchmetrics.Metric` also works — it self-aggregates under DDP, no `dist_reduce`
needed. A plain function is NOT a valid training metric: the trainer calls `.to(device)`
on every metric, which a function lacks — see the metrics row in the injection table.)

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
  ranks, so it must divide by the GPU count). `num_workers`/`prefetch_factor` are
  **per-rank** — watch host RAM.
- **Single GPU**: `gpus=[1]` runs on physical GPU 1; `device=cpu` forces CPU.
- **Precision**: bf16 autocast by default on capable GPUs; `trainer.config.precision:
  fp32` disables AMP; `fp16` uses a GradScaler.
- **compile**: `trainer.config.compile: true` (+ `compile_mode`) for launch-bound nets.

## Checkpoints, resume, finetune

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
  weights_loader:
    # Use PartialWeightsLoader for an echelon3 checkpoint .tar: it unwraps the
    # {model_state_dict, optimizer_state_dict, ...} tar and loads matching name+shape
    # keys (strict=False), so it also survives architecture changes.
    module: echelon3.weightloaders.partial
    type: PartialWeightsLoader
    config: { strip_prefix: "module." }      # optional: strip a key prefix (e.g. DDP "module.")
    # WeightsLoader (echelon3.weightloaders.basic) does a STRICT load of a RAW state_dict
    # file (torch.save(model.state_dict())) — it does NOT understand an echelon3 .tar.
  ```
- **Finetune (only when you need freeze / per-layer LRs):** `echelon3 finetune` +
  `init_from.checkpoint` (warm-start), `finetune.freeze_patterns` (regex freeze),
  `finetune.head_only` / `finetune.param_groups`. If you just want to start from
  weights, use plain `train` above.

## evaluate / export / run

- **`evaluate`** — score the latest checkpoint. Reads `net`, `target` (loads latest
  ckpt), `transform.test`, `data.test`, `dataloaders.test`, `metrics`, and
  `evaluator: { module, type, config, metric: <name> }` — `evaluator.metric` picks **one**
  metric by name from the `metrics` list; the evaluator runs it over the test set.
  Note: unlike `train`, `evaluate` requires a **single** `data.test`/`dataloaders.test`
  (not a named-dict of sets) and expects `transform.test.preprocess` to be present.
- **`export`** — serialize to ONNX (needs `echelon3[export]`). Reads `net`, optional
  `target` (load weights), and an `export:` section with optional `preprocess:` /
  `postprocess:` chains and an `exporters:` dict of named exporter components; each
  exporter's `.export()` writes one graph (preprocess→net→postprocess).
- **`run`** — inference over images / video. Reads `net`, optional `target`, reuses
  `export.preprocess` / `export.postprocess` (and optional `export.wrapper`), plus a
  `runner:` component whose `.process(model, preprocess, postprocess)` does the work.
  Honors `precision` / `tf32` / `cudnn_benchmark`.

## Gotchas

- Every component block needs a `config:` (empty `{}` is fine).
- Datasets **must** accept `augment=`/`preprocess=` even if unused (echelon3 always
  passes them); a preprocess entry needs a `name:` field.
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
