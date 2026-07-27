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
  `torchmetrics.Metric`. A metric that must span **several test sets at once** (e.g.
  retrieval: a query set against a gallery) subclasses `MultiDatasetMetric` instead — see
  the cookbook.

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
A metric that needs **several** of these sets together (e.g. retrieval: a *query* set
matched against a *gallery*) can't be expressed per-set — use a `MultiDatasetMetric` (see
the cookbook): it declares the sets it spans, receives the source-set name in each
`update`, and computes once over all of them.

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
  images together as the net input `(B_base, B_query)`. To `evaluate` a pair model, use its
  counterpart `echelon3.evaluators.pair.PairEvaluator` (since 0.10.5) — the generic
  `evaluator` calls `net(single_input)` and can't feed a pair net; `PairEvaluator` mirrors
  PairTrainer's forward and feeds the chosen metric the same `(prediction, gt)` it sees in
  training validation. Config: `evaluator: { module: echelon3.evaluators.pair, type:
  PairEvaluator, config: { return_features: false }, metric: <name> }`.

Common `trainer.config` keys: `epochs`, `times_to_validate_per_epoch`,
`keep_best_on`, `metrics_on`, `precision` (`auto|bf16|fp16|fp32`), `compile`
(+`compile_mode`), `high_is_better`, `float_labels`.

At start echelon3 runs an **initial validation before training** (the step-0 / loaded
checkpoint baseline the run must beat), then trains; each validation prints
`--> Trained epoch N: …% …, lr=…, <losses>` and `--> Evaluated [set]: <losses, metrics>`.

## Tabular / fit-predict models (the estimator path)

For models that are **fit once, not trained by SGD** — gradient-boosted trees
(CatBoost / XGBoost / LightGBM / sklearn) and tabular foundation models (TabPFN / TabICL /
TabFM / TabGPT, all sklearn-compatible fit/predict) — echelon3 has a **separate trainer
family**, still driven by `echelon3 train`.

- **Route by section:** an estimator config has a **`model:`** section (and **no `net:`**);
  a `net:` config is the image/SGD path. `echelon3 train` routes automatically.
- **No `optimizer` / `loss` / `dataloaders` / `scheduler`.** The objective/loss is a
  **hyperparameter of the model itself** (e.g. CatBoost `loss_function: Logloss`), set in
  `model.config` — there is no separate `loss:` section. Swap engines by changing only the
  `model:` block (share the rest via `defaults:`). The engine package
  (`catboost`/`xgboost`/`lightgbm`) must be pip-installed; the sklearn models need nothing extra.

Pieces (all in `echelon3` core):
- **Data** — `echelon3.data.tabular.TabularDataset`: reads csv/parquet/feather/json/tsv, a
  **SQL** connection (`source: sql`, `sql:`/`table:` + `connection:`), or an in-memory
  `frame:`; returns the whole `(X, y)`. `target:` is a column name **or a list** (multi-target).
- **Feature preprocessing** — optional **top-level** `feature_transform:` section (sibling of
  `model`/`data`), fit on train / applied to test (no leakage) and saved into the bundle:
  `feature_transform: { module: echelon3.data.tabular, type: TabularPreprocessor, config: { scale: true } }`.
  `TabularPreprocessor` is a declarative sklearn `ColumnTransformer` (impute/scale numeric +
  impute/encode categorical) — keeps engine-swap frictionless on categorical/NaN data (CatBoost
  handles those natively and can skip it; LogReg/TabPFN need it).
- **Trainer** — `echelon3.trainers.estimator.EstimatorTrainer` (single target) or
  `MultiTargetEstimatorTrainer` (a **list** of targets → one cloned model per target, fit only on
  rows where that target is present — NaN-masked; bundle `{target: model}`). `trainer.config`:
  `keep_best_on`, `fit_kwargs`, `eval_set` (pass first test set to `model.fit`), `use_categorical`.
  Note: here `keep_best_on` is **report-only** (a single fit → one saved bundle); it does **not**
  gate/select a best checkpoint like the SGD path's `keep_best_on`.
- **Metrics** — `echelon3.metrics.tabular`: classification AUC/Gini/KS/LogLoss/Accuracy and
  regression MAE/RMSE/R2/SpearmanR/PearsonR (trainer feeds `predict_proba` for classifiers,
  `predict` for regressors).
- **Inference** — `echelon3.inference.tabular` `load_bundle(path)` + `predict(bundle, data)`; the
  saved `.tar` bundle (model(s) + fitted feature_transform + feature names + target) is
  self-contained and `predict` re-applies the feature pipeline. Multi-target → `{target: preds}`.

```yaml
model: { module: catboost, type: CatBoostClassifier,
         config: { iterations: 500, depth: 6, loss_function: Logloss, eval_metric: AUC, verbose: false } }
data:
  train: { module: echelon3.data.tabular, type: TabularDataset, config: { path: train.csv, target: default } }
  test:  { module: echelon3.data.tabular, type: TabularDataset, config: { path: test.csv,  target: default } }
metrics:
  - auc:  { module: echelon3.metrics.tabular, type: AUC,  config: {} }
  - gini: { module: echelon3.metrics.tabular, type: Gini, config: {} }
trainer: { module: echelon3.trainers.estimator, type: EstimatorTrainer, config: { keep_best_on: [auc] } }
target: { path: ./out, checkpoints_to_keep: 1 }
# echelon3 train -cd . -cn my_tabular device=cpu   (no optimizer/loss/dataloaders)
```

Multi-target (e.g. several endpoints at once) needs **both** a list `target:` and the
multi-target trainer (regression metrics; one model per target):
```yaml
data:
  train: { module: echelon3.data.tabular, type: TabularDataset, config: { path: train.csv, target: [LogD, KSOL, HLM_CLint] } }
  test:  { module: echelon3.data.tabular, type: TabularDataset, config: { path: test.csv,  target: [LogD, KSOL, HLM_CLint] } }
metrics:
  - mae: { module: echelon3.metrics.tabular, type: MAE, config: {} }
  - r2:  { module: echelon3.metrics.tabular, type: R2,  config: {} }
trainer: { module: echelon3.trainers.estimator, type: MultiTargetEstimatorTrainer, config: {} }
# + a top-level feature_transform: if the model needs numeric features (e.g. SmilesFeaturizer for SMILES)
```

**Molecular / ADMET** (SMILES → property) is this same tabular path; the cheminformatics pieces
live in the **`echelon3_zoo`** package (`pip install "echelon3-zoo[molecular]"`, pulls rdkit):
`echelon3_zoo.molecular.SmilesFeaturizer` (RDKit descriptors + Morgan, as a `feature_transform`)
and `echelon3_zoo.molecular.MoleculeGraphDataset` + `echelon3_zoo.nets.mol_gcn.MolGCN` (a 2D GNN
that runs on the ordinary SGD `Trainer`). The **`[docking]`** extra adds protein-ligand **pose
prediction**: `echelon3_zoo.docking` (`DockingComplexDataset` / `PoseBustersDataset` /
`PDBBindDataset` for PDBBind/PoseBusters complexes, `DockingCollate`), metrics in
`echelon3_zoo.docking.metrics` (`RMSD` / `SuccessRate2A` / `CentroidRMSD`), loss
`echelon3_zoo.docking.loss.CoordMSELoss`, and net `echelon3_zoo.nets.egnn.EGNN` — on the
ordinary SGD `Trainer`, with `DockingCollate` set as the `dataloaders.*.config.collate_fn`
component for the variable-size complexes.

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

**Cross-dataset metric (`MultiDatasetMetric`)** — for a metric defined over several test
sets at once (retrieval, query-vs-gallery matching, anything needing cross-set context).
Declare the sets in `self.datasets`; `update` gets the **source-set name** each batch; the
trainer `reset()`s once, feeds every declared loader in one pass, then `compute()`s **once**
after all of them (preceded by a single `dist_reduce()` **under DDP** only; ordinary metrics
are untouched):
```python
import torch
from echelon3.metrics.base import MultiDatasetMetric, all_gather_cat

class RecallAt1(MultiDatasetMetric):
    """Recall@1 of a query set against a gallery, matched by label id."""
    def __init__(self, query_dataset, gallery_dataset):
        self.datasets = [query_dataset, gallery_dataset]   # test sets this metric spans
        self._q = query_dataset
        self.reset()
    def reset(self):
        self.qe, self.qi, self.ge, self.gi = [], [], [], []
    def update(self, predicted, target, dataset):          # `dataset` = source-set name
        (self.qe if dataset == self._q else self.ge).append(predicted)
        (self.qi if dataset == self._q else self.gi).append(target)
    def dist_reduce(self):                                 # DDP: gather full sets on every rank
        self.qe = [all_gather_cat(torch.cat(self.qe))]; self.qi = [all_gather_cat(torch.cat(self.qi))]
        self.ge = [all_gather_cat(torch.cat(self.ge))]; self.gi = [all_gather_cat(torch.cat(self.gi))]
    def compute(self):
        q, g = torch.cat(self.qe), torch.cat(self.ge)
        nn = (q @ g.t()).argmax(1)                         # nearest gallery row per query
        return (torch.cat(self.gi)[nn] == torch.cat(self.qi)).float().mean().item()
```
```yaml
data:
  test:
    queries: { module: my_pkg.data, type: Queries, config: {} }
    gallery: { module: my_pkg.data, type: Gallery, config: {} }
metrics:
  - recall1: { module: my_pkg.metrics, type: RecallAt1,
               config: { query_dataset: queries, gallery_dataset: gallery } }
trainer: { module: echelon3.trainers.baseline, type: Trainer,
           config: { keep_best_on: { recall1: { mode: directional, value: high } } } }
```
- Every name in `datasets` must be a declared test set and the roster non-empty, else
  validation raises (no silent empty compute). `keep_best_on` tracks it by name.
- Console: after the per-set `Evaluated […]` lines, one `Finalizing multi-dataset
  metrics…` / `Finalized multi-dataset metrics: …` summary.
- Ordinary vs cross-set signature differs: ordinary metrics get `update(pred, label)`,
  a `MultiDatasetMetric` gets `update(pred, label, dataset)`. Unlike counter metrics
  (which SUM-reduce), it `all_gather_cat`s its buffers so `compute()` sees the full sets
  (DistributedSampler may add a few padding duplicates — dedup by id if you need exactness).

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
  already has checkpoints (optimizer + scheduler + epoch state are restored). Because the
  checkpoint's optimizer state wins, the config's `lr` / `weight_decay` / schedule are
  **ignored on resume** while *other* config (e.g. `batch_size`) still applies — a silent
  mix. echelon3 warns for each optimizer hyperparameter the checkpoint overrides.
- **Continue from the weights but with NEW hyperparameters:** `trainer.config.reset: true`.
  It keeps the checkpoint **weights** but restarts optimizer / scheduler / epoch from the
  config, so a changed `lr` / `weight_decay` / schedule actually take effect (this is the
  direct answer to "resume, but with a different LR"). Not to be confused with
  `net.weights` below, which loads weights into a *fresh* run.
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
  metric by name from the `metrics` list; the evaluator runs it over each test set.
  `data.test`/`dataloaders.test` may be a single set OR a named dict of sets (same two
  formats as `train`, since 0.10.4): a named dict is evaluated per set (each paired with its
  `dataloaders.test` entry, a fresh metric per set) and prints `Validation [name] …`. Expects
  `transform.test.preprocess` to be present.
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
- A config with a **`model:`** section (and no `net:`) is the fit/predict **estimator** path
  (trees / tabular FMs) — no `optimizer`/`loss`/`dataloaders`; the objective lives in
  `model.config`. A `net:` config is the image/SGD path. `echelon3 train` routes on this.
- For **variable-size / graph batches** (sets, molecular complexes), set a custom collate as
  a component: `dataloaders.*.config.collate_fn: { module, type, config }` — echelon3 builds it
  into the DataLoader's `collate_fn` (a plain dict there would break the loader).
- `echelon3.data.basic.MultiPartDataset` (mix parts by `share`) must be paired with the
  `echelon3.dataloaders.multipart.MultiPartDataLoader` — its index is a `(part, sample)` tuple,
  so a plain `DataLoader` sends int indices and fails (echelon3 raises a clear error). Under DDP
  it rank-shards the largest part automatically (don't add a `DistributedSampler`).
- Config loading is OmegaConf (not Hydra); `${oc.env:VAR,default}` works; `defaults:`
  composes and `hydra:` blocks are ignored.
