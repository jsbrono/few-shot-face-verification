# Few-shot face verification

Given one or more reference photos of a person and a new photo, is it the same person? This
repo trains Siamese networks for that task on CASIA-WebFace and evaluates them on the standard
LFW benchmark. It compares:

- **two ways to learn**: metric learning (contrastive or batch-hard triplet loss) against a
  learned verification head trained with binary cross-entropy;
- **two backbones**: an ImageNet-pretrained ResNet-18 (11M parameters) against a small CNN
  trained from scratch (about 100k);
- **one reference vs several**: every model accepts a variable number of reference images per
  person, so we can measure how much extra references help.

## How it works

Every image goes through the same embedding network (the "Siamese" part): a backbone, a linear
projection to 128 dimensions, then L2 normalisation. A query is scored against its references by
one of two heads.

**Cosine head** (used with contrastive and triplet training). The reference embeddings are
averaged and the score is the cosine similarity between that average and the query. Training
shapes the embedding space directly:

- *Contrastive loss* pulls every same-person pair in a batch together and pushes
  different-person pairs at least a margin apart.
- *Batch-hard triplet loss* (Hermans et al., 2017) takes each image as an anchor and uses only
  its hardest triplet in the batch: the furthest image of the same person and the closest image
  of anyone else.

Both losses use every pair in a batch of P people x K images, which gives far more training
signal per forward pass than fixed pairs.

**Differential head** (trained with binary cross-entropy). A small MLP looks at two features:
how far the query is from the references (mean of `|q - r_i|`), and how much the references vary
among themselves (mean of `|r_i - r_j|`). The second tells the model how much variation is
normal for this person, for example across lighting or age. Training uses episodes of 1–5
references and a query, re-drawn every epoch.

## Evaluation

- **LFW, 10-fold protocol**: 6,000 pairs in 10 folds. The decision threshold for each fold is
  chosen on the other nine, so the accuracy is not tuned on the data it is measured on. ROC AUC
  and the true accept rate at 1% false accept rate are reported too.
- **Held-out CASIA identities**: 5% of people are kept out of training. They are used to pick the
  best epoch, and to measure ROC AUC with exactly 1, 3 and 5 reference images.

## Results

20 epochs on CASIA-WebFace, one seed each. LFW numbers use
the standard 6,000 pairs and 10 folds, with each fold's threshold chosen on the other nine. The
last column is ROC AUC on CASIA identities held out from training, comparing one probe photo
against 1, 3 or 5 reference photos of the claimed person.

| Config | LFW accuracy | LFW AUC | TAR @ 1% FAR | AUC, 1 / 3 / 5 refs |
|---|---|---|---|---|
| `resnet18-triplet` | **91.7% ± 1.3** | **0.973** | **0.691** | **0.966 / 0.986 / 0.990** |
| `resnet18-contrastive` | 85.3% ± 1.3 | 0.929 | 0.222 | 0.923 / 0.947 / 0.956 |
| `resnet18-differential` | 80.4% ± 1.8 | 0.888 | 0.213 | 0.879 / 0.926 / 0.930 |
| `smallcnn-contrastive` | 84.3% ± 2.1 | 0.921 | 0.274 | 0.909 / 0.931 / 0.944 |
| `smallcnn-triplet` | 62.0% ± 1.9 | 0.645 | 0.021 | 0.694 / 0.689 / 0.715 |

- **Batch-hard triplet loss is clearly best**, especially at strict thresholds: at 1% false
  accepts it verifies 69% of genuine pairs, against about 22% for the other two.
- **Extra reference photos help every model.** Going from one to five references removes about
  70% of the triplet model's remaining AUC error (0.966 to 0.990).
- **The triplet loss is a poor progress signal.** It sits at the margin (0.2) for the first 12
  epochs while validation AUC climbs from 0.55 to 0.97: on the hardest positive and negative in
  each batch the margin is barely violated, even as the embedding improves for typical pairs.
  Watch validation AUC, not the loss (`runs/<config>/metrics.json`).
- The learned verification head (`differential`) is the weakest, probably because it learns the distance
  function as well as the embedding, from the same data.
- **A small CNN gets close with contrastive loss.** Trained from scratch with about 100k
  parameters (100 times fewer than ResNet-18), it reaches 84.3% on LFW against the pretrained
  ResNet-18's 85.3% with the same loss. It is slightly better at strict thresholds (TAR 0.274 vs
  0.222) and slightly worse with several references (AUC 0.944 vs 0.956 with five).
- **The small CNN does not learn with batch-hard triplet loss.** Validation AUC stays between
  0.52 and 0.69 for all 20 epochs (62% on LFW). Batch-hard mining picks the hardest pairs in each batch,
  which a randomly initialised network cannot yet separate; the pretrained ResNet-18 starts with
  useful features and gets past this. Pretraining, a softer mining scheme, or a warm-up with
  contrastive loss would be the next things to try.

## Usage

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch torchvision                        # pick your CUDA build from pytorch.org
pip install -e ".[dev]"
```

**Data.** Put the datasets under `data/` (or change the paths in the config):

```
data/
  casia-webface/<person>/<image>.jpg        CASIA-WebFace, one folder per identity
  lfw-deepfunneled/<Name>/<Name>_0001.jpg   LFW (deep-funnelled)
  pairs.csv                                 LFW's View 2 pairs list
```

Both datasets are on Kaggle (for example `ntl0601/casia-webface` and
`jessicali9530/lfw-dataset`). The pairs file can be the Kaggle mirror's `pairs.csv` or LFW's
official `pairs.txt` (set `data.lfw_pairs`).

**Train and evaluate:**

```bash
python -m faceverify.train --config configs/resnet18-triplet.yaml
python -m faceverify.train --config configs/resnet18-differential.yaml --seed 1

# Re-evaluate a saved model, and save its most confident LFW mistakes
python -m faceverify.evaluate runs/resnet18-triplet-seed0/model.pt --errors results/errors.png

pytest
```

Each run writes `config.yaml`, `metrics.json` (training loss and validation AUC per epoch),
`model.pt` (best epoch) and `eval.json` to `runs/<config>-seed<seed>/`.

## Layout

```
configs/                   one YAML file per experiment
src/faceverify/
  models.py                backbones, embedding network, cosine and differential heads
  losses.py                contrastive and batch-hard triplet losses
  data.py                  identity folders and splits, P x K sampler, episodes, LFW pairs
  train.py                 training CLI
  evaluate.py              metrics (k-fold accuracy, ROC AUC, TAR@FAR) and evaluation CLI
  config.py                config dataclasses and checkpoints
tests/                     unit tests, plus end-to-end runs on synthetic identities
```

## Limitations

- Some people appear in both CASIA-WebFace and LFW. Published face-recognition results usually
  remove these overlapping identities from training; this code does not, so LFW numbers may be
  slightly optimistic.
- Faces are used as provided by each dataset (no landmark alignment), at 112x112.
