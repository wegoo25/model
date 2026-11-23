# SpeakerNet – speaker recognition & verification (PyTorch)

Research code for speaker embedding networks (MKRC family, BiMKRC, VOT/transformer variants, ECAPA-TDNN, ResNetSE, CAM++, RawNet3, TDNN, AMCRN) trained with classification (softmax / AM / AAM / focal) or metric-learning (angular prototypical, prototypical, GE2E, softmax+proto, triplet) objectives, and evaluated by EER / MinDCF on verification trial lists.

The original (pre-refactor) code is kept in git under the tag `pre-refactor`.

## Installation

```bash
python -m venv venv && source venv/bin/activate
# 1) PyTorch for your hardware (CUDA, ROCm or CPU): https://pytorch.org/get-started/locally/
# 2) everything else
pip install -r requirements.txt
```

Device selection is automatic (`--device auto`): CUDA, or AMD ROCm builds (which also expose `torch.cuda`), else CPU. Force with `--device cpu|cuda`. Mixed precision (`--mixedprec`) is only active on CUDA/ROCm; the loss is always computed in fp32.

## Layout

```
speakernet/
  config.py            argparse + YAML (CLI > YAML > defaults)
  models/              16 architectures + registry (build_model, MODELS, input_type)
  losses/              softmax, amsoftmax, aamsoftmax, focal, focal_am, angleproto, proto, ge2e, softmaxproto, triplet
  data/                features (fbank), augment (MUSAN/RIR), dataset, sampler
  engine/              speaker_net (model+loss), trainer, checkpoint, evaluation, runner (train/eval/DDP)
  metrics.py           EER, MinDCF; plotting.py  DET curves; utils.py  logging, device, seed
scripts/               train.py  evaluate.py  extract_embeddings.py  plot_det.py  prepare_data.py
configs/               one YAML per model/loss combination
lists/                 VoxCeleb download lists
tests/                 pytest suite (synthetic data)
```

## Usage

Prepare VoxCeleb (and optionally MUSAN/RIR): `python scripts/prepare_data.py --save_path data --download --extract --convert [--augment]`.

```bash
# train (+ periodic evaluation); resumes automatically from the latest checkpoint in <save_path>/model
python scripts/train.py --config configs/MKRC_AAM.yaml \
    --train_list lists/train_list.txt --train_path data/voxceleb2 \
    --test_list lists/veri_test2.txt --test_path data/voxceleb1 --save_path exps/MKRC_AAM

# multi-GPU (one process per GPU; add --world_size N --device cpu to try DDP without GPUs)
python scripts/train.py --config configs/MKRC_AAM.yaml ... --distributed

# evaluate a checkpoint (writes scores.npy, labels.npy, trials_scored.txt)
python scripts/evaluate.py --config configs/MKRC_AAM.yaml --initial_model exps/MKRC_AAM/model/model000000030.model \
    --test_list lists/veri_test2.txt --test_path data/voxceleb1 --save_path exps/MKRC_AAM

# extract embeddings for a list of wavs -> .pt dict {file: (num_eval, dim)}
python scripts/extract_embeddings.py --config configs/MKRC_AAM.yaml --initial_model <ckpt> \
    --wav_list files.txt --wav_root data/voxceleb1 --output emb.pt

# DET curves of several runs
python scripts/plot_det.py --run MKRC:exps/MKRC_AAM/model --run TDNN:exps/TDNN_AAM/model --output det.png
```

Command-line flags override YAML values. Metric-learning losses (`angleproto`, `proto`, `ge2e`, `softmaxproto`, `triplet`) need `--nPerSpeaker 2` (or more). Augmentation requires `--augment true --musan_path <musan_split> --rir_path <simulated_rirs>`.

Model notes: VOT, VOT_serial, VOT_serial_Memory and transformer hard-code a 301-frame sequence, so use `max_frames: 300` / `eval_frames: 300` (as the shipped configs do). Embedding size is 512 for all models except MKRC_GAT (256, set `nOut: 256`); only RawNet3 honours `--nOut`. The legacy model name `test_GAT` is an alias of `MKRC_GAT`.

## Checkpoint compatibility

Checkpoints are plain `state_dict`s with `__S__.*` (backbone) and `__L__.*` (loss) keys, named `model%09d.model`, identical to the original. Parameter names/shapes of all models are unchanged (verified by `tests/test_equivalence.py`), so original checkpoints load. The loader also accepts DDP `module.` prefixes, `{"model": backbone_state}` files, and skips size-mismatched tensors with a warning (e.g. a different `nClasses`).

## Behaviour changes vs. the original (bug fixes)

- `MKRC`: the in-place `+=` on views of the input broke `backward()` and corrupted the residual input; now cloned. Outputs of `MKRC` therefore differ from the original *forward* result, which was affected by the corruption.
- `VOT`, `transformer`: `alpha`/`gama` were uninitialised (`torch.FloatTensor(1)`, could be NaN); now initialised to 0.5. Loading an original checkpoint overrides them.
- `triplet`: pairwise-distance matrix computed explicitly (the original relied on broadcasting that current PyTorch rejects).
- Metric-learning losses get properly shaped `(batch, nPerSpeaker, dim)` input (the reshape was commented out).
- `--augment false` was parsed as true; YAML overrode CLI flags; `scores.npy` was overwritten with booleans; unlabeled trials got random labels; `--mixedprec` was a no-op; hard-coded `/media/...` paths and `.to("cuda")` calls were removed; `RawNet3_AM.yaml` used `aamsoftmax` instead of `amsoftmax`; `MKRC_GAT_AAM.yaml` had the wrong `nOut`.
- Preserved as-is: `BiMKRC`'s backward branch appends `combined` rather than `out` (looks unintended, but changing it would alter the published architecture).

## Tests

```bash
python -m pytest tests -q
```

Covers every model (forward/backward, checkpoint round-trip), numerical equivalence with the original code (models and losses, using `git archive pre-refactor`), metrics, data pipeline, all configs, train → resume → eval → embedding extraction, MUSAN/RIR augmentation (synthetic), and 2-process DDP over gloo on CPU.

Not covered: GPU/ROCm execution and AMP, NCCL, and training on the real VoxCeleb/MUSAN data (none were available during the refactor).

## Full-dataset run (not executed during the refactor)

The refactor was validated only on synthetic data on a CPU-only machine (2 cores, ~7 GB RAM, ~7 GB free disk). A full run was not possible there: VoxCeleb2 dev is ~300 GB of AAC audio (1,092,009 utterances, 5,994 speakers), VoxCeleb1 is ~40 GB, the downloads require registered credentials, and training needs a GPU for days.

**Requirements**
- ~500 GB free disk after wav conversion; `ffmpeg` and `wget`; VoxCeleb credentials; a CUDA or ROCm GPU.
- MUSAN and RIRS_NOISES only for `--augment true`.
- `train_list.txt` (`speaker_id relative/path.wav` per line) and `veri_test2.txt` (`label path1 path2`), downloaded from the VoxCeleb site (not in this repo).

**Commands**
```bash
python scripts/prepare_data.py --save_path data --user U --password P --download --extract --convert --augment
python scripts/train.py --config configs/MKRC_AAM.yaml --train_list lists/train_list.txt --train_path data/voxceleb2 \
    --test_list lists/veri_test2.txt --test_path data/voxceleb1 --nClasses 5994 --save_path exps/MKRC_AAM [--distributed --mixedprec]
```

**Expected output**
- Per epoch: `Epoch N, TEER/TAcc …, TLOSS …, LR …`; every `--eval_interval` epochs (default 3) and at the last epoch: `Epoch N, VEER x.xx, MinDCF y.yyy` plus `model%09d.model` and `.eer` in `<save_path>/model/`.
- `scripts/evaluate.py` additionally writes `scores.npy`, `labels.npy`, `trials_scored.txt`.
- Sanity check: a well-trained model typically reaches single-digit (or lower) EER on VoxCeleb1-O; EER near 50% or a flat TLOSS indicates a problem.
- Cheap pre-check before a full run: `--max_epoch 2 --max_seg_per_spk 20`.

## Test details

`python -m pytest tests -q` → last full run: 81 passed, 1 skipped (triplet equivalence: the original no longer runs on current PyTorch), ~95 s on CPU, torch 2.9.1.

| File | Covers |
|---|---|
| `test_models.py` | forward/backward of all 16 models, 3D/4D input, checkpoint round-trip, legacy checkpoint formats |
| `test_equivalence.py` | models and losses vs. original code from tag `pre-refactor` (needs git and that tag) |
| `test_components.py` | losses, metrics vs. brute force, fbank, dataset/sampler, all YAML configs |
| `test_pipeline.py` | train → resume → eval, CLI scripts, nPerSpeaker=2 |
| `test_distributed_augment.py` | MUSAN/RIR augmentation (synthetic), 2-process gloo DDP on CPU |

Not tested: GPU/ROCm execution, AMP, NCCL, real VoxCeleb/MUSAN data, trained-model EER.
