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
