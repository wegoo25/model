"""Shared synthetic-data fixtures (no real datasets are needed)."""
import os
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SR = 16000


@pytest.fixture(scope="session")
def toy_data(tmp_path_factory):
    """6 speakers x 4 short wavs (1.5-2.5 s of tone + noise) plus a trial list."""
    root = tmp_path_factory.mktemp("toy")
    rng = np.random.RandomState(0)
    train_lines, files = [], {}
    for spk in range(6):
        for utt in range(4):
            n = int(SR * (1.5 + 0.25 * utt))
            t = np.arange(n) / SR
            wav = 0.3 * np.sin(2 * np.pi * (200 + 80 * spk) * t) + 0.05 * rng.randn(n)
            rel = f"id{spk:02d}/utt{utt}.wav"
            os.makedirs(root / f"id{spk:02d}", exist_ok=True)
            sf.write(root / rel, wav.astype(np.float32), SR)
            train_lines.append(f"id{spk:02d} {rel}")
            files.setdefault(spk, []).append(rel)
    (root / "train.txt").write_text("\n".join(train_lines) + "\n")

    trials = []
    for spk in range(6):
        trials.append(f"1 {files[spk][0]} {files[spk][1]}")
        trials.append(f"0 {files[spk][0]} {files[(spk + 1) % 6][2]}")
    (root / "trials.txt").write_text("\n".join(trials) + "\n")
    (root / "files.txt").write_text("\n".join(f[0] for f in files.values()) + "\n")
    return root
