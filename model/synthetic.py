"""
Synthetic fallback dataset generator.

The real KDD Cup 99 dataset (kddcup.data_10_percent_corrected) is not bundled
in this repo (it's ~70MB and license-restricted for redistribution) and this
build environment has no network access to fetch it. This module generates a
schema-accurate, statistically-plausible stand-in so the full training and
inference pipeline can be exercised end-to-end.

IMPORTANT: a model trained on this synthetic data is a DEMO ONLY. Before
relying on accuracy numbers, run train.py with the real dataset (see
data/README.md for how to obtain it) — the pipeline is identical either way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common.preprocessing import RAW_COLUMNS

_RNG_SEED = 42

# One representative raw attack name per category, so the shared
# attack_type_for_label() mapping is exercised exactly as it would be on
# real data.
_LABEL_FOR_CLASS = {
    "normal": "normal",
    "dos": "neptune",
    "probe": "satan",
    "r2l": "guess_passwd",
    "u2r": "buffer_overflow",
}


def _clip_nonneg(arr):
    return np.clip(arr, 0, None)


def _generate_class(rng: np.random.Generator, cls: str, n: int) -> pd.DataFrame:
    zeros = np.zeros(n)

    if cls == "normal":
        duration = _clip_nonneg(rng.lognormal(2.0, 1.5, n))
        protocol = rng.choice(["tcp", "udp", "icmp"], n, p=[0.75, 0.20, 0.05])
        flag = rng.choice(["SF", "S0", "REJ"], n, p=[0.9, 0.05, 0.05])
        src_bytes = _clip_nonneg(rng.lognormal(5.0, 2.0, n))
        dst_bytes = _clip_nonneg(rng.lognormal(5.5, 2.0, n))
        logged_in = rng.choice([0, 1], n, p=[0.3, 0.7])
        count = _clip_nonneg(rng.normal(8, 5, n))
        srv_count = _clip_nonneg(rng.normal(8, 5, n))
        serror_rate = _clip_nonneg(rng.normal(0.01, 0.02, n))
        rerror_rate = _clip_nonneg(rng.normal(0.01, 0.02, n))
        same_srv_rate = np.clip(rng.normal(0.95, 0.05, n), 0, 1)
        diff_srv_rate = np.clip(rng.normal(0.03, 0.03, n), 0, 1)
        hot = zeros
        num_failed_logins = zeros
        root_shell = zeros
        su_attempted = zeros
        num_file_creations = zeros
        num_shells = zeros
        is_guest_login = rng.choice([0, 1], n, p=[0.97, 0.03])

    elif cls == "dos":
        duration = _clip_nonneg(rng.normal(0.2, 0.3, n))
        protocol = rng.choice(["tcp", "icmp"], n, p=[0.7, 0.3])
        flag = rng.choice(["S0", "SF", "REJ"], n, p=[0.85, 0.1, 0.05])
        src_bytes = _clip_nonneg(rng.lognormal(1.0, 1.5, n))
        dst_bytes = zeros
        logged_in = zeros
        count = _clip_nonneg(rng.normal(400, 100, n))
        srv_count = _clip_nonneg(rng.normal(400, 100, n))
        serror_rate = np.clip(rng.normal(0.95, 0.05, n), 0, 1)
        rerror_rate = _clip_nonneg(rng.normal(0.02, 0.03, n))
        same_srv_rate = np.clip(rng.normal(0.95, 0.05, n), 0, 1)
        diff_srv_rate = np.clip(rng.normal(0.02, 0.03, n), 0, 1)
        hot = zeros
        num_failed_logins = zeros
        root_shell = zeros
        su_attempted = zeros
        num_file_creations = zeros
        num_shells = zeros
        is_guest_login = zeros

    elif cls == "probe":
        duration = _clip_nonneg(rng.normal(1.0, 1.5, n))
        protocol = rng.choice(["tcp", "udp", "icmp"], n, p=[0.6, 0.2, 0.2])
        flag = rng.choice(["REJ", "S0", "SF"], n, p=[0.5, 0.3, 0.2])
        src_bytes = _clip_nonneg(rng.lognormal(2.0, 1.5, n))
        dst_bytes = zeros
        logged_in = zeros
        count = _clip_nonneg(rng.normal(50, 30, n))
        srv_count = _clip_nonneg(rng.normal(5, 4, n))
        serror_rate = np.clip(rng.normal(0.3, 0.2, n), 0, 1)
        rerror_rate = np.clip(rng.normal(0.4, 0.2, n), 0, 1)
        same_srv_rate = np.clip(rng.normal(0.2, 0.15, n), 0, 1)
        diff_srv_rate = np.clip(rng.normal(0.7, 0.2, n), 0, 1)
        hot = zeros
        num_failed_logins = zeros
        root_shell = zeros
        su_attempted = zeros
        num_file_creations = zeros
        num_shells = zeros
        is_guest_login = zeros

    elif cls == "r2l":
        duration = _clip_nonneg(rng.lognormal(1.5, 1.5, n))
        protocol = np.full(n, "tcp")
        flag = rng.choice(["SF", "RSTO"], n, p=[0.85, 0.15])
        src_bytes = _clip_nonneg(rng.lognormal(4.0, 1.5, n))
        dst_bytes = _clip_nonneg(rng.lognormal(3.0, 1.5, n))
        logged_in = rng.choice([0, 1], n, p=[0.4, 0.6])
        count = _clip_nonneg(rng.normal(2, 1.5, n))
        srv_count = _clip_nonneg(rng.normal(2, 1.5, n))
        serror_rate = _clip_nonneg(rng.normal(0.02, 0.03, n))
        rerror_rate = _clip_nonneg(rng.normal(0.05, 0.05, n))
        same_srv_rate = np.clip(rng.normal(0.6, 0.3, n), 0, 1)
        diff_srv_rate = np.clip(rng.normal(0.1, 0.1, n), 0, 1)
        hot = _clip_nonneg(rng.poisson(0.5, n))
        num_failed_logins = _clip_nonneg(rng.poisson(0.7, n))
        root_shell = zeros
        su_attempted = zeros
        num_file_creations = zeros
        num_shells = zeros
        is_guest_login = rng.choice([0, 1], n, p=[0.5, 0.5])

    else:  # u2r
        duration = _clip_nonneg(rng.lognormal(3.0, 1.5, n))
        protocol = np.full(n, "tcp")
        flag = np.full(n, "SF")
        src_bytes = _clip_nonneg(rng.lognormal(4.5, 1.5, n))
        dst_bytes = _clip_nonneg(rng.lognormal(4.0, 1.5, n))
        logged_in = np.ones(n)
        count = _clip_nonneg(rng.normal(1.5, 1.0, n))
        srv_count = _clip_nonneg(rng.normal(1.5, 1.0, n))
        serror_rate = zeros
        rerror_rate = zeros
        same_srv_rate = np.ones(n)
        diff_srv_rate = zeros
        hot = _clip_nonneg(rng.poisson(1.5, n))
        num_failed_logins = zeros
        root_shell = rng.choice([0, 1], n, p=[0.3, 0.7])
        su_attempted = rng.choice([0, 1], n, p=[0.5, 0.5])
        num_file_creations = _clip_nonneg(rng.poisson(1.0, n))
        num_shells = _clip_nonneg(rng.poisson(0.5, n))
        is_guest_login = zeros

    dst_host_count = _clip_nonneg(rng.normal(150, 80, n))
    dst_host_srv_count = _clip_nonneg(rng.normal(count.mean() + 5, 40, n))

    df = pd.DataFrame({
        "duration": duration,
        "protocol_type": protocol,
        "service": "other",
        "flag": flag,
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "land": zeros,
        "wrong_fragment": zeros,
        "urgent": zeros,
        "hot": hot,
        "num_failed_logins": num_failed_logins,
        "logged_in": logged_in,
        "num_compromised": zeros,
        "root_shell": root_shell,
        "su_attempted": su_attempted,
        "num_root": zeros,
        "num_file_creations": num_file_creations,
        "num_shells": num_shells,
        "num_access_files": zeros,
        "num_outbound_cmds": zeros,
        "is_host_login": zeros,
        "is_guest_login": is_guest_login,
        "count": count,
        "srv_count": srv_count,
        "serror_rate": serror_rate,
        "srv_serror_rate": serror_rate,
        "rerror_rate": rerror_rate,
        "srv_rerror_rate": rerror_rate,
        "same_srv_rate": same_srv_rate,
        "diff_srv_rate": diff_srv_rate,
        "srv_diff_host_rate": np.clip(rng.normal(0.05, 0.05, n), 0, 1),
        "dst_host_count": dst_host_count,
        "dst_host_srv_count": dst_host_srv_count,
        "dst_host_same_srv_rate": same_srv_rate,
        "dst_host_diff_srv_rate": diff_srv_rate,
        "dst_host_same_src_port_rate": np.clip(rng.normal(0.3, 0.2, n), 0, 1),
        "dst_host_srv_diff_host_rate": np.clip(rng.normal(0.05, 0.05, n), 0, 1),
        "dst_host_serror_rate": serror_rate,
        "dst_host_srv_serror_rate": serror_rate,
        "dst_host_rerror_rate": rerror_rate,
        "dst_host_srv_rerror_rate": rerror_rate,
        "target": _LABEL_FOR_CLASS[cls] + ".",
    })
    return df[RAW_COLUMNS]


def generate_synthetic_raw_dataframe(n_per_class: int = 4000, seed: int = _RNG_SEED) -> pd.DataFrame:
    """Return a raw, KDD-schema DataFrame (41 cols incl. 'target') with
    n_per_class synthetic rows for each of normal/dos/probe/r2l/u2r."""
    rng = np.random.default_rng(seed)
    frames = [_generate_class(rng, cls, n_per_class) for cls in _LABEL_FOR_CLASS]
    df = pd.concat(frames, ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)
