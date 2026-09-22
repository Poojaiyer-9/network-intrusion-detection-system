"""
Shared preprocessing logic for the Network Intrusion Detection System.

This mirrors the pipeline in network_intrusion.ipynb exactly (column list,
attack-category mapping, categorical encodings, dropped/correlated columns)
so that training (model/train.py) and inference (api/predict.py) can never
drift apart. Single source of truth for the 30-feature schema.
"""
from __future__ import annotations

# --- Raw KDD Cup 99 schema (41 features + target) -------------------------

RAW_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "target",
]

# 22 known attack names (10%-subset) -> 4 attack categories, plus normal.
ATTACK_TYPES = {
    "normal": "normal",
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos",
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe", "satan": "probe",
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l", "multihop": "r2l",
    "phf": "r2l", "spy": "r2l", "warezclient": "r2l", "warezmaster": "r2l",
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r", "rootkit": "u2r",
}

CLASS_LABELS = ["normal", "dos", "probe", "r2l", "u2r"]

PROTOCOL_MAP = {"icmp": 0, "tcp": 1, "udp": 2}
FLAG_MAP = {
    "SF": 0, "S0": 1, "REJ": 2, "RSTR": 3, "RSTO": 4, "SH": 5,
    "S1": 6, "S2": 7, "RSTOS0": 8, "S3": 9, "OTH": 10,
}

HIGHLY_CORRELATED = [
    "num_root", "srv_serror_rate", "srv_rerror_rate",
    "dst_host_srv_serror_rate", "dst_host_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "dst_host_same_srv_rate",
]
ZERO_VARIANCE = ["is_host_login", "num_outbound_cmds"]
DROPPED_EXTRA = ["service"]

_DROPPED = set(HIGHLY_CORRELATED) | set(ZERO_VARIANCE) | set(DROPPED_EXTRA)

# The final 30-feature schema used for scaling/training/inference, in a
# fixed, stable order derived from RAW_COLUMNS (matches the notebook's
# X_train.shape == (*, 30) after all drops).
FINAL_FEATURE_ORDER = [
    c for c in RAW_COLUMNS if c not in _DROPPED and c != "target"
]

CATEGORICAL_FEATURES = {"protocol_type": PROTOCOL_MAP, "flag": FLAG_MAP}


class InvalidInputError(ValueError):
    """Raised when a prediction request payload is malformed or out of range."""


def attack_type_for_label(raw_label: str) -> str:
    """Map a raw KDD target label (e.g. 'neptune.') to its attack category."""
    name = raw_label[:-1] if raw_label.endswith(".") else raw_label
    try:
        return ATTACK_TYPES[name]
    except KeyError as exc:
        raise InvalidInputError(f"Unknown attack label: {raw_label!r}") from exc


def encode_record(raw: dict) -> list:
    """
    Convert a dict of the 30 final feature names (protocol_type/flag as
    human-readable strings, everything else numeric) into an ordered numeric
    feature vector matching FINAL_FEATURE_ORDER, ready for the fitted scaler.

    Raises InvalidInputError with a descriptive message on missing fields,
    unknown categorical values, or non-numeric values, rather than letting a
    bad request silently poison the model with NaNs.
    """
    missing = [f for f in FINAL_FEATURE_ORDER if f not in raw]
    if missing:
        raise InvalidInputError(f"Missing required field(s): {', '.join(missing)}")

    vector = []
    for feature in FINAL_FEATURE_ORDER:
        value = raw[feature]
        if feature in CATEGORICAL_FEATURES:
            mapping = CATEGORICAL_FEATURES[feature]
            key = str(value).strip()
            if key not in mapping:
                allowed = ", ".join(sorted(mapping))
                raise InvalidInputError(
                    f"Invalid value {value!r} for '{feature}'. Allowed: {allowed}"
                )
            vector.append(float(mapping[key]))
        else:
            try:
                vector.append(float(value))
            except (TypeError, ValueError) as exc:
                raise InvalidInputError(
                    f"Field '{feature}' must be numeric, got {value!r}"
                ) from exc
    return vector


def dataframe_to_features_and_labels(df):
    """
    Reproduce the notebook's preprocessing pipeline on a raw KDD-schema
    DataFrame (41 columns incl. 'target'). Returns (X, y) where X is a
    DataFrame restricted to FINAL_FEATURE_ORDER with protocol_type/flag
    already numerically encoded, and y is the 'Attack Type' Series.
    """
    df = df.copy()
    df["Attack Type"] = df["target"].apply(attack_type_for_label)
    df = df.drop(columns=["target"]).dropna(axis="columns")

    y = df[["Attack Type"]].values.ravel()
    X = df.drop(columns=["Attack Type"])

    X["protocol_type"] = X["protocol_type"].map(PROTOCOL_MAP)
    X["flag"] = X["flag"].map(FLAG_MAP)

    X = X.drop(columns=list(_DROPPED), errors="ignore")
    X = X[FINAL_FEATURE_ORDER]
    return X, y
