"""Missing-modality sampling protocol, following Sec. 4.1 ("Setting of
Missing Modality") of the paper.

Given a missing rate eta and a scenario:
  - "missing_both": eta/2 of samples become text-only (image missing),
    eta/2 become image-only (text missing), (1-eta) remain complete.
  - "missing_text": eta of samples have their text removed (image-only),
    (1-eta) remain complete.
  - "missing_image": eta of samples have their image removed (text-only),
    (1-eta) remain complete.
"""
import numpy as np

COMPLETE = 0
MISSING_TEXT = 1   # text removed -> only image present
MISSING_IMAGE = 2  # image removed -> only text present

SCENARIOS = ("missing_both", "missing_text", "missing_image")


def assign_missing_types(n: int, scenario: str, missing_rate: float, rng: np.random.RandomState) -> np.ndarray:
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}, got {scenario!r}")
    if not 0.0 <= missing_rate <= 1.0:
        raise ValueError(f"missing_rate must be in [0, 1], got {missing_rate}")

    ids = np.full(n, COMPLETE, dtype=np.int64)
    perm = rng.permutation(n)

    if scenario == "missing_both":
        n_missing = int(round(n * missing_rate))
        n_text_missing = n_missing // 2
        n_image_missing = n_missing - n_text_missing
        ids[perm[:n_text_missing]] = MISSING_TEXT
        ids[perm[n_text_missing : n_text_missing + n_image_missing]] = MISSING_IMAGE
    elif scenario == "missing_text":
        n_missing = int(round(n * missing_rate))
        ids[perm[:n_missing]] = MISSING_TEXT
    else:  # missing_image
        n_missing = int(round(n * missing_rate))
        ids[perm[:n_missing]] = MISSING_IMAGE

    return ids
