import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """3x3 skew-symmetric matrix: skew(v) @ w == np.cross(v, w)."""
    v = np.asarray(v, dtype=float)
    return np.array([
        [0,     -v[2],  v[1]],
        [v[2],   0,    -v[0]],
        [-v[1],  v[0],  0   ]
    ])
