"""Coordinate transforms between NED sim and scratch_built_daa training world."""

from __future__ import annotations

import numpy as np

from ..frames import Quaternion


def ned_position_to_training(position_ned: np.ndarray) -> np.ndarray:
    """
    Training world: x=north, y=east, z=up (altitude).

    NED: x=north, y=east, z=down.
    """
    p = np.asarray(position_ned, dtype=np.float64).reshape(3)
    return np.array([p[0], p[1], -p[2]], dtype=np.float64)


def training_position_to_ned(position_training: np.ndarray) -> np.ndarray:
    p = np.asarray(position_training, dtype=np.float64).reshape(3)
    return np.array([p[0], p[1], -p[2]], dtype=np.float64)


def ned_quaternion_to_training(quat: Quaternion) -> np.ndarray:
    """Body-to-world quaternion [w,x,y,z]; axes align for horizontal flight."""
    return quat.to_array().astype(np.float64)
