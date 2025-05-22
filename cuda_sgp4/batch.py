from __future__ import annotations

from datetime import datetime
from typing import Iterable, Tuple, Sequence

import numpy as np
from numba import cuda

from .src.cuda_sgp4 import propagate_orbit
from .src.initialize_tle_arrays import initialize_tle_arrays_from_lines


TLELines = Sequence[Tuple[str, str]]


def cuda_sgp4_batch(
    tle_lines: Iterable[Tuple[str, str]],
    timestep_length_in_seconds: int,
    total_sim_seconds: int,
    start_time: datetime,
) -> Tuple[np.ndarray, np.ndarray]:
    """Propagate a batch of TLEs on the GPU.

    Parameters
    ----------
    tle_lines:
        Iterable of ``(line1, line2)`` pairs.
    timestep_length_in_seconds:
        Length of each time step in seconds.
    total_sim_seconds:
        Total propagation duration in seconds.
    start_time:
        Start epoch of the propagation.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Positions and velocities with shape ``(n_sats, n_steps, 3)``.
    """
    tle_arrays, _ = initialize_tle_arrays_from_lines(list(tle_lines), start_time)
    num_satellites = tle_arrays.shape[0]
    total_timesteps = total_sim_seconds // timestep_length_in_seconds

    d_tles = cuda.to_device(tle_arrays.astype(np.float64))
    d_r = cuda.device_array((3, num_satellites, total_timesteps), dtype=np.float64)
    d_v = cuda.device_array((3, num_satellites, total_timesteps), dtype=np.float64)

    threads_per_block = 256
    blocks_per_grid = (num_satellites + (threads_per_block - 1)) // threads_per_block

    propagate_orbit[blocks_per_grid, threads_per_block](
        d_tles, d_r, d_v, total_timesteps, timestep_length_in_seconds
    )

    r = d_r.copy_to_host()
    v = d_v.copy_to_host()

    positions = np.transpose(r, (1, 2, 0))
    velocities = np.transpose(v, (1, 2, 0))

    return positions, velocities
