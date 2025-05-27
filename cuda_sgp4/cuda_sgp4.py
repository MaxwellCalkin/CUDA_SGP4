"""CUDA accelerated SGP4 propagation interface."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Sequence, Tuple, Union

import numpy as np
from numba import cuda

from .src.cuda_sgp4 import propagate_orbit, _transpose_arrays
from .src.initialize_tle_arrays import initialize_tle_arrays_from_lines

TLELines = Sequence[Tuple[str, str]]


def cuda_sgp4(
    tle_lines: Iterable[Tuple[str, str]],
    timestep_length_in_seconds: int,
    total_sim_seconds: int,
    start_time: datetime,
    return_device_arrays: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]]:
    """Propagate a collection of TLEs on the GPU.

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
    return_device_arrays:
        If True, returns CUDA device arrays that remain on the GPU.
        If False (default), returns NumPy arrays copied to host memory.
        Use True when you want to do further GPU processing on the results.

    Returns
    -------
    Union[Tuple[np.ndarray, np.ndarray], Tuple[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]]
        Positions and velocities with shape ``(n_sats, n_steps, 3)``.
        If return_device_arrays=False: Returns NumPy arrays on host.
        If return_device_arrays=True: Returns CUDA device arrays on GPU.
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

    if return_device_arrays:
        # Return device arrays directly - data stays on GPU
        # Transpose to match expected output format (n_sats, n_steps, 3)
        d_r_transposed = cuda.device_array((num_satellites, total_timesteps, 3), dtype=np.float64)
        d_v_transposed = cuda.device_array((num_satellites, total_timesteps, 3), dtype=np.float64)
        
        # Use a simple kernel to transpose the arrays on GPU
        _transpose_arrays[blocks_per_grid, threads_per_block](d_r, d_v, d_r_transposed, d_v_transposed, num_satellites, total_timesteps)
        
        return d_r_transposed, d_v_transposed
    else:
        # Original behavior - copy to host and return NumPy arrays
        r = d_r.copy_to_host()
        v = d_v.copy_to_host()

        positions = np.transpose(r, (1, 2, 0))
        velocities = np.transpose(v, (1, 2, 0))

        return positions, velocities


def device_arrays_to_host(
    device_positions: cuda.devicearray.DeviceNDArray,
    device_velocities: cuda.devicearray.DeviceNDArray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert CUDA device arrays to NumPy host arrays.
    
    This is a convenience function for when you've finished GPU processing
    and want to copy the results back to host memory.
    
    Parameters
    ----------
    device_positions:
        CUDA device array containing positions with shape (n_sats, n_steps, 3).
    device_velocities:
        CUDA device array containing velocities with shape (n_sats, n_steps, 3).
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        NumPy arrays on host with positions and velocities.
    """
    return device_positions.copy_to_host(), device_velocities.copy_to_host()


def get_device_array_info(device_array: cuda.devicearray.DeviceNDArray) -> dict:
    """Get information about a CUDA device array.
    
    Parameters
    ----------
    device_array:
        CUDA device array to inspect.
        
    Returns
    -------
    dict
        Dictionary containing shape, dtype, size, and memory info.
    """
    return {
        'shape': device_array.shape,
        'dtype': device_array.dtype,
        'size': device_array.size,
        'nbytes': device_array.nbytes,
        'device_id': device_array.gpu_data.device.id if hasattr(device_array.gpu_data, 'device') else 'unknown'
    }
