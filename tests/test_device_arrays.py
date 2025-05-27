"""Test device array functionality for CUDA_SGP4."""

import numpy as np
from datetime import datetime
from numba import cuda
import pytest
import warnings
from numba.core.errors import NumbaPerformanceWarning

from cuda_sgp4 import cuda_sgp4, device_arrays_to_host, get_device_array_info

# Suppress CUDA performance warnings for test cases with small workloads
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)


def _make_simple_tle() -> tuple[str, str]:
    """Generate a simple test TLE."""
    line1 = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753"
    line2 = "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667"
    return line1, line2


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA is not available")
def test_device_arrays_basic():
    """Test basic device array functionality."""
    tle_lines = [_make_simple_tle()]
    start_time = datetime(2000, 1, 1)
    timestep_seconds = 60
    total_seconds = 180  # 3 timesteps
    
    # Test device arrays
    device_pos, device_vel = cuda_sgp4(
        tle_lines,
        timestep_length_in_seconds=timestep_seconds,
        total_sim_seconds=total_seconds,
        start_time=start_time,
        return_device_arrays=True
    )
    
    # Verify they are device arrays
    assert isinstance(device_pos, cuda.devicearray.DeviceNDArray)
    assert isinstance(device_vel, cuda.devicearray.DeviceNDArray)
    
    # Check shapes
    assert device_pos.shape == (1, 3, 3)  # (n_sats, n_steps, 3)
    assert device_vel.shape == (1, 3, 3)
    
    # Check dtypes
    assert device_pos.dtype == np.float64
    assert device_vel.dtype == np.float64


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA is not available")
def test_device_vs_host_arrays():
    """Test that device arrays and host arrays contain the same data."""
    tle_lines = [_make_simple_tle()]
    start_time = datetime(2000, 1, 1)
    timestep_seconds = 60
    total_seconds = 180  # 3 timesteps
    
    # Get host arrays (default behavior)
    host_pos, host_vel = cuda_sgp4(
        tle_lines,
        timestep_length_in_seconds=timestep_seconds,
        total_sim_seconds=total_seconds,
        start_time=start_time,
        return_device_arrays=False
    )
    
    # Get device arrays
    device_pos, device_vel = cuda_sgp4(
        tle_lines,
        timestep_length_in_seconds=timestep_seconds,
        total_sim_seconds=total_seconds,
        start_time=start_time,
        return_device_arrays=True
    )
    
    # Convert device arrays to host
    device_pos_host, device_vel_host = device_arrays_to_host(device_pos, device_vel)
    
    # Compare the results
    np.testing.assert_allclose(host_pos, device_pos_host, rtol=1e-12)
    np.testing.assert_allclose(host_vel, device_vel_host, rtol=1e-12)


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA is not available")
def test_device_array_info():
    """Test the device array info function."""
    tle_lines = [_make_simple_tle()]
    start_time = datetime(2000, 1, 1)
    timestep_seconds = 60
    total_seconds = 120  # 2 timesteps
    
    device_pos, device_vel = cuda_sgp4(
        tle_lines,
        timestep_length_in_seconds=timestep_seconds,
        total_sim_seconds=total_seconds,
        start_time=start_time,
        return_device_arrays=True
    )
    
    # Test info function
    pos_info = get_device_array_info(device_pos)
    vel_info = get_device_array_info(device_vel)
    
    # Check info contents
    assert pos_info['shape'] == (1, 2, 3)
    assert pos_info['dtype'] == np.float64
    assert pos_info['size'] == 6  # 1 * 2 * 3
    assert pos_info['nbytes'] == 48  # 6 * 8 bytes per float64
    
    assert vel_info['shape'] == (1, 2, 3)
    assert vel_info['dtype'] == np.float64


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA is not available")
def test_multiple_satellites_device_arrays():
    """Test device arrays with multiple satellites."""
    # Create multiple TLEs
    tle_lines = []
    for i in range(3):
        line1 = f"1 {10000+i:05d}U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753"
        line2 = f"2 {10000+i:05d}  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667"
        tle_lines.append((line1, line2))
    
    start_time = datetime(2000, 1, 1)
    timestep_seconds = 60
    total_seconds = 240  # 4 timesteps
    
    device_pos, device_vel = cuda_sgp4(
        tle_lines,
        timestep_length_in_seconds=timestep_seconds,
        total_sim_seconds=total_seconds,
        start_time=start_time,
        return_device_arrays=True
    )
    
    # Check shapes for multiple satellites
    assert device_pos.shape == (3, 4, 3)  # (3 sats, 4 steps, 3 coords)
    assert device_vel.shape == (3, 4, 3)
    
    # Convert to host and verify data integrity
    host_pos, host_vel = device_arrays_to_host(device_pos, device_vel)
    
    # Basic sanity checks
    assert not np.any(np.isnan(host_pos))
    assert not np.any(np.isnan(host_vel))
    assert np.all(np.isfinite(host_pos))
    assert np.all(np.isfinite(host_vel))


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA is not available")
def test_backward_compatibility():
    """Test that the default behavior (return_device_arrays=False) still works."""
    tle_lines = [_make_simple_tle()]
    start_time = datetime(2000, 1, 1)
    timestep_seconds = 60
    total_seconds = 120
    
    # Test default behavior (should return host arrays)
    pos, vel = cuda_sgp4(
        tle_lines,
        timestep_length_in_seconds=timestep_seconds,
        total_sim_seconds=total_seconds,
        start_time=start_time
    )
    
    # Should be NumPy arrays, not device arrays
    assert isinstance(pos, np.ndarray)
    assert isinstance(vel, np.ndarray)
    assert not isinstance(pos, cuda.devicearray.DeviceNDArray)
    assert not isinstance(vel, cuda.devicearray.DeviceNDArray)
    
    # Test explicit False
    pos2, vel2 = cuda_sgp4(
        tle_lines,
        timestep_length_in_seconds=timestep_seconds,
        total_sim_seconds=total_seconds,
        start_time=start_time,
        return_device_arrays=False
    )
    
    # Should be identical
    np.testing.assert_array_equal(pos, pos2)
    np.testing.assert_array_equal(vel, vel2) 