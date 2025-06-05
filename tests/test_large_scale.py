"""Large-scale test for CUDA_SGP4 with many satellites and time steps."""

import time
import numpy as np
from datetime import datetime
from numba import cuda
import pytest
import warnings
from numba.core.errors import NumbaPerformanceWarning

from cuda_sgp4 import cuda_sgp4

# Suppress CUDA performance warnings for test cases
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)


def generate_test_tles(num_satellites=65):
    """Generate test TLE data for the specified number of satellites."""
    tle_lines = []
    
    # Base TLE data - we'll modify satellite numbers and some orbital elements
    base_line1 = "1 {satnum:05d}U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753"
    base_line2 = "2 {satnum:05d}  {incl:7.4f} {raan:8.4f} {ecc:7s} {argp:8.4f}  {ma:8.4f} {mm:11.8f}{rev:5d}"
    
    for i in range(num_satellites):
        satnum = 10000 + i
        
        # Vary orbital parameters slightly for each satellite
        incl = 34.2682 + (i % 10) * 2.0  # Inclination variation
        raan = 348.7242 + (i % 20) * 18.0  # RAAN variation
        ecc_val = 0.1859667 + (i % 5) * 0.01  # Eccentricity variation
        # Format eccentricity correctly (remove leading 0. and pad to 7 digits)
        ecc = f"{ecc_val:.7f}"[2:]  # Remove "0." and keep 7 digits
        argp = 331.7664 + (i % 15) * 24.0  # Argument of perigee variation
        ma = 19.3264 + (i % 12) * 30.0  # Mean anomaly variation
        mm = 10.82419157  # Keep mean motion constant for simplicity
        rev = 41366 + i  # Revolution number
        
        line1 = base_line1.format(satnum=satnum)
        line2 = base_line2.format(
            satnum=satnum, incl=incl, raan=raan, ecc=ecc,
            argp=argp, ma=ma, mm=mm, rev=rev
        )
        
        tle_lines.append((line1, line2))
    
    return tle_lines


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA is not available")
@pytest.mark.slow
def test_large_scale_propagation():
    """Test large-scale propagation with 65 satellites and 11800 time steps.
    
    This test verifies that the package can handle large workloads efficiently
    and produces valid results for extended propagation periods.
    """
    # Test configuration
    num_satellites = 65
    num_timesteps = 11800
    timestep_seconds = 60  # 1 minute steps
    total_seconds = num_timesteps * timestep_seconds  # ~8.2 days
    start_time = datetime(2000, 6, 29, 12, 50, 19)
    
    # Generate test data
    tle_lines = generate_test_tles(num_satellites)
    assert len(tle_lines) == num_satellites
    
    # Memory estimation for validation
    expected_elements = num_satellites * num_timesteps * 3 * 2  # positions + velocities
    expected_memory_mb = expected_elements * 8 / (1024 * 1024)  # 8 bytes per float64
    
    # Run the propagation
    start_time_exec = time.time()
    
    positions, velocities = cuda_sgp4(
        tle_lines=tle_lines,
        timestep_length_in_seconds=timestep_seconds,
        total_sim_seconds=total_seconds,
        start_time=start_time,
        return_device_arrays=False  # CPU output
    )
    
    execution_time = time.time() - start_time_exec
    
    # Verify results structure
    assert positions.shape == (num_satellites, num_timesteps, 3)
    assert velocities.shape == (num_satellites, num_timesteps, 3)
    assert positions.dtype == np.float64
    assert velocities.dtype == np.float64
    
    # Verify memory usage is as expected
    actual_memory_mb = (positions.nbytes + velocities.nbytes) / (1024 * 1024)
    assert abs(actual_memory_mb - expected_memory_mb) < 1.0  # Within 1 MB
    
    # Check for numerical validity
    assert np.all(np.isfinite(positions)), "Found non-finite values in positions"
    assert np.all(np.isfinite(velocities)), "Found non-finite values in velocities"
    
    # Sanity checks on orbital mechanics
    # Positions should be reasonable for Earth satellites (within ~50,000 km)
    position_magnitudes = np.linalg.norm(positions, axis=2)
    assert np.all(position_magnitudes > 6000), "Some satellites below Earth's surface"
    assert np.all(position_magnitudes < 50000), "Some satellites unreasonably far from Earth"
    
    # Velocities should be reasonable for Earth satellites (0.1 to 15 km/s)
    velocity_magnitudes = np.linalg.norm(velocities, axis=2)
    assert np.all(velocity_magnitudes > 0.1), "Some velocities unreasonably low"
    assert np.all(velocity_magnitudes < 15.0), "Some velocities unreasonably high"
    
    # Performance validation
    total_satellite_timesteps = num_satellites * num_timesteps
    throughput = total_satellite_timesteps / execution_time
    
    # Should process at least 10,000 satellite-timesteps per second
    assert throughput > 10000, f"Throughput too low: {throughput:.0f} satellite-timesteps/second"
    
    # Test should complete in reasonable time (less than 30 seconds)
    assert execution_time < 30.0, f"Test took too long: {execution_time:.2f} seconds"
    
    # Log performance metrics for monitoring
    print(f"\nLarge-scale test performance:")
    print(f"- Satellites: {num_satellites}")
    print(f"- Timesteps: {num_timesteps}")
    print(f"- Total satellite-timesteps: {total_satellite_timesteps:,}")
    print(f"- Execution time: {execution_time:.2f} seconds")
    print(f"- Throughput: {throughput:.0f} satellite-timesteps/second")
    print(f"- Memory usage: {actual_memory_mb:.1f} MB")


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA is not available")
@pytest.mark.slow
def test_large_scale_device_arrays():
    """Test large-scale propagation with device arrays for maximum efficiency."""
    # Smaller scale for device array test to keep it reasonable
    num_satellites = 32
    num_timesteps = 5000
    timestep_seconds = 60
    total_seconds = num_timesteps * timestep_seconds
    start_time = datetime(2000, 6, 29, 12, 50, 19)
    
    # Generate test data
    tle_lines = generate_test_tles(num_satellites)
    
    # Test device array workflow
    start_time_exec = time.time()
    
    # Method 1: Traditional approach
    pos1, vel1 = cuda_sgp4(
        tle_lines=tle_lines,
        timestep_length_in_seconds=timestep_seconds,
        total_sim_seconds=total_seconds,
        start_time=start_time,
        return_device_arrays=True
    )
    
    traditional_time = time.time() - start_time_exec
    
    # Method 2: Device array input approach
    from cuda_sgp4 import tle_lines_to_device_array
    
    start_time_exec = time.time()
    tle_device_array = tle_lines_to_device_array(tle_lines, start_time)
    preprocessing_time = time.time() - start_time_exec
    
    start_time_exec = time.time()
    pos2, vel2 = cuda_sgp4(
        tle_device_array=tle_device_array,
        timestep_length_in_seconds=timestep_seconds,
        total_sim_seconds=total_seconds,
        return_device_arrays=True
    )
    device_input_time = time.time() - start_time_exec
    
    # Verify results are close (allowing for numerical differences)
    from cuda_sgp4 import device_arrays_to_host
    pos1_host, vel1_host = device_arrays_to_host(pos1, vel1)
    pos2_host, vel2_host = device_arrays_to_host(pos2, vel2)
    
    # Use relaxed tolerance for large-scale comparisons
    np.testing.assert_allclose(pos1_host, pos2_host, rtol=1e-8, atol=1e-6)
    np.testing.assert_allclose(vel1_host, vel2_host, rtol=1e-8, atol=1e-6)
    
    # Verify shapes and types
    assert pos1.shape == (num_satellites, num_timesteps, 3)
    assert vel1.shape == (num_satellites, num_timesteps, 3)
    assert pos2.shape == (num_satellites, num_timesteps, 3)
    assert vel2.shape == (num_satellites, num_timesteps, 3)
    
    # Performance comparison
    total_time_device_input = preprocessing_time + device_input_time
    
    print(f"\nDevice array performance comparison:")
    print(f"- Traditional approach: {traditional_time:.4f} seconds")
    print(f"- Device input approach: {total_time_device_input:.4f} seconds")
    print(f"  - Preprocessing: {preprocessing_time:.4f} seconds")
    print(f"  - Propagation: {device_input_time:.4f} seconds")
    
    # For single runs, traditional might be faster due to preprocessing overhead
    # But device input approach should be competitive
    assert device_input_time < traditional_time * 2.0, "Device input propagation should be efficient" 