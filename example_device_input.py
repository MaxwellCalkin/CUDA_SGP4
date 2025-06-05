#!/usr/bin/env python3
"""
Example demonstrating CUDA_SGP4 device array input capabilities.

This example shows:
1. Traditional usage (TLE strings → NumPy arrays)
2. Device output (TLE strings → CUDA device arrays)
3. Device input (CUDA device arrays → CUDA device arrays)
4. CuPy integration for GPU-accelerated workflows
5. Performance comparison between different modes
"""

import time
from datetime import datetime
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not available - skipping CuPy examples")

from cuda_sgp4 import (
    cuda_sgp4, 
    tle_lines_to_device_array, 
    device_arrays_to_host, 
    get_device_array_info
)


def main():
    # Sample TLE data
    tle_lines = [
        ("1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753",
         "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667"),
        ("1 00006U 58002C   00179.78495062  .00000023  00000-0  28098-4 0  4754",
         "2 00006  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413668"),
    ]
    
    start_time = datetime(2000, 6, 29, 12, 50, 19)
    timestep_seconds = 60
    total_seconds = 3600  # 1 hour
    
    print("CUDA_SGP4 Device Array Input Example")
    print("=" * 50)
    
    # 1. Traditional usage (TLE strings → NumPy arrays)
    print("\n1. Traditional Usage (TLE strings → NumPy arrays)")
    print("-" * 50)
    
    start = time.time()
    positions, velocities = cuda_sgp4(
        tle_lines=tle_lines,
        timestep_length_in_seconds=timestep_seconds,
        total_sim_seconds=total_seconds,
        start_time=start_time,
        return_device_arrays=False
    )
    traditional_time = time.time() - start
    
    print(f"Shape: {positions.shape}")
    print(f"Type: {type(positions)}")
    print(f"Time: {traditional_time:.4f} seconds")
    print(f"Sample position: {positions[0, 0, :]}")
    
    # 2. Device output (TLE strings → CUDA device arrays)
    print("\n2. Device Output (TLE strings → CUDA device arrays)")
    print("-" * 50)
    
    start = time.time()
    device_positions, device_velocities = cuda_sgp4(
        tle_lines=tle_lines,
        timestep_length_in_seconds=timestep_seconds,
        total_sim_seconds=total_seconds,
        start_time=start_time,
        return_device_arrays=True
    )
    device_output_time = time.time() - start
    
    info = get_device_array_info(device_positions)
    print(f"Shape: {info['shape']}")
    print(f"Type: {type(device_positions)}")
    print(f"Memory: {info['nbytes']} bytes")
    print(f"Time: {device_output_time:.4f} seconds")
    
    # Convert to host to show sample
    sample_pos, _ = device_arrays_to_host(device_positions, device_velocities)
    print(f"Sample position: {sample_pos[0, 0, :]}")
    
    # 3. Device input (CUDA device arrays → CUDA device arrays)
    print("\n3. Device Input (CUDA device arrays → CUDA device arrays)")
    print("-" * 50)
    
    # Pre-process TLE data to device array (one-time cost)
    start = time.time()
    tle_device_array = tle_lines_to_device_array(tle_lines, start_time)
    preprocessing_time = time.time() - start
    
    print(f"TLE preprocessing time: {preprocessing_time:.4f} seconds")
    print(f"TLE device array shape: {tle_device_array.shape}")
    
    # Now run propagation using device array input
    start = time.time()
    device_pos2, device_vel2 = cuda_sgp4(
        tle_device_array=tle_device_array,
        timestep_length_in_seconds=timestep_seconds,
        total_sim_seconds=total_seconds,
        return_device_arrays=True
    )
    device_input_time = time.time() - start
    
    print(f"Propagation time: {device_input_time:.4f} seconds")
    print(f"Total time (preprocessing + propagation): {preprocessing_time + device_input_time:.4f} seconds")
    
    # Verify results are identical
    pos2_host, vel2_host = device_arrays_to_host(device_pos2, device_vel2)
    print(f"Results identical to method 2: {np.allclose(sample_pos, pos2_host)}")
    
    # 4. Multiple runs with same TLE data (efficiency demonstration)
    print("\n4. Multiple Runs with Same TLE Data")
    print("-" * 50)
    
    # Run multiple scenarios with different timesteps
    timesteps = [30, 60, 120, 240]
    
    print("Using device array input (efficient):")
    start = time.time()
    for ts in timesteps:
        pos, vel = cuda_sgp4(
            tle_device_array=tle_device_array,
            timestep_length_in_seconds=ts,
            total_sim_seconds=1800,  # 30 minutes
            return_device_arrays=True
        )
    device_multiple_time = time.time() - start
    print(f"Time for {len(timesteps)} runs: {device_multiple_time:.4f} seconds")
    
    print("\nUsing traditional input (less efficient):")
    start = time.time()
    for ts in timesteps:
        pos, vel = cuda_sgp4(
            tle_lines=tle_lines,
            timestep_length_in_seconds=ts,
            total_sim_seconds=1800,
            start_time=start_time,
            return_device_arrays=True
        )
    traditional_multiple_time = time.time() - start
    print(f"Time for {len(timesteps)} runs: {traditional_multiple_time:.4f} seconds")
    print(f"Speedup: {traditional_multiple_time / device_multiple_time:.2f}x")
    
    # 5. CuPy integration (if available)
    if CUPY_AVAILABLE:
        print("\n5. CuPy Integration")
        print("-" * 50)
        
        # Get device arrays from CUDA_SGP4
        device_pos, device_vel = cuda_sgp4(
            tle_device_array=tle_device_array,
            timestep_length_in_seconds=60,
            total_sim_seconds=3600,
            return_device_arrays=True
        )
        
        # Convert to CuPy arrays
        cupy_positions = cp.asarray(device_pos)
        cupy_velocities = cp.asarray(device_vel)
        
        print(f"CuPy positions shape: {cupy_positions.shape}")
        print(f"CuPy positions type: {type(cupy_positions)}")
        
        # Perform GPU computations with CuPy
        distances = cp.linalg.norm(cupy_positions, axis=2)
        speeds = cp.linalg.norm(cupy_velocities, axis=2)
        
        print(f"Distances shape: {distances.shape}")
        print(f"Average distance (satellite 0): {cp.mean(distances[0]):.2f} km")
        print(f"Average speed (satellite 0): {cp.mean(speeds[0]):.2f} km/s")
        
        # All operations stayed on GPU!
        print("All computations performed on GPU - no host transfers!")
    
    # Performance summary
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Traditional (strings → host):     {traditional_time:.4f}s")
    print(f"Device output (strings → device): {device_output_time:.4f}s")
    print(f"Device input (device → device):   {device_input_time:.4f}s")
    print(f"TLE preprocessing:                {preprocessing_time:.4f}s")
    print()
    print("Use device input when:")
    print("- Running multiple propagations with same TLE data")
    print("- Building GPU-accelerated pipelines")
    print("- Integrating with CuPy or other GPU libraries")
    print("- Minimizing memory transfers")


if __name__ == "__main__":
    main() 