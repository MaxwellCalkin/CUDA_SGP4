#!/usr/bin/env python3
"""
Example demonstrating CUDA_SGP4 device array functionality.

This example shows how to:
1. Get device arrays that stay on the GPU
2. Perform further GPU processing on the results
3. Convert back to host when needed
"""

import numpy as np
from datetime import datetime
from numba import cuda
import math

# Import the enhanced CUDA_SGP4 functions
from cuda_sgp4 import cuda_sgp4, device_arrays_to_host, get_device_array_info


def main():
    """Demonstrate device array functionality."""
    
    # Check if CUDA is available
    if not cuda.is_available():
        print("CUDA is not available. This example requires a CUDA-capable GPU.")
        return
    
    print("CUDA_SGP4 Device Array Example")
    print("=" * 40)
    
    # Sample TLE data (ISS)
    tle_lines = [
        (
            "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
            "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
        )
    ]
    
    start_time = datetime(2008, 9, 20)
    timestep_seconds = 60  # 1 minute steps
    total_seconds = 3600   # 1 hour total
    
    print(f"Propagating {len(tle_lines)} satellite(s)")
    print(f"Time step: {timestep_seconds} seconds")
    print(f"Total duration: {total_seconds} seconds ({total_seconds//timestep_seconds} steps)")
    print()
    
    # Example 1: Traditional usage (returns host arrays)
    print("1. Traditional usage (host arrays):")
    host_pos, host_vel = cuda_sgp4(
        tle_lines,
        timestep_length_in_seconds=timestep_seconds,
        total_sim_seconds=total_seconds,
        start_time=start_time,
        return_device_arrays=False  # Default behavior
    )
    
    print(f"   Host positions shape: {host_pos.shape}")
    print(f"   Host velocities shape: {host_vel.shape}")
    print(f"   Type: {type(host_pos)}")
    print()
    
    # Example 2: New device array usage (keeps data on GPU)
    print("2. Device array usage (GPU arrays):")
    device_pos, device_vel = cuda_sgp4(
        tle_lines,
        timestep_length_in_seconds=timestep_seconds,
        total_sim_seconds=total_seconds,
        start_time=start_time,
        return_device_arrays=True  # New feature!
    )
    
    print(f"   Device positions shape: {device_pos.shape}")
    print(f"   Device velocities shape: {device_vel.shape}")
    print(f"   Type: {type(device_pos)}")
    print()
    
    # Example 3: Get device array information
    print("3. Device array information:")
    pos_info = get_device_array_info(device_pos)
    vel_info = get_device_array_info(device_vel)
    
    print(f"   Position array info: {pos_info}")
    print(f"   Velocity array info: {vel_info}")
    print()
    
    # Example 4: Perform additional GPU processing
    print("4. Additional GPU processing example:")
    
    # Calculate distances from Earth center on GPU
    distances = cuda.device_array(device_pos.shape[:2], dtype=np.float64)  # (n_sats, n_steps)
    
    # Define a simple kernel to calculate distances
    @cuda.jit
    def calculate_distances(positions, distances):
        sat_idx, step_idx = cuda.grid(2)
        if sat_idx < positions.shape[0] and step_idx < positions.shape[1]:
            x = positions[sat_idx, step_idx, 0]
            y = positions[sat_idx, step_idx, 1]
            z = positions[sat_idx, step_idx, 2]
            distances[sat_idx, step_idx] = math.sqrt(x*x + y*y + z*z)
    
    # Launch the kernel
    threads_per_block = (16, 16)
    blocks_per_grid_x = (device_pos.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (device_pos.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    calculate_distances[blocks_per_grid, threads_per_block](device_pos, distances)
    
    # Copy distances back to host for display
    host_distances = distances.copy_to_host()
    
    print(f"   Calculated distances on GPU")
    print(f"   Distance array shape: {host_distances.shape}")
    print(f"   Min distance: {np.min(host_distances):.2f} km")
    print(f"   Max distance: {np.max(host_distances):.2f} km")
    print(f"   Mean distance: {np.mean(host_distances):.2f} km")
    print()
    
    # Example 5: Convert device arrays to host when needed
    print("5. Converting device arrays to host:")
    converted_pos, converted_vel = device_arrays_to_host(device_pos, device_vel)
    
    print(f"   Converted positions shape: {converted_pos.shape}")
    print(f"   Converted velocities shape: {converted_vel.shape}")
    print(f"   Type: {type(converted_pos)}")
    
    # Verify they match the original host arrays
    pos_match = np.allclose(host_pos, converted_pos, rtol=1e-12)
    vel_match = np.allclose(host_vel, converted_vel, rtol=1e-12)
    
    print(f"   Positions match original: {pos_match}")
    print(f"   Velocities match original: {vel_match}")
    print()
    
    # Example 6: Memory usage comparison
    print("6. Memory usage comparison:")
    
    # Calculate memory usage
    host_memory = host_pos.nbytes + host_vel.nbytes
    device_memory = device_pos.nbytes + device_vel.nbytes
    
    print(f"   Host arrays memory: {host_memory / 1024**2:.2f} MB")
    print(f"   Device arrays memory: {device_memory / 1024**2:.2f} MB")
    print(f"   Additional GPU processing memory: {distances.nbytes / 1024**2:.2f} MB")
    print()
    
    print("Benefits of device arrays:")
    print("- No host-device memory transfers for intermediate results")
    print("- Can chain multiple GPU operations efficiently")
    print("- Reduced memory bandwidth usage")
    print("- Better performance for GPU-heavy workflows")
    print()
    
    print("Use device arrays when:")
    print("- You plan to do more GPU processing on the results")
    print("- You want to minimize memory transfers")
    print("- You're building a GPU-accelerated pipeline")
    print()
    
    print("Use host arrays when:")
    print("- You need the data for CPU processing")
    print("- You want to save/visualize the results immediately")
    print("- You're doing one-off calculations")


if __name__ == "__main__":
    main() 