#!/usr/bin/env python3
"""
Simple example showing the difference between host and device arrays.
"""

from datetime import datetime
from cuda_sgp4 import cuda_sgp4, device_arrays_to_host
from numba import cuda
import numpy as np

# Sample TLE
tle_lines = [
    (
        "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
        "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    )
]

start_time = datetime(2008, 9, 20)

print("=== OLD WAY (always copies to host) ===")
positions, velocities = cuda_sgp4(
    tle_lines,
    timestep_length_in_seconds=60,
    total_sim_seconds=3600,
    start_time=start_time,
    return_device_arrays=False  # Default behavior
)
print(f"Type: {type(positions)}")
print(f"Shape: {positions.shape}")
print(f"Data is on: Host (CPU memory)")

print("\n=== NEW WAY (keeps data on GPU) ===")
device_positions, device_velocities = cuda_sgp4(
    tle_lines,
    timestep_length_in_seconds=60,
    total_sim_seconds=3600,
    start_time=start_time,
    return_device_arrays=True  # New feature!
)
print(f"Type: {type(device_positions)}")
print(f"Shape: {device_positions.shape}")
print(f"Data is on: GPU (device memory)")

# Now you can do more GPU processing without copying back and forth!
# For example, calculate distances on GPU:

@cuda.jit
def calculate_distances_kernel(positions, distances):
    """Calculate distance from Earth center for each position."""
    sat_idx, step_idx = cuda.grid(2)
    if sat_idx < positions.shape[0] and step_idx < positions.shape[1]:
        x = positions[sat_idx, step_idx, 0]
        y = positions[sat_idx, step_idx, 1]
        z = positions[sat_idx, step_idx, 2]
        distances[sat_idx, step_idx] = (x*x + y*y + z*z) ** 0.5

# Allocate output array on GPU
distances = cuda.device_array(device_positions.shape[:2], dtype=np.float64)

# Launch kernel
threads_per_block = (16, 16)
blocks_per_grid = (1, 4)  # Adjust based on your data size
calculate_distances_kernel[blocks_per_grid, threads_per_block](device_positions, distances)

print(f"\n=== ADDITIONAL GPU PROCESSING ===")
print(f"Calculated distances on GPU without any host transfers!")

# Only copy to host when you actually need the results
host_distances = distances.copy_to_host()
print(f"Distance range: {np.min(host_distances):.1f} - {np.max(host_distances):.1f} km")

# Convert device arrays to host when needed
final_positions, final_velocities = device_arrays_to_host(device_positions, device_velocities)
print(f"\nFinal results copied to host: {type(final_positions)}")

print("\n=== BENEFITS ===")
print("✓ No unnecessary GPU ↔ CPU memory transfers")
print("✓ Can chain multiple GPU operations efficiently")
print("✓ Better performance for GPU-heavy workflows")
print("✓ Backward compatible - old code still works!") 