# GPU Utilization Guide for CUDA SGP4

## Understanding the Warning

When you see this warning:

```
NumbaPerformanceWarning: Grid size 1 will likely result in GPU under-utilization due to low occupancy.
```

**What it means:** Your CUDA kernel is not using enough threads to fully utilize the GPU's parallel processing capabilities.

**Why it happens:** GPUs are designed to run thousands of threads simultaneously. When you only have a few satellites (< 32), you're not providing enough work to keep the GPU busy.

**Is it a problem?**

- ❌ **For production workloads**: Yes, you're not getting the performance benefits of GPU acceleration
- ✅ **For small test cases**: No, it's expected and harmless

## Solutions

### Solution 1: Use Optimal Launch Configuration (Recommended)

The `get_optimal_launch_config()` function automatically calculates the best grid and block sizes for your workload:

```python
from cuda_sgp4.src.cuda_sgp4 import propagate_orbit, get_optimal_launch_config

# Instead of fixed configuration:
# threads_per_block = 256
# blocks_per_grid = (num_satellites + threads_per_block - 1) // threads_per_block

# Use optimal configuration:
blocks_per_grid, threads_per_block = get_optimal_launch_config(num_satellites)
propagate_orbit[blocks_per_grid, threads_per_block](d_tles, d_r, d_v, total_timesteps, timestep_seconds)
```

**How it works:**

- Analyzes your GPU's capabilities (max threads per block, number of SMs)
- Ensures minimum number of blocks for good occupancy
- Balances thread utilization vs. SM utilization
- Uses warp-aligned block sizes (multiples of 32)

**Example outputs:**

- 1 satellite: (1 block, 32 threads) - minimal overhead
- 10 satellites: (1 block, 32 threads) - still efficient
- 100 satellites: (4 blocks, 32 threads) - better SM utilization
- 1000 satellites: (32 blocks, 32 threads) - excellent utilization

### Solution 2: Suppress Warnings for Test Cases

For test suites and small workloads where the warning is expected:

```python
import warnings
from numba.core.errors import NumbaPerformanceWarning

# Suppress the warning
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)

# Or use the convenience function
from cuda_sgp4.src.cuda_sgp4 import suppress_cuda_warnings
suppress_cuda_warnings()
```

### Solution 3: Batch Small Workloads

For production code with small satellite counts, consider batching:

```python
def process_satellites_efficiently(satellite_batches):
    """Process multiple small batches together for better GPU utilization."""

    # Combine small batches into larger ones
    min_batch_size = 64  # Ensure good GPU utilization

    combined_batch = []
    for batch in satellite_batches:
        combined_batch.extend(batch)

        if len(combined_batch) >= min_batch_size:
            # Process the combined batch
            blocks_per_grid, threads_per_block = get_optimal_launch_config(len(combined_batch))
            propagate_orbit[blocks_per_grid, threads_per_block](...)
            combined_batch = []

    # Process remaining satellites
    if combined_batch:
        blocks_per_grid, threads_per_block = get_optimal_launch_config(len(combined_batch))
        propagate_orbit[blocks_per_grid, threads_per_block](...)
```

### Solution 4: Hybrid CPU/GPU Approach

For very small workloads, CPU might actually be faster:

```python
def smart_propagation(tle_arrays, r, v, total_timesteps, timestep_seconds):
    """Automatically choose CPU or GPU based on workload size."""

    num_satellites = tle_arrays.shape[0]

    # Threshold where GPU becomes beneficial (depends on your hardware)
    gpu_threshold = 32

    if num_satellites < gpu_threshold:
        # Use CPU for small workloads
        for s, tle in enumerate(tles):
            for t_step in range(total_timesteps):
                tsince = tle_arrays[s, tIdx] + t_step * timestep_seconds / 60.0
                r_cpu = [0.0, 0.0, 0.0]
                v_cpu = [0.0, 0.0, 0.0]
                sgp4(tle.rec, tsince, r_cpu, v_cpu)
                r[:, s, t_step] = r_cpu
                v[:, s, t_step] = v_cpu
    else:
        # Use GPU for larger workloads
        d_tles = cuda.to_device(tle_arrays.astype(np.float64))
        d_r = cuda.device_array((3, num_satellites, total_timesteps), dtype=np.float64)
        d_v = cuda.device_array((3, num_satellites, total_timesteps), dtype=np.float64)

        blocks_per_grid, threads_per_block = get_optimal_launch_config(num_satellites)
        propagate_orbit[blocks_per_grid, threads_per_block](
            d_tles, d_r, d_v, total_timesteps, timestep_seconds
        )

        r[:] = d_r.copy_to_host()
        v[:] = d_v.copy_to_host()
```

## GPU Architecture Basics

Understanding these concepts helps you write better CUDA code:

### Warps and Blocks

- **Warp**: 32 threads that execute together (SIMD)
- **Block**: Group of threads (typically 32-1024) that can share memory
- **Grid**: Collection of blocks

### Occupancy

- **Theoretical Occupancy**: Maximum threads your GPU can run simultaneously
- **Achieved Occupancy**: Actual threads you're using
- **Goal**: Keep occupancy high (> 50%) for good performance

### Memory Coalescing

- Threads in a warp should access contiguous memory addresses
- Our SGP4 implementation is already optimized for this

## Performance Guidelines

### When GPU is Worth It

- **✅ Many satellites** (> 32): Excellent parallelization
- **✅ Many timesteps**: Amortizes GPU setup overhead
- **✅ Repeated computations**: GPU stays warm

### When CPU Might Be Better

- **❌ Few satellites** (< 10): GPU overhead dominates
- **❌ Single timestep**: Setup cost too high
- **❌ One-off computations**: CPU is simpler

### Optimal Workload Sizes

- **Sweet spot**: 100-10,000 satellites
- **Minimum for GPU benefit**: ~32 satellites
- **Maximum tested**: 50+ satellites (scales well)

## Example: Complete Optimized Usage

```python
import numpy as np
from datetime import datetime
from numba import cuda
from cuda_sgp4.src.initialize_tle_arrays import initialize_tle_arrays_from_lines
from cuda_sgp4.src.cuda_sgp4 import propagate_orbit, get_optimal_launch_config, suppress_cuda_warnings

def propagate_satellites_optimized(tle_lines, start_time, timestep_seconds=60, total_timesteps=10):
    """
    Optimized satellite propagation with smart GPU utilization.
    """

    # For test cases, suppress warnings
    if len(tle_lines) < 32:
        suppress_cuda_warnings()

    # Initialize TLE arrays
    tle_arrays, tles = initialize_tle_arrays_from_lines(tle_lines, start_time)
    num_satellites = tle_arrays.shape[0]

    # Prepare GPU arrays
    d_tles = cuda.to_device(tle_arrays.astype(np.float64))
    d_r = cuda.device_array((3, num_satellites, total_timesteps), dtype=np.float64)
    d_v = cuda.device_array((3, num_satellites, total_timesteps), dtype=np.float64)

    # Use optimal launch configuration
    blocks_per_grid, threads_per_block = get_optimal_launch_config(num_satellites)

    print(f"Processing {num_satellites} satellites with {blocks_per_grid} blocks × {threads_per_block} threads")

    # Launch kernel
    propagate_orbit[blocks_per_grid, threads_per_block](
        d_tles, d_r, d_v, total_timesteps, timestep_seconds
    )

    # Get results
    r = d_r.copy_to_host()
    v = d_v.copy_to_host()

    return r, v

# Example usage
if __name__ == "__main__":
    # Your TLE data here
    tle_lines = [...]  # List of (line1, line2) tuples
    start_time = datetime(2000, 1, 1)

    r, v = propagate_satellites_optimized(tle_lines, start_time)
    print(f"Propagated {r.shape[1]} satellites for {r.shape[2]} timesteps")
```

## Summary

The NumbaPerformanceWarning is helpful for identifying suboptimal GPU usage, but it's not always a problem:

1. **For production code**: Use `get_optimal_launch_config()` to maximize GPU utilization
2. **For test cases**: Use `suppress_cuda_warnings()` to hide expected warnings
3. **For very small workloads**: Consider using CPU instead
4. **For batch processing**: Combine small workloads into larger batches

The CUDA SGP4 implementation now provides tools for all these scenarios, ensuring you get optimal performance regardless of your use case.
