# Repository Guide for LLM Agents

This repository implements a CUDA accelerated version of the SGP4 orbit
propagator. The primary entry point is `cuda_sgp4/cuda_sgp4.py`.

## Key Capabilities

**CUDA_SGP4 supports both input and output of device arrays for maximum GPU efficiency:**

1. **Traditional Mode**: Input TLE strings, output NumPy arrays (host)
2. **Device Output Mode**: Input TLE strings, output CUDA device arrays (GPU)
3. **Device Input Mode**: Input CUDA device arrays, output CUDA device arrays (GPU)
4. **CuPy Integration**: Seamless conversion between CUDA device arrays and CuPy arrays

## Key Files

- `cuda_sgp4/cuda_sgp4.py` – user facing API. The `cuda_sgp4` function
  loads TLE data, launches the GPU kernel defined in
  `cuda_sgp4/src/cuda_sgp4.py` and returns the results.
- `cuda_sgp4/src/initialize_tle_arrays.py` – converts a CSV of TLEs into an
  array format suitable for the GPU.
- `cuda_sgp4/src/cuda_sgp4.py` – contains the CUDA kernels including the
  `propagate_orbit` kernel.

## API Functions

### `cuda_sgp4(...)`

Main propagation function with flexible input/output modes:

**Traditional usage (TLE strings → NumPy arrays):**

```python
positions, velocities = cuda_sgp4(
    tle_lines=[(line1, line2)],
    timestep_length_in_seconds=60,
    total_sim_seconds=3600,
    start_time=datetime.utcnow()
)
```

**Device output (TLE strings → CUDA device arrays):**

```python
device_pos, device_vel = cuda_sgp4(
    tle_lines=[(line1, line2)],
    timestep_length_in_seconds=60,
    total_sim_seconds=3600,
    start_time=datetime.utcnow(),
    return_device_arrays=True
)
```

**Device input (CUDA device arrays → CUDA device arrays):**

```python
device_pos, device_vel = cuda_sgp4(
    tle_device_array=preprocessed_tle_data,
    timestep_length_in_seconds=60,
    total_sim_seconds=3600,
    return_device_arrays=True
)
```

### `tle_lines_to_device_array(tle_lines, start_time)`

Pre-processes TLE strings into CUDA device arrays for efficient reuse:

```python
tle_device_array = tle_lines_to_device_array(tle_lines, start_time)
```

### `device_arrays_to_host(device_positions, device_velocities)`

Converts CUDA device arrays to NumPy host arrays when needed.

### `get_device_array_info(device_array)`

Returns metadata about CUDA device arrays (shape, dtype, memory usage).

## Usage Patterns for AI Agents

### Single Propagation

Use traditional mode for one-off calculations:

```python
from cuda_sgp4 import cuda_sgp4
positions, velocities = cuda_sgp4(tle_lines=tle_data, ...)
```

### Multiple Propagations (Same TLEs)

Use device input mode for efficiency:

```python
from cuda_sgp4 import cuda_sgp4, tle_lines_to_device_array

# Pre-process once
tle_device_array = tle_lines_to_device_array(tle_lines, start_time)

# Run multiple scenarios efficiently
for timestep in [30, 60, 120]:
    pos, vel = cuda_sgp4(
        tle_device_array=tle_device_array,
        timestep_length_in_seconds=timestep,
        return_device_arrays=True
    )
```

### CuPy Integration

Convert between CUDA device arrays and CuPy arrays:

```python
import cupy as cp
from cuda_sgp4 import cuda_sgp4

device_pos, device_vel = cuda_sgp4(..., return_device_arrays=True)
cupy_positions = cp.asarray(device_pos)
cupy_velocities = cp.asarray(device_vel)
```

## Development Notes

- Python 3.10 is expected.
- Run all commands via `poetry run` to ensure the correct environment is used.
- Unit tests live in the `tests` directory. Run them with `poetry run pytest`
  and ensure they pass before committing.
- To perform a quick syntax check run:
  `python -m py_compile $(git ls-files '*.py')`
- When modifying the code follow standard PEP8 style and keep the existing
  structure.

## Performance Tips

- Use `tle_device_array` input when running multiple propagations with the same TLE data
- Use `return_device_arrays=True` when doing further GPU processing
- Convert to host arrays only when necessary for CPU operations
- For CuPy workflows, keep everything on GPU using device arrays
