# CUDA SGP4

CUDA_SGP4 is a Python package that accelerates the SGP4 satellite orbit
propagation algorithm using Numba's CUDA support. It exposes a single
function `cuda_sgp4` that accepts two-line element (TLE) strings and
returns the propagated position and velocity vectors for each satellite.

## Installation

This project targets Python 3.10 and requires a CUDA capable GPU.
Install the package and its dependencies with `pip` or `poetry`:

```bash
pip install -e .
# or
poetry install
```

## Usage

### Basic Usage (Host Arrays)

```python
from datetime import datetime
from cuda_sgp4 import cuda_sgp4

line1 = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753"
line2 = "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667"

# propagate with a 60 second step for one hour
positions, velocities = cuda_sgp4(
    [(line1, line2)],
    timestep_length_in_seconds=60,
    total_sim_seconds=3600,
    start_time=datetime.utcnow(),
)
```

`positions` and `velocities` have shape `(n_sats, n_steps, 3)` and are NumPy arrays on the host.

### Device Arrays (GPU Memory)

**New Feature**: Keep arrays on GPU for further processing!

```python
from datetime import datetime
from cuda_sgp4 import cuda_sgp4, device_arrays_to_host, get_device_array_info

# Get device arrays that stay on GPU
device_positions, device_velocities = cuda_sgp4(
    [(line1, line2)],
    timestep_length_in_seconds=60,
    total_sim_seconds=3600,
    start_time=datetime.utcnow(),
    return_device_arrays=True,  # Keep data on GPU!
)

# Do more GPU processing here...
# Your CUDA kernels can work directly with device_positions and device_velocities

# Get array information
info = get_device_array_info(device_positions)
print(f"Shape: {info['shape']}, Memory: {info['nbytes']} bytes")

# Convert to host when needed
host_positions, host_velocities = device_arrays_to_host(
    device_positions, device_velocities
)
```

### Benefits of Device Arrays

- **No unnecessary memory transfers**: Keep data on GPU between operations
- **Better performance**: Chain multiple GPU operations efficiently
- **Reduced memory bandwidth**: Avoid host-device copies
- **GPU pipeline friendly**: Perfect for GPU-accelerated workflows

### When to Use Each Mode

**Use `return_device_arrays=True` when:**

- You plan to do more GPU processing on the results
- You want to minimize memory transfers
- You're building a GPU-accelerated pipeline
- You have custom CUDA kernels that operate on the position/velocity data

**Use `return_device_arrays=False` (default) when:**

- You need the data for CPU processing
- You want to save/visualize the results immediately
- You're doing one-off calculations
- You prefer the simplicity of NumPy arrays

## API Reference

### `cuda_sgp4(tle_lines, timestep_length_in_seconds, total_sim_seconds, start_time, return_device_arrays=False)`

**Parameters:**

- `tle_lines`: Iterable of `(line1, line2)` TLE pairs
- `timestep_length_in_seconds`: Length of each time step in seconds
- `total_sim_seconds`: Total propagation duration in seconds
- `start_time`: Start epoch of the propagation
- `return_device_arrays`: If `True`, returns CUDA device arrays. If `False` (default), returns NumPy host arrays.

**Returns:**

- If `return_device_arrays=False`: `Tuple[np.ndarray, np.ndarray]` - NumPy arrays on host
- If `return_device_arrays=True`: `Tuple[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]` - CUDA device arrays on GPU

### `device_arrays_to_host(device_positions, device_velocities)`

Convert CUDA device arrays to NumPy host arrays.

### `get_device_array_info(device_array)`

Get information about a CUDA device array (shape, dtype, memory usage, etc.).

## Example

See `example_device_arrays.py` for a comprehensive example demonstrating:

- Traditional host array usage
- Device array usage with GPU processing
- Memory usage comparisons
- Best practices

## Checking the Code

Unit tests are provided under the `tests` directory. Run them with

```bash
poetry run pytest
```

You can still perform a quick syntax check with:

```bash
python -m py_compile $(git ls-files '*.py')
```

## License

This repository contains code derived from the public SGP4 implementation by
David Vallado. See individual file headers for attribution.
