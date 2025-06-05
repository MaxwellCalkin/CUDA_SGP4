# CUDA SGP4

CUDA_SGP4 is a Python package that accelerates the SGP4 satellite orbit
propagation algorithm using Numba's CUDA support. It exposes a single
function `cuda_sgp4` that accepts two-line element (TLE) strings and
returns the propagated position and velocity vectors for each satellite.

**Key Features:**

- **GPU Acceleration**: Leverages CUDA for high-performance orbit propagation
- **Device Array Support**: Both input and output can stay on GPU for efficient pipelines
- **CuPy Compatible**: Works seamlessly with CuPy arrays and GPU workflows
- **Flexible I/O**: Choose between host arrays (NumPy) or device arrays (CUDA) for both input and output

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
    tle_lines=[(line1, line2)],
    timestep_length_in_seconds=60,
    total_sim_seconds=3600,
    start_time=datetime.utcnow(),
)
```

`positions` and `velocities` have shape `(n_sats, n_steps, 3)` and are NumPy arrays on the host.

### Device Arrays (GPU Memory)

**Keep arrays on GPU for further processing!**

```python
from datetime import datetime
from cuda_sgp4 import cuda_sgp4, device_arrays_to_host, get_device_array_info

# Get device arrays that stay on GPU
device_positions, device_velocities = cuda_sgp4(
    tle_lines=[(line1, line2)],
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

### Device Array Input (GPU-to-GPU Workflows)

**NEW**: Accept pre-processed TLE data as device arrays for maximum efficiency!

```python
from datetime import datetime
from cuda_sgp4 import cuda_sgp4, tle_lines_to_device_array

# Pre-process TLE data to device array (one-time cost)
tle_device_array = tle_lines_to_device_array(
    tle_lines=[(line1, line2)],
    start_time=datetime.utcnow()
)

# Now run multiple propagations efficiently (no host-device transfers!)
positions1, velocities1 = cuda_sgp4(
    tle_device_array=tle_device_array,
    timestep_length_in_seconds=60,
    total_sim_seconds=3600,
    return_device_arrays=True
)

positions2, velocities2 = cuda_sgp4(
    tle_device_array=tle_device_array,
    timestep_length_in_seconds=30,  # Different timestep
    total_sim_seconds=1800,         # Different duration
    return_device_arrays=True
)

# Perfect for parameter sweeps, Monte Carlo simulations, etc.
```

### CuPy Integration

**Works seamlessly with CuPy arrays and GPU workflows:**

```python
import cupy as cp
from cuda_sgp4 import cuda_sgp4, tle_lines_to_device_array

# Pre-process TLE data
tle_device_array = tle_lines_to_device_array(tle_lines, start_time)

# Get device arrays from CUDA_SGP4
device_pos, device_vel = cuda_sgp4(
    tle_device_array=tle_device_array,
    timestep_length_in_seconds=60,
    total_sim_seconds=3600,
    return_device_arrays=True
)

# Convert to CuPy arrays for further processing
cupy_positions = cp.asarray(device_pos)
cupy_velocities = cp.asarray(device_vel)

# Now use CuPy's extensive GPU computing capabilities
distances = cp.linalg.norm(cupy_positions, axis=2)
speeds = cp.linalg.norm(cupy_velocities, axis=2)

# All operations stay on GPU - maximum efficiency!
```

### Benefits of Device Arrays

- **No unnecessary memory transfers**: Keep data on GPU between operations
- **Better performance**: Chain multiple GPU operations efficiently
- **Reduced memory bandwidth**: Avoid host-device copies
- **GPU pipeline friendly**: Perfect for GPU-accelerated workflows
- **CuPy compatible**: Seamless integration with CuPy ecosystem
- **Parameter sweeps**: Efficient for running multiple scenarios with same TLE data

### When to Use Each Mode

**Use device array input (`tle_device_array=...`) when:**

- You're running multiple propagations with the same TLE data
- You're doing parameter sweeps or Monte Carlo simulations
- You want maximum GPU efficiency
- You're integrating with CuPy or other GPU libraries
- TLE data is already processed and you want to reuse it

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

### `cuda_sgp4(tle_lines=None, timestep_length_in_seconds=60, total_sim_seconds=3600, start_time=None, return_device_arrays=False, tle_device_array=None)`

**Parameters:**

- `tle_lines`: Iterable of `(line1, line2)` TLE pairs. Required if `tle_device_array` is None.
- `timestep_length_in_seconds`: Length of each time step in seconds (default: 60)
- `total_sim_seconds`: Total propagation duration in seconds (default: 3600)
- `start_time`: Start epoch of the propagation. Required if `tle_device_array` is None.
- `return_device_arrays`: If `True`, returns CUDA device arrays. If `False` (default), returns NumPy host arrays.
- `tle_device_array`: Pre-processed TLE data as CUDA device array. If provided, `tle_lines` and `start_time` are ignored.

**Returns:**

- If `return_device_arrays=False`: `Tuple[np.ndarray, np.ndarray]` - NumPy arrays on host
- If `return_device_arrays=True`: `Tuple[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]` - CUDA device arrays on GPU

### `tle_lines_to_device_array(tle_lines, start_time)`

Convert TLE lines to a CUDA device array for efficient reuse.

**Parameters:**

- `tle_lines`: Iterable of `(line1, line2)` TLE pairs
- `start_time`: Start epoch for TLE processing

**Returns:**

- `cuda.devicearray.DeviceNDArray`: Processed TLE data ready for `cuda_sgp4(tle_device_array=...)`

### `device_arrays_to_host(device_positions, device_velocities)`

Convert CUDA device arrays to NumPy host arrays.

### `get_device_array_info(device_array)`

Get information about a CUDA device array (shape, dtype, memory usage, etc.).

## Example

See `example_device_arrays.py` for a comprehensive example demonstrating:

- Traditional host array usage
- Device array usage with GPU processing
- Device array input for efficient workflows
- CuPy integration examples
- Memory usage comparisons
- Best practices

**NEW**: See `example_device_input.py` for a detailed demonstration of:

- Device array input capabilities
- Performance comparisons between different modes
- CuPy integration for GPU-accelerated workflows
- Efficient parameter sweeps and multiple runs
- Best practices for GPU-to-GPU workflows

## Checking the Code

Unit tests are provided under the `tests` directory. Run them with

```bash
poetry run pytest
```

### Test Categories

The test suite includes different categories of tests:

- **Standard tests**: Fast tests that run in a few seconds
- **Large-scale tests**: Marked with `@pytest.mark.slow` - these test with 65 satellites and 11,800 time steps

To run only fast tests (excluding large-scale tests):

```bash
poetry run pytest -m "not slow"
```

To run only large-scale tests:

```bash
poetry run pytest -m "slow"
```

To run all tests with verbose output:

```bash
poetry run pytest -v
```

You can still perform a quick syntax check with:

```bash
python -m py_compile $(git ls-files '*.py')
```

## License

This repository contains code derived from the public SGP4 implementation by
David Vallado. See individual file headers for attribution.
