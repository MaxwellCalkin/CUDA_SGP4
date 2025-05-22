# CUDA SGP4

CUDA_SGP4 is a Python package that accelerates the SGP4 satellite orbit
propagation algorithm using Numba's CUDA support. It exposes a single
function ``cuda_sgp4`` that accepts two-line element (TLE) strings and
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

`positions` and `velocities` have shape `(n_sats, n_steps, 3)`.

## Checking the Code

There are currently no unit tests. You can perform a basic syntax check with:

```bash
python -m py_compile $(git ls-files '*.py')
```

## License

This repository contains code derived from the public SGP4 implementation by
David Vallado. See individual file headers for attribution.
