# CUDA SGP4

CUDA_SGP4 is a Python package that accelerates the SGP4 satellite orbit
propagation algorithm using Numba's CUDA support. The package provides a
function `cuda_sgp4` that loads TLE data from a CSV file, propagates each
satellite across a series of time steps on the GPU and stores the resulting
position and velocity vectors in a CSV output file.

## Installation

This project targets Python 3.10 and requires a CUDA capable GPU.
Install the package and its dependencies with `pip` or `poetry`:

```bash
pip install -e .
# or
poetry install
```

## Input Data

`cuda_sgp4` expects a CSV file containing at least the columns `line1`,
`line2`, `epoch` and `satNo`. Each row represents one TLE. The `epoch`
column should be in ISO 8601 format (e.g. `2024-01-01T00:00:00.000Z`).

## Usage

```python
from datetime import datetime
from cuda_sgp4.cuda_sgp4 import cuda_sgp4

# propagate with a 60 second step for one hour
cuda_sgp4(
    timestep_length_in_seconds=60,
    total_sim_seconds=3600,
    start_time=datetime.utcnow(),
    tle_file_path="tles.csv",       # CSV with line1, line2, epoch, satNo
    output_file_path="orbit_output.csv"
)
```

The output CSV begins with a small header and then a line per satellite per
step containing the satellite number, epoch timestamp and propagated state
vectors:

```
HEADER_START
variable_name, value
...
HEADER_END
SatNo,timestamp,x,y,z,vx,vy,vz
```

## Checking the Code

There are currently no unit tests. You can perform a basic syntax check with:

```bash
python -m py_compile $(git ls-files '*.py')
```

## License

This repository contains code derived from the public SGP4 implementation by
David Vallado. See individual file headers for attribution.
