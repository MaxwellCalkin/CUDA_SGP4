# Repository Guide for LLM Agents

This repository implements a CUDA accelerated version of the SGP4 orbit
propagator. The primary entry point is `cuda_sgp4/cuda_sgp4.py`.

## Key Files

- `cuda_sgp4/cuda_sgp4.py` – user facing API. The `cuda_sgp4` function
  loads TLE data, launches the GPU kernel defined in
  `cuda_sgp4/src/cuda_sgp4.py` and writes the results to a CSV file.
- `cuda_sgp4/src/initialize_tle_arrays.py` – converts a CSV of TLEs into an
  array format suitable for the GPU.
- `cuda_sgp4/src/cuda_sgp4.py` – contains the CUDA kernels including the
  `propagate_orbit` kernel.

## Development Notes

- Python 3.10 is expected.
- Run all commands via `poetry run` to ensure the correct environment is used.
- Unit tests live in the `tests` directory. Run them with `poetry run pytest`
  and ensure they pass before committing.
- To perform a quick syntax check run:
  `python -m py_compile $(git ls-files '*.py')`
- When modifying the code follow standard PEP8 style and keep the existing
  structure.

