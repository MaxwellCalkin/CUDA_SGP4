import numpy as np
from datetime import datetime
from numba import cuda
import pytest
import warnings
from numba.core.errors import NumbaPerformanceWarning

from cuda_sgp4.src.initialize_tle_arrays import initialize_tle_arrays_from_lines
from cuda_sgp4.src.cuda_sgp4 import propagate_orbit, tIdx, get_optimal_launch_config
from cuda_sgp4.src.SGP4 import sgp4

# Suppress CUDA performance warnings for test cases with small workloads
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)

def _make_tle(
    satnum: int, inc: float, raan: float, ecc: float, argp: float,
    ma: float, mm: float, rev: int = 1, epoch: str = "00179.78495062"
) -> tuple[str, str]:
    """Generate a simple synthetic TLE.

        Parameters are inserted into a fixed width string so that the lines remain
        69 characters long which matches the example TLE used elsewhere in the
        repository.
        """
    line1 = (
        f"1 {satnum:05d}U 58002B   {epoch}  .00000023  00000-0  28098-4 0  4753"
    )

    line2 = (
        # --- one blank after satnum ---↓
        f"2 {satnum:05d} {inc:8.4f} {raan:8.4f} {int(ecc*1e7):07d} "
        f"{argp:8.4f} {ma:8.4f} {mm:11.8f} {rev:05d}"   # ← space before rev
    )

    # Sanity check
    assert len(line1) == 69, len(line1)
    assert len(line2) == 69, len(line2)
    return line1, line2



def _generate_sample_tles() -> list[tuple[str, str]]:
    """Return a list of at least 20 TLEs spanning LEO to GEO orbits."""

    tles: list[tuple[str, str]] = []

    # LEO satellites (~15 revs/day)
    for i in range(7):
        sat = 10000 + i
        tles.append(
            _make_tle(
                sat,
                inc=51.6 + i * 0.1,
                raan=247.0 + i * 5.0,
                ecc=0.0001 * (i + 1),
                argp=130.0 + i,
                ma=10.0 * i,
                mm=15.0 + 0.1 * i,
                rev=100 + i,
            )
        )

    # MEO satellites (~2 revs/day)
    for i in range(7):
        sat = 10007 + i
        tles.append(
            _make_tle(
                sat,
                inc=55.0 + i * 0.2,
                raan=20.0 + i * 15.0,
                ecc=0.0005 * (i + 1),
                argp=200.0 + i,
                ma=5.0 * i,
                mm=2.0 + 0.01 * i,
                rev=200 + i,
            )
        )

    # GEO satellites (~1 rev/day)
    for i in range(6):
        sat = 10014 + i
        tles.append(
            _make_tle(
                sat,
                inc=0.1 * (i + 1),
                raan=0.0 + i * 30.0,
                ecc=0.0001 * (i + 1),
                argp=250.0 + i,
                ma=60.0 + i,
                mm=1.0027,
                rev=300 + i,
            )
        )

    return tles


def test_cuda_matches_cpu():
    """Ensure GPU and CPU propagation agree for many TLEs and epochs."""

    if not cuda.is_available():
        pytest.skip("CUDA is not available, skipping GPU test")

    tle_lines = _generate_sample_tles()
    start_times = [
        datetime(1975, 1, 1),
        datetime(2000, 1, 1),
        datetime(2025, 1, 1),
    ]

    timestep_seconds = 60
    total_timesteps = 3

    for start_time in start_times:
        tle_arrays, tles = initialize_tle_arrays_from_lines(tle_lines, start_time)
        num_satellites = tle_arrays.shape[0]

        d_tles = cuda.to_device(tle_arrays.astype(np.float64))
        d_r = cuda.device_array((3, num_satellites, total_timesteps), dtype=np.float64)
        d_v = cuda.device_array((3, num_satellites, total_timesteps), dtype=np.float64)

        blocks_per_grid, threads_per_block = get_optimal_launch_config(num_satellites)
        propagate_orbit[blocks_per_grid, threads_per_block](
            d_tles, d_r, d_v, total_timesteps, timestep_seconds
        )

        r = d_r.copy_to_host()
        v = d_v.copy_to_host()

        for s, tle in enumerate(tles):
            for t_step in range(total_timesteps):
                tsince = tle_arrays[s, tIdx] + t_step * timestep_seconds / 60.0
                r_cpu = [0.0, 0.0, 0.0]
                v_cpu = [0.0, 0.0, 0.0]
                sgp4(tle.rec, tsince, r_cpu, v_cpu)
                np.testing.assert_allclose(r[:, s, t_step], r_cpu, atol=1e-6)
                np.testing.assert_allclose(v[:, s, t_step], v_cpu, atol=1e-6)
