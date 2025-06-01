import numpy as np
from datetime import datetime, timedelta
from numba import cuda
import pytest
import warnings
from numba.core.errors import NumbaPerformanceWarning

try:
    from sgp4.api import Satrec, jday
    SGP4_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency optional for tests
    SGP4_AVAILABLE = False

from cuda_sgp4 import cuda_sgp4

warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)


def _make_tle(
    satnum: int,
    inc: float,
    raan: float,
    ecc: float,
    argp: float,
    ma: float,
    mm: float,
    rev: int = 1,
    epoch: str = "00179.78495062",
) -> tuple[str, str]:
    """Generate a synthetic TLE in the same format as other tests."""
    line1 = (
        f"1 {satnum:05d}U 58002B   {epoch}  .00000023  00000-0  28098-4 0  4753"
    )
    line2 = (
        f"2 {satnum:05d} {inc:8.4f} {raan:8.4f} {int(ecc*1e7):07d} "
        f"{argp:8.4f} {ma:8.4f} {mm:11.8f} {rev:05d}"
    )
    assert len(line1) == 69
    assert len(line2) == 69
    return line1, line2


def _sample_tles() -> list[tuple[str, str]]:
    """Return a small set of TLEs spanning different regimes."""
    return [
        _make_tle(10000, 51.6, 247.0, 0.0001, 130.0, 0.0, 15.0, 100),
        _make_tle(10001, 55.0, 20.0, 0.0005, 200.0, 0.0, 2.0, 200),
        _make_tle(10002, 0.2, 0.0, 0.0001, 250.0, 60.0, 1.0027, 300),
    ]


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(not SGP4_AVAILABLE, reason="sgp4 package not installed")
def test_cuda_matches_standard_sgp4():
    """Verify GPU propagation against the standard python sgp4 package."""
    tle_lines = _sample_tles()
    start_time = datetime(2000, 1, 1)

    timestep_seconds = 60
    total_seconds = 180  # three steps

    positions, velocities = cuda_sgp4(
        tle_lines=tle_lines,
        timestep_length_in_seconds=timestep_seconds,
        total_sim_seconds=total_seconds,
        start_time=start_time,
        return_device_arrays=False,
    )

    n_steps = total_seconds // timestep_seconds

    for i, (l1, l2) in enumerate(tle_lines):
        sat = Satrec.twoline2rv(l1, l2)
        print(f"\nSatellite {i+1}:")
        for step in range(n_steps):
            current = start_time + timedelta(seconds=step * timestep_seconds)
            jd, fr = jday(
                current.year,
                current.month,
                current.day,
                current.hour,
                current.minute,
                current.second + current.microsecond * 1e-6,
            )
            error, r, v = sat.sgp4(jd, fr)
            assert error == 0
            
            print(f"  Step {step+1} (t={step * timestep_seconds}s):")
            print(f"    Position - CUDA: {positions[i, step]}")
            print(f"    Position - SGP4: {r}")
            print(f"    Position - Diff: {np.array(positions[i, step]) - np.array(r)}")
            print(f"    Velocity - CUDA: {velocities[i, step]}")
            print(f"    Velocity - SGP4: {v}")
            print(f"    Velocity - Diff: {np.array(velocities[i, step]) - np.array(v)}")
            
            np.testing.assert_allclose(positions[i, step], r, atol=1e-10)
            np.testing.assert_allclose(velocities[i, step], v, atol=1e-10)
