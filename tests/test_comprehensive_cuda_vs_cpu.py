import numpy as np
from datetime import datetime, timedelta
from numba import cuda
import pytest
import warnings
from numba.core.errors import NumbaPerformanceWarning

from cuda_sgp4.src.initialize_tle_arrays import initialize_tle_arrays_from_lines
from cuda_sgp4.src.cuda_sgp4 import propagate_orbit, tIdx, get_optimal_launch_config
from cuda_sgp4.src.SGP4 import sgp4

# Suppress CUDA performance warnings for test cases with small workloads
warnings.filterwarnings('ignore', message='.*Grid size.*will likely result in GPU under-utilization.*')
warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)


def _make_tle(
    satnum: int, inc: float, raan: float, ecc: float, argp: float,
    ma: float, mm: float, rev: int = 1, epoch: str = "00179.78495062"
) -> tuple[str, str]:
    """Generate a synthetic TLE with proper formatting."""
    line1 = (
        f"1 {satnum:05d}U 58002B   {epoch}  .00000023  00000-0  28098-4 0  4753"
    )

    line2 = (
        f"2 {satnum:05d} {inc:8.4f} {raan:8.4f} {int(ecc*1e7):07d} "
        f"{argp:8.4f} {ma:8.4f} {mm:11.8f} {rev:05d}"
    )

    assert len(line1) == 69, len(line1)
    assert len(line2) == 69, len(line2)
    return line1, line2


class TestCudaVsCpuComprehensive:
    """Comprehensive test suite for CUDA vs CPU SGP4 implementations."""

    @pytest.fixture(autouse=True)
    def check_cuda(self):
        """Skip all tests if CUDA is not available."""
        if not cuda.is_available():
            pytest.skip("CUDA is not available")

    def _run_comparison(self, tle_lines, start_times, timestep_seconds=60, total_timesteps=3, tolerance=1e-6):
        """Helper method to run CUDA vs CPU comparison."""
        max_errors = []
        
        for start_time in start_times:
            tle_arrays, tles = initialize_tle_arrays_from_lines(tle_lines, start_time)
            num_satellites = tle_arrays.shape[0]

            # CUDA computation
            d_tles = cuda.to_device(tle_arrays.astype(np.float64))
            d_r = cuda.device_array((3, num_satellites, total_timesteps), dtype=np.float64)
            d_v = cuda.device_array((3, num_satellites, total_timesteps), dtype=np.float64)

            blocks_per_grid, threads_per_block = get_optimal_launch_config(num_satellites)
            propagate_orbit[blocks_per_grid, threads_per_block](
                d_tles, d_r, d_v, total_timesteps, timestep_seconds
            )

            r_cuda = d_r.copy_to_host()
            v_cuda = d_v.copy_to_host()

            # CPU computation and comparison
            max_r_error = 0.0
            max_v_error = 0.0
            
            for s, tle in enumerate(tles):
                for t_step in range(total_timesteps):
                    tsince = tle_arrays[s, tIdx] + t_step * timestep_seconds / 60.0
                    r_cpu = [0.0, 0.0, 0.0]
                    v_cpu = [0.0, 0.0, 0.0]
                    sgp4(tle.rec, tsince, r_cpu, v_cpu)
                    
                    # Check for errors
                    if tle.rec.error != 0:
                        continue  # Skip satellites with errors
                    
                    r_diff = np.abs(r_cuda[:, s, t_step] - r_cpu)
                    v_diff = np.abs(v_cuda[:, s, t_step] - v_cpu)
                    
                    max_r_error = max(max_r_error, np.max(r_diff))
                    max_v_error = max(max_v_error, np.max(v_diff))
                    
                    # Assert within tolerance
                    np.testing.assert_allclose(r_cuda[:, s, t_step], r_cpu, atol=tolerance,
                                             err_msg=f"Position mismatch for satellite {s}, timestep {t_step}")
                    np.testing.assert_allclose(v_cuda[:, s, t_step], v_cpu, atol=tolerance,
                                             err_msg=f"Velocity mismatch for satellite {s}, timestep {t_step}")
            
            max_errors.append((max_r_error, max_v_error))
        
        return max_errors

    def test_low_inclination_satellites(self):
        """Test satellites with very low inclinations (edge case that was failing)."""
        tle_lines = []
        
        # Very low inclination satellites (0.1 to 1.0 degrees)
        for i in range(10):
            inc = 0.1 + i * 0.1  # 0.1, 0.2, ..., 1.0 degrees
            tle_lines.append(_make_tle(
                20000 + i,
                inc=inc,
                raan=30.0 + i * 10.0,
                ecc=0.0001 + i * 0.0001,
                argp=250.0 + i,
                ma=60.0 + i * 5.0,
                mm=1.0027,
                rev=300 + i,
            ))

        start_times = [
            datetime(1975, 1, 1),  # Far past (large negative tsince)
            datetime(2000, 1, 1),  # Recent past
            datetime(2025, 1, 1),  # Future
        ]

        max_errors = self._run_comparison(tle_lines, start_times)
        
        # Verify all errors are within reasonable bounds
        for max_r_error, max_v_error in max_errors:
            assert max_r_error < 1e-6, f"Position error too large: {max_r_error}"
            assert max_v_error < 1e-6, f"Velocity error too large: {max_v_error}"

    def test_high_inclination_satellites(self):
        """Test satellites with high inclinations including polar and retrograde orbits."""
        tle_lines = []
        
        # High inclination satellites (80 to 180 degrees)
        inclinations = [80.0, 90.0, 95.0, 100.0, 120.0, 150.0, 179.9]
        for i, inc in enumerate(inclinations):
            tle_lines.append(_make_tle(
                21000 + i,
                inc=inc,
                raan=i * 30.0,
                ecc=0.001 + i * 0.001,
                argp=i * 20.0,
                ma=i * 30.0,
                mm=15.0 - i * 0.5,  # LEO to MEO
                rev=100 + i,
            ))

        start_times = [datetime(2000, 1, 1), datetime(2020, 1, 1)]
        max_errors = self._run_comparison(tle_lines, start_times)
        
        for max_r_error, max_v_error in max_errors:
            assert max_r_error < 1e-6, f"Position error too large: {max_r_error}"
            assert max_v_error < 1e-6, f"Velocity error too large: {max_v_error}"

    def test_high_eccentricity_satellites(self):
        """Test satellites with high eccentricity (elliptical orbits)."""
        tle_lines = []
        
        # High eccentricity satellites
        eccentricities = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
        for i, ecc in enumerate(eccentricities):
            tle_lines.append(_make_tle(
                22000 + i,
                inc=30.0 + i * 10.0,
                raan=i * 45.0,
                ecc=ecc,
                argp=i * 30.0,
                ma=i * 45.0,
                mm=2.0 - i * 0.2,  # Adjust mean motion for higher orbits
                rev=50 + i,
            ))

        start_times = [datetime(2000, 1, 1)]
        max_errors = self._run_comparison(tle_lines, start_times)
        
        for max_r_error, max_v_error in max_errors:
            assert max_r_error < 1e-6, f"Position error too large: {max_r_error}"
            assert max_v_error < 1e-6, f"Velocity error too large: {max_v_error}"

    def test_different_orbital_regimes(self):
        """Test satellites across different orbital regimes (LEO, MEO, GEO, HEO)."""
        tle_lines = []
        
        # LEO satellites
        for i in range(3):
            tle_lines.append(_make_tle(
                23000 + i,
                inc=51.6 + i * 5.0,
                raan=i * 60.0,
                ecc=0.001 + i * 0.001,
                argp=i * 45.0,
                ma=i * 60.0,
                mm=15.5 - i * 0.5,  # ~90-100 minute orbits
                rev=1000 + i,
            ))
        
        # MEO satellites
        for i in range(3):
            tle_lines.append(_make_tle(
                23100 + i,
                inc=55.0 + i * 5.0,
                raan=i * 90.0,
                ecc=0.01 + i * 0.01,
                argp=i * 60.0,
                ma=i * 90.0,
                mm=2.0 + i * 0.1,  # ~12 hour orbits
                rev=500 + i,
            ))
        
        # GEO satellites
        for i in range(3):
            tle_lines.append(_make_tle(
                23200 + i,
                inc=0.5 + i * 0.5,  # Small inclinations
                raan=i * 120.0,
                ecc=0.0001 + i * 0.0001,
                argp=i * 90.0,
                ma=i * 120.0,
                mm=1.0027,  # Geostationary
                rev=100 + i,
            ))

        start_times = [datetime(2000, 1, 1), datetime(2010, 1, 1)]
        max_errors = self._run_comparison(tle_lines, start_times)
        
        for max_r_error, max_v_error in max_errors:
            assert max_r_error < 1e-6, f"Position error too large: {max_r_error}"
            assert max_v_error < 1e-6, f"Velocity error too large: {max_v_error}"

    def test_extreme_time_ranges(self):
        """Test with extreme time ranges (far past and far future)."""
        tle_lines = []
        
        # Mix of different satellites
        satellites = [
            (24000, 51.6, 247.0, 0.001, 130.0, 10.0, 15.0),  # LEO
            (24001, 0.1, 30.0, 0.0002, 251.0, 61.0, 1.0027),  # Low-inc GEO
            (24002, 98.0, 100.0, 0.01, 90.0, 180.0, 14.0),   # Polar LEO
        ]
        
        for i, (satnum, inc, raan, ecc, argp, ma, mm) in enumerate(satellites):
            tle_lines.append(_make_tle(satnum, inc, raan, ecc, argp, ma, mm, rev=100+i))

        # Extreme time ranges
        start_times = [
            datetime(1970, 1, 1),   # Very far past
            datetime(1980, 1, 1),   # Far past
            datetime(2030, 1, 1),   # Future
            datetime(2050, 1, 1),   # Far future
        ]

        max_errors = self._run_comparison(tle_lines, start_times)
        
        for max_r_error, max_v_error in max_errors:
            assert max_r_error < 1e-6, f"Position error too large: {max_r_error}"
            assert max_v_error < 1e-6, f"Velocity error too large: {max_v_error}"

    def test_longer_propagation_times(self):
        """Test with longer propagation times and more timesteps."""
        tle_lines = []
        
        # A few representative satellites
        satellites = [
            (25000, 0.2, 30.0, 0.0002, 251.0, 61.0, 1.0027),  # The problematic low-inc case
            (25001, 51.6, 247.0, 0.001, 130.0, 10.0, 15.0),   # Typical LEO
            (25002, 98.7, 100.0, 0.001, 90.0, 270.0, 14.2),   # Sun-sync
        ]
        
        for i, (satnum, inc, raan, ecc, argp, ma, mm) in enumerate(satellites):
            tle_lines.append(_make_tle(satnum, inc, raan, ecc, argp, ma, mm, rev=100+i))

        start_times = [datetime(2000, 1, 1)]
        
        # Test with longer propagation
        max_errors = self._run_comparison(
            tle_lines, start_times, 
            timestep_seconds=3600,  # 1 hour steps
            total_timesteps=24,     # 24 hours total
            tolerance=1e-5          # Slightly relaxed tolerance for longer propagation
        )
        
        for max_r_error, max_v_error in max_errors:
            assert max_r_error < 1e-5, f"Position error too large: {max_r_error}"
            assert max_v_error < 1e-5, f"Velocity error too large: {max_v_error}"

    def test_boundary_conditions(self):
        """Test boundary conditions and edge cases."""
        tle_lines = []
        
        # Boundary conditions
        boundary_cases = [
            # (satnum, inc, raan, ecc, argp, ma, mm, description)
            (26000, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0027, "All zeros"),
            (26001, 180.0, 359.9, 0.99, 359.9, 359.9, 0.5, "Near maximums"),
            (26002, 0.2, 0.0, 0.0, 0.0, 0.0, 1.0027, "Exact boundary inclination"),
            (26003, 90.0, 180.0, 0.5, 180.0, 180.0, 1.0, "Mid-range values"),
        ]
        
        for satnum, inc, raan, ecc, argp, ma, mm, desc in boundary_cases:
            tle_lines.append(_make_tle(satnum, inc, raan, ecc, argp, ma, mm))

        start_times = [datetime(2000, 1, 1)]
        max_errors = self._run_comparison(tle_lines, start_times)
        
        for max_r_error, max_v_error in max_errors:
            assert max_r_error < 1e-6, f"Position error too large: {max_r_error}"
            assert max_v_error < 1e-6, f"Velocity error too large: {max_v_error}"

    def test_real_world_tles(self):
        """Test with some real-world-like TLE scenarios."""
        # These are based on real satellite characteristics but with synthetic data
        tle_lines = [
            # ISS-like
            _make_tle(27000, 51.6400, 247.4627, 0.0006703, 130.5360, 325.0288, 15.72125391),
            # GPS satellite-like
            _make_tle(27001, 55.0, 20.0, 0.01, 200.0, 5.0, 2.0),
            # Geostationary satellite-like
            _make_tle(27002, 0.1, 75.0, 0.0001, 250.0, 60.0, 1.0027),
            # Molniya orbit-like
            _make_tle(27003, 63.4, 100.0, 0.7, 270.0, 0.0, 2.0),
            # Sun-synchronous-like
            _make_tle(27004, 98.7, 100.0, 0.001, 90.0, 270.0, 14.2),
        ]

        start_times = [datetime(2000, 1, 1), datetime(2020, 1, 1)]
        max_errors = self._run_comparison(tle_lines, start_times)
        
        for max_r_error, max_v_error in max_errors:
            assert max_r_error < 1e-6, f"Position error too large: {max_r_error}"
            assert max_v_error < 1e-6, f"Velocity error too large: {max_v_error}"

    def test_stress_test_many_satellites(self):
        """Stress test with many satellites to ensure scalability."""
        tle_lines = []
        
        # Generate 50 satellites with random-ish parameters
        np.random.seed(42)  # For reproducibility
        
        for i in range(50):
            # Generate varied but reasonable orbital parameters
            inc = np.random.uniform(0.1, 179.9)
            raan = np.random.uniform(0.0, 360.0)
            ecc = np.random.uniform(0.0001, 0.8)
            argp = np.random.uniform(0.0, 360.0)
            ma = np.random.uniform(0.0, 360.0)
            mm = np.random.uniform(0.5, 16.0)  # From GEO to very low LEO
            
            tle_lines.append(_make_tle(28000 + i, inc, raan, ecc, argp, ma, mm, rev=100+i))

        start_times = [datetime(2000, 1, 1)]
        max_errors = self._run_comparison(tle_lines, start_times, total_timesteps=5)
        
        for max_r_error, max_v_error in max_errors:
            assert max_r_error < 1e-6, f"Position error too large: {max_r_error}"
            assert max_v_error < 1e-6, f"Velocity error too large: {max_v_error}"

    def test_performance_comparison(self):
        """Compare performance between CUDA and CPU implementations."""
        import time
        
        # Generate a moderate number of satellites for performance testing
        tle_lines = []
        for i in range(20):
            tle_lines.append(_make_tle(
                29000 + i,
                inc=51.6 + i * 2.0,
                raan=i * 18.0,
                ecc=0.001 + i * 0.001,
                argp=i * 18.0,
                ma=i * 18.0,
                mm=15.0 - i * 0.1,
                rev=100 + i,
            ))

        start_time = datetime(2000, 1, 1)
        tle_arrays, tles = initialize_tle_arrays_from_lines(tle_lines, start_time)
        num_satellites = tle_arrays.shape[0]
        total_timesteps = 10

        # Time CUDA implementation
        d_tles = cuda.to_device(tle_arrays.astype(np.float64))
        d_r = cuda.device_array((3, num_satellites, total_timesteps), dtype=np.float64)
        d_v = cuda.device_array((3, num_satellites, total_timesteps), dtype=np.float64)

        blocks_per_grid, threads_per_block = get_optimal_launch_config(num_satellites)
        
        cuda_start = time.time()
        propagate_orbit[blocks_per_grid, threads_per_block](d_tles, d_r, d_v, total_timesteps, 60)
        cuda.synchronize()  # Ensure GPU computation is complete
        cuda_time = time.time() - cuda_start

        # Time CPU implementation
        cpu_start = time.time()
        for s, tle in enumerate(tles):
            for t_step in range(total_timesteps):
                tsince = tle_arrays[s, tIdx] + t_step * 60 / 60.0
                r_cpu = [0.0, 0.0, 0.0]
                v_cpu = [0.0, 0.0, 0.0]
                sgp4(tle.rec, tsince, r_cpu, v_cpu)
        cpu_time = time.time() - cpu_start

        print(f"\nPerformance comparison:")
        print(f"CUDA time: {cuda_time:.4f} seconds")
        print(f"CPU time: {cpu_time:.4f} seconds")
        if cuda_time > 0:
            print(f"Speedup: {cpu_time/cuda_time:.2f}x")
        else:
            print("CUDA time too small to measure accurately")

        # The test passes if both complete without errors
        # Performance comparison is informational
        assert cuda_time >= 0 and cpu_time > 0, "Both implementations should complete"


if __name__ == "__main__":
    # Run a quick test if executed directly
    test_instance = TestCudaVsCpuComprehensive()
    if cuda.is_available():
        print("Running quick verification test...")
        test_instance.test_low_inclination_satellites()
        print("✓ Low inclination test passed")
        test_instance.test_boundary_conditions()
        print("✓ Boundary conditions test passed")
        print("All quick tests passed! Run with pytest for full test suite.")
    else:
        print("CUDA not available - tests would be skipped") 