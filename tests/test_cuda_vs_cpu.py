import numpy as np
from datetime import datetime
from numba import cuda

from cuda_sgp4.src.initialize_tle_arrays import initialize_tle_arrays_from_lines
from cuda_sgp4.src.cuda_sgp4 import propagate_orbit, tIdx
from cuda_sgp4.src.SGP4 import sgp4


def test_cuda_matches_cpu():
    # Sample TLE from sgp4 tests
    line1 = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753"
    line2 = "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667"

    start_time = datetime(2000, 1, 1)
    timestep_seconds = 60
    total_sim_seconds = 180

    tle_arrays, tles = initialize_tle_arrays_from_lines([(line1, line2)], start_time)
    num_satellites = tle_arrays.shape[0]
    total_timesteps = total_sim_seconds // timestep_seconds

    d_tles = cuda.to_device(tle_arrays.astype(np.float64))
    d_r = cuda.device_array((3, num_satellites, total_timesteps), dtype=np.float64)
    d_v = cuda.device_array((3, num_satellites, total_timesteps), dtype=np.float64)

    threads_per_block = 256
    blocks_per_grid = (num_satellites + threads_per_block - 1) // threads_per_block
    propagate_orbit[blocks_per_grid, threads_per_block](d_tles, d_r, d_v, total_timesteps, timestep_seconds)

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
