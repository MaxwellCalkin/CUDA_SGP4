# File: cuda_sgp4/cuda_sgp4.py

import numpy as np
from datetime import datetime, timedelta
from numba import cuda
import hashlib
import os
import logging
from .src.cuda_sgp4 import propagate_orbit
from .src.initialize_tle_arrays import initialize_tle_arrays

def cuda_sgp4(timestep_length_in_seconds, total_sim_seconds, start_time, tle_file_path, output_file_path):
    """
    Propagates satellite orbits using CUDA-accelerated SGP4.

    Parameters:
    - timestep_length_in_seconds (int): Length of each timestep in seconds.
    - total_sim_seconds (int): Total simulation time in seconds.
    - start_time (datetime): Start time of the simulation.
    - tle_file_path (str): Path to the TLE CSV file (must include columns for "line1" "line2" "epoch" and "satNo").
    - output_file_path (str): Path where the output CSV will be saved.

    Returns:
    - None
    """
    try:
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        logger.info("Initializing simulation parameters...")
        total_timesteps = total_sim_seconds // timestep_length_in_seconds

        logger.info("Loading TLE data...")
        tle_arrays, tles = initialize_tle_arrays(tle_file_path, start_time)
        num_satellites = tle_arrays.shape[0]

        logger.info("Allocating device memory...")
        d_tles = cuda.to_device(tle_arrays.astype(np.float64))
        d_r = cuda.device_array((3, num_satellites, total_timesteps), dtype=np.float64)
        d_v = cuda.device_array((3, num_satellites, total_timesteps), dtype=np.float64)

        logger.info("Defining CUDA kernel configuration...")
        threads_per_block = 256
        blocks_per_grid = (num_satellites + (threads_per_block - 1)) // threads_per_block

        logger.info("Starting GPU execution timer...")
        start_event = cuda.event()
        end_event = cuda.event()

        start_event.record()

        logger.info("Launching the kernel...")
        propagate_orbit[blocks_per_grid, threads_per_block](d_tles, d_r, d_v, total_timesteps, timestep_length_in_seconds)

        end_event.record()
        end_event.synchronize()
        gpu_execution_time = cuda.event_elapsed_time(start_event, end_event) / 1000

        logger.info(f"GPU execution time: {gpu_execution_time} seconds for {num_satellites} satellites over {total_timesteps} timesteps.")

        logger.info("Retrieving results from GPU...")
        r = d_r.copy_to_host()
        v = d_v.copy_to_host()

        logger.info("Preparing data for saving...")
        num_timesteps = r.shape[2]
        timestamps = np.arange(0, timestep_length_in_seconds * num_timesteps, timestep_length_in_seconds)
        satnumIdx = 1  # Ensure 'satnum' is index 1 of attributes

        data = create_data(tle_arrays, timestamps, r, v, num_timesteps, num_satellites, satnumIdx)

        logger.info("Saving data to CSV...")
        header_start = f"""HEADER_START
variable_name, value
time_start, {start_time.strftime("%Y-%m-%d %H:%M:%S")}
time_end, {start_time + timedelta(seconds=total_sim_seconds)}
number_of_timesteps, {num_timesteps}
timestep_length_seconds, {timestep_length_in_seconds}
HEADER_END"""
        column_names = 'SatNo,timestamp,x,y,z,vx,vy,vz'
        header = f"{header_start}\n{column_names}"

        fmt = ['%d', '%d'] + ['%.16f'] * 6

        np.savetxt(output_file_path, data, delimiter=',', fmt=fmt, header=header, comments='')

        logger.info(f"Data successfully saved to: {output_file_path}")
        logger.info(f"Simulation started at: {start_time}")
        logger.info(f"Simulation ended at: {start_time + timedelta(seconds=total_sim_seconds)}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


def compute_hash(timestep_length_in_seconds, total_sim_seconds):
    param_string = f"{timestep_length_in_seconds}_{total_sim_seconds}"
    return hashlib.sha256(param_string.encode()).hexdigest()

def create_data(tle_arrays, timestamps, r, v, num_timesteps, num_satellites, satnumIdx):
    data = []
    for t in range(num_timesteps):
        for s in range(num_satellites):
            satnum = int(tle_arrays[s, satnumIdx])
            x, y, z = r[0, s, t], r[1, s, t], r[2, s, t]
            vx, vy, vz = v[0, s, t], v[1, s, t], v[2, s, t]
            data.append([satnum, timestamps[t], x, y, z, vx, vy, vz])
    return np.array(data)
