"""
Tools for getting water structure profile plots
"""

from typing import Tuple
import numpy as np

from pakku.physics import water_density  # TODO: fix for packaging


def bin_edges_to_grid(bin_edges: np.ndarray):
    """
    Get the center of each bin, to plot each bin as a point in
    a line plot
    """
    return bin_edges[:-1] + np.diff(bin_edges) / 2


def rho_cos_theta_profile(
    z_positions: np.ndarray,
    cos_theta: np.ndarray,
    n_frames: int,
    z_max: float,
    bin_size: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    TODO: to write
    """
    cos_theta[np.isnan(cos_theta)] = 0

    counts, bin_edges = np.histogram(
        z_positions.flatten(),
        bins=int(z_max / bin_size),
        range=(0, z_max),
        weights=cos_theta,
    )

    x = bin_edges_to_grid(bin_edges)
    rho_cos_theta = counts / n_frames
    return x, rho_cos_theta


def water_density_profile(
    z_positions: np.ndarray,
    n_frames: int,
    area: float,
    z_max: float,
    bin_size: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    TODO: write
    """
    counts, bin_edges = np.histogram(
        z_positions.flatten(),
        bins=int(z_max / bin_size),
        range=(0, z_max),
    )
    x = bin_edges_to_grid(bin_edges)
    rho = water_density(counts / n_frames, area * bin_size)
    return x, rho
