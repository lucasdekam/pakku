"""
Analysis of water simulation trajectories
"""

from typing import Tuple, Iterable
import numpy as np
from ase import Atoms
from tqdm import tqdm

from pakku import topology  # TODO: change when packaging


def collect_positions(
    reader: Iterable[Atoms],
    strict: bool = False,
    oxygen_indices: np.ndarray | None = None,
    hydrogen_indices: np.ndarray | None = None,
) -> Tuple[np.ndarray, int]:
    """
    Collect coordinates of water-like oxygen atoms over an entire trajectory.

    Parameters:
        reader (Iterable[ase.Atoms]): Trajectory frames reader, such as from ase.io.iread.
        oxygen_indices (np.ndarray | None): Optional list of indices for oxygen atoms
            to include in water molecule identification.
        hydrogen_indices (np.ndarray | None): Optional list of indices for hydrogen atoms
            to include in water molecule identification.
        strict (bool): If True, re-identifies water molecules in every frame.
            Improves accuracy but increases runtime (default is False).

    Returns:
        Tuple: (oxygen positions as list of 3-element np.ndarrays, number of frames analyzed).
    """
    water_positions = []
    n_frames = 0

    for atoms in tqdm(reader):
        if n_frames == 0 or strict:
            water_molecules = topology.identify_water_molecules(
                atoms,
                oxygen_indices,
                hydrogen_indices,
            )
            water_indices = [w.index for w in water_molecules if w.is_waterlike()]

        water_positions += [atoms[i].position for i in water_indices]
        n_frames += 1

    return np.array(water_positions), n_frames


def collect_costheta(
    reader: Iterable[Atoms],
    axis: int = 2,
    strict: bool = False,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Collect coordinates of oxygen atoms over an entire trajectory and calculate
    cos(theta) for H2O molecules with respect to a specified axis.

    Parameters:
        reader (Iterable[ase.Atoms]): Trajectory frames reader, such as from ase.io.iread.
        axis (int): Axis (0=x, 1=y, 2=z) for cos(theta) projection.
        strict (bool): If True, re-identifies water molecules in every frame.
            Improves accuracy but increases runtime (default is False).
        **kwargs: Optional arguments for specific atom indices, including:
            - 'oxygen_indices' (np.ndarray | None): Indices of oxygen atoms
              to include in water molecule identification.
            - 'hydrogen_indices' (np.ndarray | None): Indices of hydrogen atoms
              to include in water molecule identification.
            - 'surface_indices' (np.ndarray | None): Indices of surface atoms,
              shape (N x M), with N the number of surfaces per cell
              and M the number of atoms per layer. Can be found using
              `pakku.topology.get_surface`.

    Returns:
        Tuple: (oxygen positions, cos(theta) values with np.nan for non-H2O molecules,
                number of frames analyzed).
    """
    # Extract specific indices from kwargs, if provided
    surface_indices = kwargs.get("surface_indices")

    water_positions = []
    cos_theta_values = []
    n_frames = 0

    for atoms in tqdm(reader):
        # Get surface position
        z_reference = topology.get_cell_reference_coord(atoms, surface_indices, axis)

        # Identify water molecules in the first frame or if strict checking is enabled
        if n_frames == 0 or strict:
            water_molecules = topology.identify_water_molecules(
                atoms,
                kwargs.get("oxygen_indices"),
                kwargs.get("hydrogen_indices"),
            )
        else:
            topology.update_water_molecules(atoms, water_molecules)

        # Collect positions and orientation information
        for molecule in water_molecules:
            z_coord, cos_theta = molecule.cos_theta(
                cell=atoms.cell, pbc=atoms.pbc, axis=axis
            )
            water_positions.append(z_coord - z_reference)
            cos_theta_values.append(cos_theta)

        n_frames += 1

    return np.array(water_positions), np.array(cos_theta_values), n_frames
