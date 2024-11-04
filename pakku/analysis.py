"""
Analysis of water simulation trajectories
"""

from typing import Tuple, Iterable, Optional
import numpy as np
from ase import Atoms
from tqdm import tqdm

from pakku.topology import (
    identify_water_molecules,
    update_water_molecules,
)  # TODO: change when packaging


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
            water_molecules = identify_water_molecules(
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
    oxygen_indices: Optional[np.ndarray] = None,
    hydrogen_indices: Optional[np.ndarray] = None,
    surface_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Collect coordinates of oxygen atoms over an entire trajectory and calculate
    cos(theta) for H2O molecules with respect to a specified axis.

    Parameters:
        reader (Iterable[ase.Atoms]): Trajectory frames reader, such as from ase.io.iread.
        axis (int): Axis (0=x, 1=y, 2=z) for cos(theta) projection.
        strict (bool): If True, re-identifies water molecules in every frame.
            Improves accuracy but increases runtime (default is False).
        oxygen_indices (np.ndarray | None): Optional list of indices for oxygen atoms
            to include in water molecule identification.
        hydrogen_indices (np.ndarray | None): Optional list of indices for hydrogen atoms
            to include in water molecule identification.

    Returns:
        Tuple: (oxygen positions, cos(theta) values with np.nan for non-H2O molecules,
                number of frames analyzed).
    """
    water_positions = []
    cos_theta_values = []
    n_frames = 0

    for atoms in tqdm(reader):
        # Get surface position
        z_reference = get_cell_reference_coord(atoms, surface_indices, axis)

        # Identify water molecules in the first frame or if strict checking is enabled
        if n_frames == 0 or strict:
            water_molecules = identify_water_molecules(
                atoms,
                oxygen_indices,
                hydrogen_indices,
            )
        else:
            update_water_molecules(atoms, water_molecules)

        # Collect positions and orientation information
        for molecule in water_molecules:
            z_coord, cos_theta = molecule.cos_theta(
                cell=atoms.cell, pbc=atoms.pbc, axis=axis
            )
            water_positions.append(z_coord - z_reference)
            cos_theta_values.append(cos_theta)

        n_frames += 1

    return np.array(water_positions), np.array(cos_theta_values), n_frames


def get_cell_reference_coord(
    atoms: Atoms,
    surface_indices: Optional[np.ndarray],
    axis: int,
) -> float:
    """
    Determine the reference z-coordinate of a simulation cell.
    If surface indices for two surfaces are provided, finds the midpoint between the
    lowest and highest surface atoms; otherwise, defaults to 0.

    Parameters:
        atoms (Atoms): ASE Atoms object representing the system.
        surface_indices (N x M np.ndarray | None): Indices of surface atoms in the cell,
            where N is the number of surfaces in the cell (1 or 2) and M is the number
            of atoms in the surface.

    Returns:
        float: Reference z-coordinate in the cell.
    """
    if surface_indices is not None:
        surface_indices = np.atleast_2d(surface_indices)
        z_surface = [np.mean(atoms[s].positions[:, axis]) for s in surface_indices]
        return min(z_surface) + 0.5 * (max(z_surface) - min(z_surface))

    return 0.0
