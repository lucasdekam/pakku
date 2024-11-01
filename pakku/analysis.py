"""
Analysis of water simulation trajectories
"""

from typing import Tuple

import numpy as np
from ase.io import iread
from tqdm import tqdm

from pakku.topology import identify_water_molecules, get_atom_list


def get_traj_water_positions(
    traj_filename: str,
    index=None,
    fmt: str = None,
    oxygen_indices: np.ndarray | None = None,
    hydrogen_indices: np.ndarray | None = None,
) -> Tuple[np.ndarray, int]:
    """
    Get the coordinates of water-like oxygen atoms over an entire
    trajectory

    Returns: list of oxygen coordinates (3-element np.ndarrays),
    number of frames analyzed
    """
    oxygen_positions = []
    n_frames = 0

    for atoms in tqdm(iread(traj_filename, index=index, format=fmt)):
        o_atoms = get_atom_list(atoms, "O", oxygen_indices)
        h_atoms = get_atom_list(atoms, "H", hydrogen_indices)

        water_molecules = identify_water_molecules(
            o_atoms,
            h_atoms,
            atoms.cell,
            atoms.pbc,
        )

        oxygen_positions += [w.position for w in water_molecules if w.is_waterlike()]
        n_frames += 1
    # BUG: normalizing count by n_frames gives wrong density
    return np.array(oxygen_positions), n_frames


def get_traj_water_costheta(
    traj_filename: str,
    index=None,
    fmt: str = None,
    axis: int = 2,
    oxygen_indices: np.ndarray | None = None,
    hydrogen_indices: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Get the coordinates of oxygen atoms corresponding to H2O molecules,
    and calculate cos theta (cos of the angle w.r.t. the specified axis).

    Returns: list of oxygen coordinates (3-element np.ndarrays),
    list of cos theta for the corresponding water molecules, number of frames analyzed
    """
    oxygen_positions = []
    orientation_vectors = []
    n_frames = 0

    for atoms in tqdm(iread(traj_filename, index=index, format=fmt)):
        o_atoms = get_atom_list(atoms, "O", oxygen_indices)
        h_atoms = get_atom_list(atoms, "H", hydrogen_indices)

        water_molecules = identify_water_molecules(
            o_atoms,
            h_atoms,
            atoms.cell,
            atoms.pbc,
        )

        oxygen_positions += [w.position for w in water_molecules if w.is_h2o()]
        orientation_vectors += [
            w.get_orientation(
                cell=atoms.cell,
                pbc=atoms.pbc,
            )
            for w in water_molecules
            if w.is_h2o()
        ]
        n_frames += 1

    # Project dipole vector on main axis
    orientation_vectors = np.array(orientation_vectors)
    cos_theta = (orientation_vectors[:, axis]) / np.linalg.norm(
        orientation_vectors, axis=-1
    )
    # BUG: normalizing count by n_frames gives wrong density
    return np.array(oxygen_positions)[:, axis], cos_theta, n_frames
