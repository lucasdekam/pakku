"""
Analysis of water simulation trajectories
"""

from typing import Tuple, Iterable
import numpy as np
from ase import Atoms
from ase.geometry import conditional_find_mic
from tqdm import tqdm

from pakku.topology import identify_water_molecules  # TODO: change when packaging


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
    oxygen_indices: np.ndarray | None = None,
    hydrogen_indices: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Collect oxygen atom coordinates for H2O molecules, calculate cos(theta) w.r.t. specified axis.

    Parameters:
        reader (Iterable[Atoms]): Trajectory frames.
        axis (int): Axis (0=x, 1=y, 2=z) for cos(theta) projection.
        strict (bool): Re-identify water molecules every frame if True.

    Returns:
        Tuple: (oxygen positions along the specified axis, cos(theta), number of frames).
    """
    water_positions = []
    orientation = []
    n_frames = 0

    for atoms in tqdm(reader):
        if n_frames == 0 or strict:
            water_molecules = identify_water_molecules(
                atoms,
                oxygen_indices,
                hydrogen_indices,
            )
            water_indices = [w.indices for w in water_molecules if w.is_h2o()]

        water_positions.extend([atoms[i].position for i, _, _ in water_indices])
        orientation.extend(compute_orientation_vectors(atoms, water_indices))
        n_frames += 1

    # Project dipole vector on main axis
    orientation = np.array(orientation)
    cos_theta = orientation[:, axis] / np.linalg.norm(orientation, axis=-1)

    return np.array(water_positions)[:, axis], cos_theta, n_frames


def compute_orientation_vectors(atoms: Atoms, water_indices: list) -> list:
    """
    Compute orientation vectors for given water molecule indices, considering minimum image
    convention.

    Parameters:
        atoms (ase.Atoms): Structure that includes water molecules.
        water_indices (list): Lists of indices for oxygen and two hydrogens in each water molecule.

    Returns:
        np.ndarray: Orientation vectors for each water molecule.
    """
    o_h1_list, _ = conditional_find_mic(
        np.array([atoms[h].position - atoms[o].position for o, h, _ in water_indices]),
        atoms.cell,
        atoms.pbc,
    )
    o_h2_list, _ = conditional_find_mic(
        np.array([atoms[h].position - atoms[o].position for o, _, h in water_indices]),
        atoms.cell,
        atoms.pbc,
    )
    return [o_h1 + o_h2 for o_h1, o_h2 in zip(o_h1_list, o_h2_list)]
