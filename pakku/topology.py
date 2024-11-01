"""
Utilities related to the topology of water networks, such as identifying
water molecules
"""

from typing import List
import numpy as np
from ase import Atom, Atoms
from ase.geometry import get_distances

from pakku.watermolecule import WaterMolecule  # TODO: from . import ... when installed


def get_atom_list(
    atoms: Atoms,
    symbol: str,
    indices: np.ndarray | None = None,
) -> List[Atom]:
    """
    Get a list of Atom objects of a single element. It is possible to
    specify an array of indices from which to select the Atoms.

    Note: a list of Atom objects conserves the indices corresponding
    to the original Atoms object, whereas slicing an Atoms object
    does not.
    """
    if indices is not None:
        atom_list = [atoms[i] for i in indices if atoms[i].symbol == symbol]
    else:
        atom_list = [a for a in atoms if a.symbol == symbol]
    return atom_list


def get_atom_list_from_indices(atoms: Atoms, indices: np.ndarray):
    """
    Get a list of Atom objects from an array of indices
    """
    return [atoms[i] for i in indices]


def get_atom_indices(atoms: Atoms, symbol: str) -> np.ndarray:
    """
    Get indices of a certain type
    """
    return np.nonzero(atoms.symbols == symbol)


def identify_water_molecules(
    atoms: Atoms,
    oxygen_indices: np.ndarray | None = None,
    hydrogen_indices: np.ndarray | None = None,
) -> List[WaterMolecule]:
    """
    Assign water molecules to their closest oxygen atom to form
    WaterMolecule objects.
    """
    hydrogen_atoms = get_atom_list(atoms, "H", hydrogen_indices)
    oxygen_atoms = get_atom_list(atoms, "O", oxygen_indices)

    _, oh_distances = get_distances(
        np.array([h.position for h in hydrogen_atoms]),
        np.array([o.position for o in oxygen_atoms]),
        cell=atoms.cell,
        pbc=atoms.pbc,
    )

    nearest_o_indices = np.argmin(oh_distances, axis=1)

    water_molecules = [WaterMolecule(o) for o in oxygen_atoms]

    for idx, h in zip(nearest_o_indices, hydrogen_atoms):
        water_molecules[idx].add_hydrogen(h)

    return water_molecules
