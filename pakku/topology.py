"""
Utilities related to the topology of water networks, such as identifying
water molecules
"""

from typing import List
import numpy as np
from ase import Atom, Atoms
from ase.geometry import get_distances
from ase.cell import Cell

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
    oxygen_atoms: List[Atom] | Atoms,
    hydrogen_atoms: List[Atom] | Atoms,
    cell: Cell,
    pbc,
) -> List[WaterMolecule]:
    """
    Assign water molecules to their closest oxygen atom to form
    WaterMolecule objects.
    """

    water_molecules = [WaterMolecule(o) for o in oxygen_atoms]

    for h in hydrogen_atoms:
        _, oh_distances = get_distances(
            h.position,
            np.array([o.position for o in oxygen_atoms]),
            cell=cell,
            pbc=pbc,
        )
        index_o = np.argmin(oh_distances)
        water_molecules[index_o].add_hydrogen(h)

    return water_molecules
