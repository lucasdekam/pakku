"""
Utilities related to the topology of water networks, such as identifying
water molecules
"""

from typing import List, Tuple
import numpy as np
from ase import Atom, Atoms
from ase.geometry import get_distances, get_layers

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


def atom_symbol_assertion(atom: Atom, symbol: str) -> None:
    """
    Check whether atom has a certain element symbol
    """
    assert atom.symbol == symbol, (
        f"Tried updating {atom} with an Atom that is not {symbol}."
        "Indices of different frames might correspond to different atoms."
    )


def update_water_molecules(
    atoms: Atoms,
    water_molecules=List[WaterMolecule],
) -> None:
    """
    Update positions of the Atom objects in the WaterMolecules, using
    the same indices as before
    """
    for molecule in water_molecules:
        atom_symbol_assertion(atoms[molecule.index], "O")
        molecule.oxygen.position = atoms[molecule.index].position
        for h_atom, h_ind in zip(molecule.hydrogens, molecule.h_indices):
            atom_symbol_assertion(atoms[h_ind], "H")
            h_atom.position = atoms[h_ind].position


def get_surface(
    atoms: Atoms,
    surface_tags: Tuple[int] | int,
    miller: Tuple[int],
    tolerance: float = 1.4,
    symbol: str = "Pt",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts the indices and coordinates of surface layers of specified metal atoms from a
    given Atoms object.

    Parameters
    ----------
    atoms : Atoms
    surface_tags : Tuple[int] | int
        The tags of the surface layers to extract. This can be a single integer or a tuple
        of integers. For example, if there are 4 layers and the second and third layers are
        surfaces, the tuple should be [1, 2]. So: zero-based indexing starting from the
        lowest layer.
    miller : Tuple[int]
        A tuple representing the Miller indices of the plane(s) for which to find the
        surface layers. For 111-layers stacked in the z-direction: (0, 0, 1).
        (because the supercell is different from the primitive cell)
    tolerance : float, optional
        The maximum distance in Angstrom along the plane normal for
        counting two atoms as belonging to the same plane. Default is 1.4.
    symbol : str, optional
        The chemical symbol of the metal atoms to be considered. Default is "Pt".

    Returns
    -------
    surface_indices : np.ndarray
        A 1D array containing the indices of the atoms that belong to the specified surface
        layers.
    coord : np.ndarray
        A 1D array of coordinates representing the positions of the identified surface layers,
        relative to the center point between the two surfaces. If there's only one surface,
        the array only contains the position of that surface relative to the zero of
        atoms.cell.
    """
    assert symbol in atoms.symbols, f"{symbol} not found in atoms.symbols"
    indices = np.argwhere(atoms.symbols == symbol).squeeze()
    metal = atoms[indices]
    tags, layers = get_layers(metal, miller=miller, tolerance=tolerance)

    surface_indices = []
    for t in surface_tags:
        surface_indices.append(indices[tags == t])

    coord = np.array([layers[i] for i in surface_tags])
    reference = min(coord) + 0.5 * (max(coord) - min(coord))
    coord -= reference

    return np.array(surface_indices), coord
