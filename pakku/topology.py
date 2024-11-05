"""
Utilities related to the topology of water networks, such as identifying
water molecules
"""

from typing import List, Tuple, Optional
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
    surface_layers: Tuple[int] | int,
    axis: int = 2,
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
    axis : int, default 2
        The axis perpendicular to the surface
    tolerance : float, optional
        The maximum distance in Angstrom between planes for counting two atoms as belonging
        to the same plane. Default is 1.4 (~half the distance between Pt and Au layers).
    symbol : str, optional
        The chemical symbol of the metal atoms to be considered. Default is "Pt".

    Returns
    -------
    surface_indices : np.ndarray
        A 2D array containing the indices of the atoms that belong to the specified surface
        layers. Each row corresponds to a different surface.
    coord : np.ndarray
        A 1D array of coordinates representing the positions of the identified surface layers,
        relative to the center point between the two surfaces. If there's only one surface,
        the array only contains 0.
    """
    assert symbol in atoms.symbols, f"{symbol} not found in atoms.symbols"
    indices = np.argwhere(atoms.symbols == symbol).squeeze()
    metal = atoms[indices]

    # Need to define the Miller indices of the plane(s) for which to find the
    # surface layers. For 111-layers stacked in the z-direction: (0, 0, 1)
    # (because the supercell is different from the primitive cell).
    miller = np.zeros(3)
    miller[axis] = 1
    tags, layers = get_layers(metal, miller=miller, tolerance=tolerance)

    surface_layers = np.atleast_1d(surface_layers)
    surface_indices = []
    for t in surface_layers:
        surface_indices.append(indices[tags == t])
    surface_indices = np.array(surface_indices)

    surface_coords = np.array([layers[i] for i in surface_layers])
    surface_coords -= get_cell_reference_coord(atoms, surface_indices, axis)

    return surface_indices, surface_coords


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
