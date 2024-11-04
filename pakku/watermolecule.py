"""
Implementation of a WaterMolecule class
"""

from typing import List, Tuple, Optional
import numpy as np
from ase import Atom, Atoms
from ase.geometry import find_mic


class WaterMolecule:
    """
    Class to keep track of water molecules. Uses 'Atom' objects to keep the 'index'
    attribute.

    To account for water (auto-)ionization, WaterMolecules consist of an oxygen Atom
    and any number of hydrogen Atom objects.

    Parameters
    ----------
    oxygen: ase.Atom
        single Atom object for the oxygen atom
    hydrogens: list of Atom objects

    """

    def __init__(
        self,
        oxygen: Atom,
        hydrogens: Optional[List[Atom]] = None,
    ):
        self.oxygen = oxygen
        if hydrogens is not None:
            self.hydrogens = hydrogens
        else:
            self.hydrogens = []

    @property
    def position(self) -> np.ndarray:
        """
        Get the position of the oxygen atom of the water molecule
        """
        return self.oxygen.position

    @property
    def index(self) -> int:
        """
        Get the Atom index of the oxygen atom
        """
        return self.oxygen.index

    @property
    def h_indices(self) -> List[int]:
        """
        Get the Atom indices of the hydrogen atoms
        """
        return [h.index for h in self.hydrogens]

    def add_hydrogen(self, hydrogen: Atom):
        """
        Append a hydrogen Atom to the WaterMolecule.
        """
        self.hydrogens.append(hydrogen)

    def to_ase_atoms(self, cell: Optional[np.ndarray] = None, pbc: np.ndarray = False):
        """
        Convert the WaterMolecule to an ase.Atoms object (loss of Atom indices).
        """
        return Atoms([self.oxygen] + self.hydrogens, cell=cell, pbc=pbc)

    def is_h2o(self):
        """
        Returns True if the object has two hydrogens, otherwise False
        """
        return len(self.hydrogens) == 2

    def is_waterlike(self):
        """
        Returns True if the object has one, two or three hydrogens
        """
        return 1 <= len(self.hydrogens) <= 3

    def cos_theta(
        self,
        axis: int = 2,
        cell: Optional[np.ndarray] = None,
        pbc: np.ndarray | List[bool] | bool = False,
    ) -> Tuple[float, Optional[float]]:
        """
        Calculate the cosine of the angle (θ) between the orientation vector of a water molecule
        and a specified axis in the simulation cell.

        Parameters
        ----------
        cell : np.ndarray
            The simulation cell matrix, defining the periodic boundaries.
        pbc : np.ndarray or List[bool]
            The periodic boundary conditions along each cell dimension.
        axis : int, optional
            The axis (0, 1, or 2 for x, y, or z) to calculate the cosine projection along.
            Default is 2 (z-axis).

        Returns
        -------
        Tuple[float, Optional[float]]
            - The position along the specified axis (`self.position[axis]`) of the oxygen atom.
            - The cosine of the angle (θ) between the orientation vector and the specified axis.
              Returns `np.nan` if the molecule is not H₂O.
        """
        if self.is_h2o():
            o_h1, _ = find_mic(self.hydrogens[0].position - self.position, cell, pbc)
            o_h2, _ = find_mic(self.hydrogens[1].position - self.position, cell, pbc)
            orientation_vector = o_h1.squeeze() + o_h2.squeeze()
            cos_theta = orientation_vector[axis] / np.linalg.norm(orientation_vector)
        else:
            cos_theta = np.nan
        return self.position[axis], cos_theta

    # TODO: def __len__

    # TODO: def __getitem__

    # TODO: def __iter__
