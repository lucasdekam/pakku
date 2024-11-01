"""
Implementation of a WaterMolecule class
"""

from typing import List
import numpy as np
from ase import Atom, Atoms


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
        hydrogens: List[Atom] | None = None,
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
    def indices(self) -> List[int]:
        """
        Get the Atom indices of the hydrogen atoms
        """
        return [self.oxygen.index] + [h.index for h in self.hydrogens]

    def add_hydrogen(self, hydrogen: Atom):
        """
        Append a hydrogen Atom to the WaterMolecule.
        """
        self.hydrogens.append(hydrogen)

    def to_ase_atoms(self, cell: np.ndarray = None, pbc: np.ndarray = False):
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

    # TODO: def __len__

    # TODO: def __getitem__

    # TODO: def __iter__
