"""
Implementation of a WaterMolecule class
"""

from typing import List
import numpy as np
from ase import Atom, Atoms
from ase.geometry import get_distances


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

    def get_orientation(self, cell, pbc) -> np.ndarray:
        """
        Get the dipole vector
        """
        assert len(self.hydrogens) == 2, (
            "This water molecule does not have two hydrogens, so a dipole vector "
            "cannot be computed. Select for water molecules with the "
            "WaterMolecule.is_h2o() method. "
        )

        # Compute bond vector
        bond_vecs = [
            get_distances(
                p1=self.oxygen.position,
                p2=h.position,
                cell=cell,
                pbc=pbc,
            )[0].squeeze()
            for h in self.hydrogens
        ]
        bisector = bond_vecs[0] + bond_vecs[1]
        return bisector

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
