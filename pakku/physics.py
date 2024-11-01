"""
Simple physics utilities, such as computing the density or dipole orientation vector.
"""

# import numpy as np
from scipy import constants


def density(n, v, mol_mass: float):
    """
    calculate density (g/cm^3) from the number of particles

    Parameters
    ----------
    n : int or array
        number of particles
    v : float or array
        volume
    mol_mass : float
        mole mass in g/mol
    """
    rho = (n / constants.Avogadro * mol_mass) / (
        v * (constants.angstrom / constants.centi) ** 3
    )
    return rho


def water_density(n, v):
    """
    calculate the water density (g/cm^3)

    Parameters
    ----------
    n : int or array
        number of particles
    v : float or array
        volume
    """
    return density(n, v, 18.015)
