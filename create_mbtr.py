from dscribe.descriptors import MBTR
from ase.build import bulk, molecule
from ase import Atoms
import numpy as np 


def create_star_carbon_structure(n=5, bond_length=1.34):
    """
    Creates a star structure: one central carbon atom with n outer carbons equally spaced in a circle.
    """
    positions = [(0, 0, 0)]  # central carbon
    for i in range(n):
        angle = 2 * np.pi * i / n
        x = bond_length * np.cos(angle)
        y = bond_length * np.sin(angle)
        positions.append((x, y, 0))
    return Atoms('C' * (n + 1), positions=positions)




def create_linear_carbon_chain(n=5, bond_length=1.34):
    """
    Creates a linear chain of n carbon atoms (C=C=...=C) with specified bond length.
    """
    positions = [(i * bond_length, 0, 0) for i in range(n)]
    return Atoms('C' * n, positions=positions)

# Example usage for linear chain:
if __name__ == "__main__":
    linear_carbons = create_linear_carbon_chain(n=5)
    star_carbons = create_star_carbon_structure(n=5)

    print("Linear carbon chain positions:", linear_carbons.get_positions())
    print("Star carbon structure positions:", star_carbons.get_positions())

    # Initialize MBTR descriptor for carbon
    mbtr = MBTR(
        species=["C"],
        k1={"geometry": {"function": "atomic_number"}},
        k2={"geometry": {"function": "distance"}},
        k3={"geometry": {"function": "angle"}},
        periodic=False,
        flatten=False,
    )

    mbtr_linear = mbtr.create(linear_carbons)
    mbtr_star = mbtr.create(star_carbons)

    print("MBTR for linear carbon chain:", mbtr_linear)
    print("MBTR for star carbon structure:", mbtr_star)