from dscribe.descriptors import MBTR
from ase.build import bulk, molecule
import numpy as np 



def main():
    # Create a bulk structure
    bulk_structure = bulk('Si', 'diamond', a=5.43)

    # Create a molecule structure
    molecule_structure = molecule('H2O')

    # Initialize the MBTR descriptor
    mbtr = MBTR(
        species=["Si", "H", "O"],
        k1={"geometry": {"function": "atomic_number"}},
        k2={"geometry": {"function": "distance"}},
        k3={"geometry": {"function": "angle"}},
        periodic=True,
        flatten=True,
    )

    # Compute the MBTR descriptor for the bulk structure
    mbtr_bulk = mbtr.create(bulk_structure)
    print("MBTR for bulk structure:", mbtr_bulk)

    # Compute the MBTR descriptor for the molecule structure
    mbtr_molecule = mbtr.create(molecule_structure)
    print("MBTR for molecule structure:", mbtr_molecule)