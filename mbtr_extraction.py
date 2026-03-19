import sys
import argparse
import warnings

# Suppress FutureWarning from dscribe (atoms.get_calculator → atoms.calc)
warnings.filterwarnings("ignore", category=FutureWarning, module="dscribe")

from dscribe.descriptors import MBTR
from ase.db import connect
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from itertools import islice

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# from configs.config import C2DB_PATH as db_path, MBTR_CLASSIFICATION_PATH as output_path, N_JOBS

db_path = "/home/luis-menezes/Documents/MSc/c2db_free_atoms/data/raw/c2db.db"
output_path = "/home/luis-menezes/Documents/MSc/mbtr_studies/data/mbtr_descriptors.h5"
N_JOBS=32

# Define metal threshold
METAL_THRESHOLD = 1e-1  # eV - materials with gap <= 0.1 eV are considered metals
BATCH_SIZE = 512 # Number of structures to process in each batch (adjust based on memory constraints)
N_SAMPLES4TEST = 10000
RESOLUTION = 128  # Number of grid points per unit distance (for k=1 and k=2)
config_k1_dict = {
    "geometry_function": "atomic_number",
    "grid_min": 1,
    "grid_max": 83,
    "grid_n": (83-1+1)*RESOLUTION - (RESOLUTION-1), # (grid max - grid min + 1) * resolution - (resolution-1) (this is so that we can center our gaussian at its peak)
    "normalization": "none",
    "grid_sigma":1
}


# Quero representar distâncias de 0.01-19 Å 
#   A distância média no meu dataset é entre 1.383 - 8.476 Å
#   A distância máxima média é próximo de 19 Å (podem ser de outliers, testar entre 19 e depois até 10 ou 8 Å)

config_k2_dict = {
    "geometry_function": "distance",
    "grid_sigma": 0.05, #0.47499065998783413,
    "grid_n": (6.0-0.5+1)*RESOLUTION - (RESOLUTION-1),
    "grid_max": 6.0, "grid_min": 0.5, # Quero representer 0.01-19 Å -> distância média no meu dataset é 2.44 e máxima média é próximo de 19  
    # "weighting_scale": 1.5, #0.13175274143903676,
    "weighting_function": "exp",
    "weighting_rcut": 5.0,
    "normalization": "none",
    "weighting_threshold": 0.1
}

config_k3_dict = {
    "geometry_function": "cosine",
    "grid_min": -1.0,
    "grid_max": 1.0,
    "grid_n": 6,
    "grid_sigma": 0.1,
    "weighting_scale": 0.5,
    "weighting_threshold": 1e-3,
    # "normalization": "l2",
    # "species": [
    #     "Ag", "Al", "As", "Au", "Bi", "Br", "C",  "Ca", "Cd", "Cl",
    #     "Co", "Cr", "Cu", "F",  "Fe", "Ga", "Ge", "H",  "Hf", "Hg",
    #     "I",  "In", "Ir", "K",  "Li", "Mg", "Mn", "Mo", "N",  "Na",
    #     "Nb", "Ni", "O",  "P",  "Pb", "Pd", "Pt", "Rb", "S",  "Sb",
    #     "Sc", "Se", "Si", "Sn", "Sr", "Ta", "Te", "Ti", "Tl", "V",
    #     "W",  "Zn", "Zr",
    # ],
}

def get_all_species(db_path, selection):
    """
    Get all unique chemical species present in the structures of the database.
    """
    db = connect(db_path)
    print("Starting pre-scan of unique elements...")
    all_elements = set()
    nAtomsMax = 0
    for row in db.select(selection):
        all_elements.update(row.symbols)
        nAtomsMax = max(nAtomsMax, row.natoms)
    print(f"Chemical elements found: {sorted(list(all_elements))}")
    return list(all_elements), nAtomsMax

def generate_mbtr_descriptors(input_db_path, selection, chunk_size, output_h5_path,
                              k_terms=None):
    """
    Generate MBTR descriptors and atomic information for all structures.
    
    Saves to H5 file:
    - MBTR features (structural descriptors)
    - Database indices (for reference)
    - Band gap values (PBE and PBE_noSOC)
    - Atomic numbers for each structure (for atomic models)
    - Binary metal classification
    
    Parameters:
    - input_db_path (str): Path to the input ASE database file.
    - selection (str): Selection query to filter entries in the input database.
    - chunk_size (int): Number of structures to process in each batch.
    - output_h5_path (str): Path to the output H5 file where all data will be saved.
    - k_terms (list[int]): Which MBTR k-terms to include (subset of [1,2,3]).
    """
    if k_terms is None:
        k_terms = [1, 2, 3]

    try: 
        chem_elements, nAtomsMax = get_all_species(input_db_path, selection)
        input_db = connect(input_db_path)   
        n_samples = input_db.count(selection)
        print(f"Number of entries found: {n_samples}")

        n_to_process = min(n_samples, 1000)
        if n_to_process < n_samples:
            print("Undersampling to 1000 samples for testing...")

        db_selection = islice(input_db.select(selection), n_to_process)

    except Exception as e:
        print(f"Error connecting to database or counting entries: {e}")
        return

    configs = {1: config_k1_dict, 2: config_k2_dict, 3: config_k3_dict}

    def _build_mbtr(cfg, default_species):
        species = cfg.get("species", default_species)
        weighting = None
        if "weighting_function" in cfg:
            weighting = {
                "function": cfg.get("weighting_function", "unity"),
                # "scale": cfg["weighting_scale"],
                "threshold": cfg["weighting_threshold"],
                "r_cut": cfg.get("weighting_rcut", 10.0),
            }
        return MBTR(
            geometry={"function": cfg["geometry_function"]},
            grid={
                "min": cfg["grid_min"],
                "max": cfg["grid_max"],
                "n": cfg["grid_n"],
                "sigma": cfg["grid_sigma"],
            },
            species=species,
            periodic=True,
            sparse=False,
            normalization=cfg["normalization"],
            weighting=weighting,
        )

    mbtr_instances = {k: _build_mbtr(configs[k], chem_elements) for k in k_terms}
    n_features_per_k = {k: inst.get_number_of_features() for k, inst in mbtr_instances.items()}
    n_features_total = sum(n_features_per_k.values())

    # Identify k-terms with restricted species for batch filtering
    k_allowed_symbols = {}
    for k in k_terms:
        if "species" in configs[k]:
            k_allowed_symbols[k] = set(configs[k]["species"])

    n_elem = len(chem_elements)
    k_label = " + ".join(f"K{k}={n_features_per_k[k]}" for k in k_terms)
    print(f"Number of chemical species (database): {n_elem}")
    print(f"Selected k-terms: {k_terms}")
    for k in k_terms:
        if k in k_allowed_symbols:
            print(f"  K{k} restricted to {len(k_allowed_symbols[k])} species: {sorted(k_allowed_symbols[k])}")
    print(f"MBTR features per compound: {k_label}, total={n_features_total}")
    print(f"Maximum atoms in structures: {nAtomsMax}")
    print(f"Metal threshold: {METAL_THRESHOLD} eV")

    with h5py.File(output_h5_path, 'w') as h5f:
        # ===== STRUCTURAL DESCRIPTORS =====
        dset_features = h5f.create_dataset('mbtr_features',
                                           shape=(0, n_features_total),
                                           maxshape=(None, n_features_total), 
                                           dtype='float32',
                                           chunks=(chunk_size, n_features_total),
                                           compression='gzip',
                                           compression_opts=6
                                           )

        # ===== TARGET VALUES =====
        dset_gaps = h5f.create_dataset('gap_pbe',
                                       shape=(0,),
                                       maxshape=(None,),
                                       dtype='float32',
                                       chunks=(chunk_size,),
                                       compression='gzip'
                                       )
        
        dset_gaps_nosoc = h5f.create_dataset('gap_pbe_nosoc', 
                                           shape=(0,),
                                           maxshape=(None,),
                                           dtype='float32',
                                           chunks=(chunk_size,),
                                           compression='gzip'
                                           )
        
        dset_is_metal = h5f.create_dataset('is_metal',
                                          shape=(0,),
                                          maxshape=(None,),
                                          dtype='int8',
                                          chunks=(chunk_size,),
                                          compression='gzip'
                                          )
        
        # ===== DATABASE INDICES =====
        dset_db_ids = h5f.create_dataset('db_indices',
                                         shape=(0,),
                                         maxshape=(None,),
                                         dtype='int32',
                                         chunks=(chunk_size,),
                                         compression='gzip'
                                         )
        
        # ===== ATOMIC INFORMATION (for atomic models) =====
        # Variable-length array to store atomic numbers for each structure
        dt_vlen_int = h5py.vlen_dtype(np.dtype('int8'))
        dset_atomic_numbers = h5f.create_dataset('atomic_numbers',
                                                 shape=(0,),
                                                 maxshape=(None,),
                                                 dtype=dt_vlen_int,
                                                 chunks=(chunk_size,),
                                                 compression='gzip'
                                                 )
        
        # Add metadata
        h5f.attrs['n_samples'] = n_samples
        h5f.attrs['n_mbtr_features'] = n_features_total
        h5f.attrs['k_terms'] = k_terms
        for k in k_terms:
            h5f.attrs[f'n_k{k}_features'] = n_features_per_k[k]
            h5f.attrs[f'k{k}_parameters'] = str(configs[k])
        h5f.attrs['max_atoms_per_structure'] = nAtomsMax
        h5f.attrs['metal_threshold'] = METAL_THRESHOLD
        h5f.attrs['selection_query'] = selection
        h5f.attrs['database_path'] = str(input_db_path)
        h5f.attrs['unique_elements'] = sorted(chem_elements)
        
        # Batch accumulators
        batch_compounds = []
        batch_gaps_pbe = []
        batch_gaps_pbe_nosoc = []
        batch_is_metal = []
        batch_db_ids = []
        batch_atomic_numbers = []
        current_index = 0
        
        # Statistics counters
        metal_count = 0
        nonmetal_count = 0
        gap_values = []
        k_skip_counts = {k: 0 for k in k_allowed_symbols}
        n_to_process = min(n_samples, N_SAMPLES4TEST)
        if n_to_process < n_samples:
            print("Undersampling to 1000 samples for testing...")

        db_selection = islice(input_db.select(selection), n_to_process)

        for i, row in enumerate(db_selection):
            try:
                atoms = row.toatoms()
                if atoms is None or len(atoms) == 0:
                    print(f"\nWarning: Skipping entry {i} - invalid atoms object")
                    continue
                    
                batch_compounds.append(atoms)
                batch_db_ids.append(row.id)
                
                # Store atomic numbers (for atomic models)
                atomic_nums = atoms.get_atomic_numbers()
                batch_atomic_numbers.append(atomic_nums.astype(np.int8))
                
                # Get gap values
                gap_pbe_val = getattr(row, 'gap', None)
                batch_gaps_pbe.append(gap_pbe_val if gap_pbe_val is not None else np.nan)
                
                gap_nosoc_val = getattr(row, 'gap_dir_nosoc', None)
                batch_gaps_pbe_nosoc.append(gap_nosoc_val if gap_nosoc_val is not None else np.nan)
                
                # Binary metal classification
                if gap_pbe_val is not None and not np.isnan(gap_pbe_val):
                    is_metal = 1 if gap_pbe_val <= METAL_THRESHOLD else 0
                    if is_metal:
                        metal_count += 1
                    else:
                        nonmetal_count += 1
                    gap_values.append(gap_pbe_val)
                else:
                    is_metal = -1  # Unknown
                
                batch_is_metal.append(is_metal)

                # Process batch when full or at the end
                if len(batch_compounds) == chunk_size or (i + 1) == n_to_process:
                    
                    print(f"\rProcessing batch... {i + 1}/{n_to_process} samples", end="")
                    sys.stdout.flush()

                    try:
                        n_batch = len(batch_compounds)
                        k_str = "+".join(f"K{k}" for k in k_terms)
                        print(f"\n  Computing MBTR ({k_str}) for {n_batch} structures...")

                        parts = []
                        for k in k_terms:
                            inst = mbtr_instances[k]
                            n_feat_k = n_features_per_k[k]

                            if k in k_allowed_symbols:
                                allowed = k_allowed_symbols[k]
                                compat_idx = [
                                    j for j, atoms in enumerate(batch_compounds)
                                    if all(sym in allowed for sym in atoms.get_chemical_symbols())
                                ]
                                feat = np.zeros((n_batch, n_feat_k), dtype=np.float32)
                                if compat_idx:
                                    compat_atoms = [batch_compounds[j] for j in compat_idx]
                                    computed = np.asarray(
                                        inst.create(compat_atoms, n_jobs=N_JOBS),
                                        dtype=np.float32,
                                    )
                                    feat[compat_idx] = computed
                                n_skipped = n_batch - len(compat_idx)
                                k_skip_counts[k] += n_skipped
                                if n_skipped:
                                    print(f"    K{k}: {n_skipped}/{n_batch} structures have out-of-species elements (zeroed)")
                            else:
                                feat = np.asarray(
                                    inst.create(batch_compounds, n_jobs=N_JOBS),
                                    dtype=np.float32,
                                )
                            parts.append(feat)

                        mbtr_features = np.hstack(parts)

                        # Verify dimensions
                        if len(mbtr_features.shape) != 2:
                            print(f"\nError: Unexpected MBTR output shape: {mbtr_features.shape}")
                            continue
                            
                        if mbtr_features.shape[1] != n_features_total:
                            print(f"\nError: Feature dimension mismatch.")
                            print(f"Expected {n_features_total}, got {mbtr_features.shape[1]}")
                            continue
                        
                        if mbtr_features.shape[0] != len(batch_compounds):
                            print(f"\nError: Number of computed descriptors doesn't match batch size.")
                            continue

                        # Check for NaN or infinite values
                        if np.isnan(mbtr_features).any() or np.isinf(mbtr_features).any():
                            print(f"\nWarning: Found NaN or infinite values in MBTR features")
                            nan_count = np.isnan(mbtr_features).sum()
                            inf_count = np.isinf(mbtr_features).sum()
                            print(f"  NaN values: {nan_count}, Inf values: {inf_count}")
                            mbtr_features = np.nan_to_num(mbtr_features, nan=0.0, posinf=0.0, neginf=0.0)

                        chunk_len = mbtr_features.shape[0]
                        new_size = dset_features.shape[0] + chunk_len

                        # Resize all datasets
                        dset_features.resize(new_size, axis=0)
                        dset_gaps.resize(new_size, axis=0)
                        dset_gaps_nosoc.resize(new_size, axis=0)
                        dset_is_metal.resize(new_size, axis=0)
                        dset_db_ids.resize(new_size, axis=0)
                        dset_atomic_numbers.resize(new_size, axis=0)
                        
                        # Store all data
                        dset_features[current_index:new_size] = mbtr_features
                        dset_gaps[current_index:new_size] = batch_gaps_pbe
                        dset_gaps_nosoc[current_index:new_size] = batch_gaps_pbe_nosoc
                        dset_is_metal[current_index:new_size] = batch_is_metal
                        dset_db_ids[current_index:new_size] = batch_db_ids
                        dset_atomic_numbers[current_index:new_size] = batch_atomic_numbers
                        
                        current_index = new_size

                    except Exception as e:
                        print(f"\nError processing batch: {e}")
                        print(f"Batch size: {len(batch_compounds)}")
                        print(f"Sample structures in batch:")
                        for j, atoms in enumerate(batch_compounds[:3]):
                            print(f"  Structure {j}: {len(atoms)} atoms, formula: {atoms.get_chemical_formula()}")
                        import traceback
                        traceback.print_exc()
                        continue

                    # Reset batches
                    batch_compounds = []
                    batch_gaps_pbe = []
                    batch_gaps_pbe_nosoc = []
                    batch_is_metal = []
                    batch_db_ids = []
                    batch_atomic_numbers = []

            except Exception as e:
                print(f"\nError processing structure {i}: {e}")
                continue

        # Final statistics
        total_processed = metal_count + nonmetal_count
        metal_fraction = metal_count / total_processed if total_processed > 0 else 0
        
        print(f"\n\n{'='*80}")
        print(f"MBTR EXTRACTION COMPLETED")
        print(f"{'='*80}")
        print(f"Output file: {output_h5_path}")
        print(f"MBTR features shape: {dset_features.shape}")
        print(f"Successfully processed: {current_index} structures")
        print(f"\n📊 DATASET STATISTICS:")
        print(f"   Metals (gap ≤ {METAL_THRESHOLD} eV): {metal_count} ({metal_fraction*100:.1f}%)")
        print(f"   Non-metals (gap > {METAL_THRESHOLD} eV): {nonmetal_count} ({(1-metal_fraction)*100:.1f}%)")
        print(f"   Total classified: {total_processed}")
        
        if gap_values:
            print(f"\n📈 BAND GAP DISTRIBUTION:")
            print(f"   Min gap: {min(gap_values):.4f} eV")
            print(f"   Max gap: {max(gap_values):.4f} eV")
            print(f"   Mean gap: {np.mean(gap_values):.4f} eV")
            print(f"   Std gap: {np.std(gap_values):.4f} eV")
        
        if k_skip_counts:
            print(f"\n⚠️  RESTRICTED-SPECIES SUMMARY:")
            for k, count in k_skip_counts.items():
                print(f"   K{k}: {count}/{current_index} structures had out-of-species elements (zeroed)")

        feat_breakdown = " | ".join(f"K{k}:{n_features_per_k[k]}" for k in k_terms)
        print(f"\n💾 SAVED DATASETS:")
        print(f"   mbtr_features: {dset_features.shape} - Structural descriptors [{feat_breakdown}]")
        print(f"   gap_pbe: {dset_gaps.shape} - Band gap (PBE)")
        print(f"   gap_pbe_nosoc: {dset_gaps_nosoc.shape} - Band gap (PBE no SOC)")
        print(f"   is_metal: {dset_is_metal.shape} - Binary classification")
        print(f"   db_indices: {dset_db_ids.shape} - Database row IDs")
        print(f"   atomic_numbers: {dset_atomic_numbers.shape} - Atomic numbers per structure")
        print(f"{'='*80}\n")


def verify_h5_file(h5_path):
    """
    Verify the contents of the generated H5 file.
    """
    print(f"\n{'='*80}")
    print(f"VERIFYING H5 FILE: {h5_path}")
    print(f"{'='*80}")
    
    with h5py.File(h5_path, 'r') as h5f:
        print("\n📁 DATASETS:")
        for key in h5f.keys():
            dset = h5f[key]
            print(f"   {key}: {dset.shape}, dtype={dset.dtype}")
        
        print("\n📋 METADATA:")
        for key, value in h5f.attrs.items():
            if isinstance(value, (list, np.ndarray)) and len(str(value)) > 100:
                print(f"   {key}: [long array/list]")
            else:
                print(f"   {key}: {value}")
        
        # Sample a few entries
        print("\n🔍 SAMPLE DATA (first 3 structures):")
        for i in range(min(3, h5f['gap_pbe'].shape[0])):
            atomic_nums = h5f['atomic_numbers'][i]
            gap = h5f['gap_pbe'][i]
            is_metal = h5f['is_metal'][i]
            db_id = h5f['db_indices'][i]
            
            print(f"   Structure {i}:")
            print(f"      DB ID: {db_id}")
            print(f"      Gap: {gap:.4f} eV")
            print(f"      Is metal: {is_metal} ({'Metal' if is_metal == 1 else 'Non-metal' if is_metal == 0 else 'Unknown'})")
            print(f"      Atomic numbers: {atomic_nums} ({len(atomic_nums)} atoms)")
            print(f"      MBTR shape: {h5f['mbtr_features'][i].shape}")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract MBTR descriptors from C2DB database.",
        epilog="Examples:\n"
               "  python %(prog)s -k 2           # K2 only\n"
               "  python %(prog)s -k 1 -k 3      # K1 + K3\n"
               "  python %(prog)s                 # K1 + K2 + K3 (default)\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-k", type=int, action="append", choices=[1, 2, 3],
        dest="k_terms",
        help="K-term to include (repeatable). Default: all (1, 2, 3).",
    )
    args = parser.parse_args()
    k_terms = sorted(set(args.k_terms)) if args.k_terms else [1, 2, 3]

    selection = f"gap>=0"  # Select all non-metals and metals (including unknown gaps)

    generate_mbtr_descriptors(db_path, selection, BATCH_SIZE, output_path, k_terms)

    verify_h5_file(output_path)
