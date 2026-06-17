import re
from pathlib import Path
import numpy as np
import h5py
from utils.map_splitter import BlockMeta


class H5Manager:
    """Manages HDF5 file operations for block-based simulation data."""
    
    # --- Group/Dataset Name Constants ---
    DATASET_BUILDINGS = "buildings"
    DATASET_TRANSMITTERS = "transmitters"
    DATASET_RADIOMAP = "radiomap"
    DATASET_SOURCE_MAPPING = "source_mapping"
    
    # --- Attribute Key Constants ---
    ATTR_BLOCK_NAME = "block_name"
    ATTR_BLOCK_SIZE = "block_size_m"
    ATTR_RESOLUTION = "resolution_m"
    ATTR_OVERLAP = "overlap_m"
    ATTR_HAS_SIMULATION = "has_simulation"
    ATTR_DATASET_SIZE = "dataset_size"

    # --- File Name Pattern ---
    FILE_NAME_PATTERN = re.compile(r"^block_\d+_\d+\.h5$")

    @staticmethod
    def init_block_file(file_path, meta: BlockMeta, resolution_m: float):
        """
        Initialize the HDF5 file and write metadata attributes.
        """
        with h5py.File(file_path, "w") as f:
            f.attrs[H5Manager.ATTR_BLOCK_NAME] = meta.name
            f.attrs[H5Manager.ATTR_BLOCK_SIZE] = meta.block_size_m
            f.attrs[H5Manager.ATTR_RESOLUTION] = resolution_m
            f.attrs[H5Manager.ATTR_OVERLAP] = meta.overlap_m
            f.attrs[H5Manager.ATTR_HAS_SIMULATION] = False

    @staticmethod
    def write_dataset(file_path, name: str, data, dtype: str):
        """
        Write or overwrite a dataset by its name, then update file status.
        """
        with h5py.File(file_path, "a") as f:
            if name in f:
                del f[name]
            f.create_dataset(name, data=data, dtype=dtype, compression="gzip")
            
        # Check and update the simulation status automatically
        H5Manager._update_simulation_status(file_path)

    @staticmethod
    def clean_dataset(file_path, name: str):
        """
        clean a dataset by its name, then update file status.
        """
        with h5py.File(file_path, "a") as f:
            if name in f:
                del f[name]
            
        # Check and update the simulation status automatically
        H5Manager._update_simulation_status(file_path)

    @staticmethod
    def _update_simulation_status(file_path):
        """
        Check if all 3 datasets exist. If yes, set has_simulation to True.
        """
        required_datasets = [
            H5Manager.DATASET_BUILDINGS,
            H5Manager.DATASET_TRANSMITTERS,
            H5Manager.DATASET_RADIOMAP
        ]
        
        with h5py.File(file_path, "a") as f:
            # Check if all required datasets are in the file
            all_exist = all(name in f for name in required_datasets)
            f.attrs[H5Manager.ATTR_HAS_SIMULATION] = all_exist

    @staticmethod
    def _read_and_validate_meta(file_path, resolution_m_ref, overlap_m_ref,
                                new_block_size_m):
        """
        Validate a single HDF5 file and read its datasets.
        
        Parameters
        ----------
        file_path : Path
            Path to the HDF5 file.
        resolution_m_ref : float or None
            Reference resolution from previously processed files.
        overlap_m_ref : int or None
            Reference overlap from previously processed files.
        new_block_size_m : int
            Target block size in meters.
            
        Returns
        -------
        dict or None
            {'buildings': ndarray, 'transmitters': ndarray, 'radiomap': ndarray,
             'resolution_m': float, 'overlap_m': int, 'old_block_size': int, block_name': str}
            Returns None if the file should be skipped (invalid name or
            has_simulation=False).
        
        Raises
        ------
        ValueError
            If resolution_m or overlap_m mismatches reference values, or
            block_size_m divisibility condition fails.
        KeyError
            If required attributes are missing from the file.
        """
        # Validate file name against expected pattern
        if not H5Manager.FILE_NAME_PATTERN.match(file_path.name):
            print(f"  [SKIP] Invalid name pattern: {file_path.name}")
            return None

        with h5py.File(file_path, "r") as f:
            # Check simulation flag — raises KeyError if attribute is missing
            has_sim = f.attrs[H5Manager.ATTR_HAS_SIMULATION]
            if not has_sim:
                print(f"  [SKIP] has_simulation=False: {file_path.name}")
                return None
            
            # --- Get block name ---
            block_name = f.attrs[H5Manager.ATTR_BLOCK_NAME]

            # --- Validate resolution ---
            resolution_m = f.attrs[H5Manager.ATTR_RESOLUTION]
            if resolution_m_ref is not None and resolution_m != resolution_m_ref:
                raise ValueError(
                    f"Resolution mismatch in '{file_path.name}': "
                    f"expected {resolution_m_ref}, got {resolution_m}"
                )

            # --- Validate overlap ---
            overlap_m = f.attrs[H5Manager.ATTR_OVERLAP]
            if overlap_m_ref is not None and overlap_m != overlap_m_ref:
                raise ValueError(
                    f"Overlap mismatch in '{file_path.name}': "
                    f"expected {overlap_m_ref}, got {overlap_m}"
                )

            # --- Validate block_size_m divisibility ---
            old_block_size = f.attrs[H5Manager.ATTR_BLOCK_SIZE]
            if old_block_size < new_block_size_m:
                raise ValueError(
                    f"Block size in '{file_path.name}' ({old_block_size}) "
                    f"is smaller than target size ({new_block_size_m})"
                )
            if old_block_size % new_block_size_m != 0:
                raise ValueError(
                    f"Target block size ({new_block_size_m}) does not divide "
                    f"block size in '{file_path.name}' ({old_block_size})"
                )

            # --- Read datasets ---
            buildings = f[H5Manager.DATASET_BUILDINGS][:]
            transmitters = f[H5Manager.DATASET_TRANSMITTERS][:]
            radiomap = f[H5Manager.DATASET_RADIOMAP][:]

        return {
            "buildings": buildings,
            "transmitters": transmitters,
            "radiomap": radiomap,
            "resolution_m": resolution_m,
            "overlap_m": overlap_m,
            "old_block_size": old_block_size,
            "block_name": block_name,
        }

    @staticmethod
    def _split_and_filter_blocks(datasets, old_block_size, new_block_size_m, block_name):
        """
        Split datasets into smaller blocks, discarding those where the
        transmitters block is all zeros.
        
        Parameters
        ----------
        datasets : dict
            Keys 'buildings', 'transmitters', 'radiomap' with 2D ndarrays
            of shape (old_block_size, old_block_size).
        old_block_size : int
            Original block size in meters.
        new_block_size_m : int
            Target block size in meters.
        block_name : str
            Name of the source block.
            
        Returns
        -------
        tuple of lists
            (buildings_list, transmitters_list, radiomap_list, source_names_list)
            Each list contains 2D ndarrays of shape (new, new).
        """
        buildings_list = []
        transmitters_list = []
        radiomap_list = []
        source_names_list = []

        blocks_per_side = old_block_size // new_block_size_m
        new_size = new_block_size_m

        for i in range(blocks_per_side):
            r_start = i * new_size
            r_end = r_start + new_size
            for j in range(blocks_per_side):
                c_start = j * new_size
                c_end = c_start + new_size

                tx_block = datasets["transmitters"][r_start:r_end, c_start:c_end]

                # Discard blocks where transmitters are entirely zero
                if np.all(tx_block == 0):
                    continue

                buildings_list.append(
                    datasets["buildings"][r_start:r_end, c_start:c_end]
                )
                transmitters_list.append(tx_block)
                radiomap_list.append(
                    datasets["radiomap"][r_start:r_end, c_start:c_end]
                )
                source_names_list.append(block_name)

        return buildings_list, transmitters_list, radiomap_list, source_names_list

    @staticmethod
    def _write_merged_file(output_path, new_block_size_m, resolution_m,
                           overlap_m, total_samples, buildings_array,
                           transmitters_array, radiomap_array, source_names_array):
        """
        Write the merged HDF5 file.
        
        Parameters
        ----------
        output_path : Path
            Output file path.
        new_block_size_m : int
            Target block size in meters.
        resolution_m : float
            Resolution inherited from source files.
        overlap_m : int
            Overlap inherited from source files.
        total_samples : int
            Number of valid samples (N).
        buildings_array : ndarray
            Buildings data, shape (N, new, new), dtype uint8.
        transmitters_array : ndarray
            Transmitters data, shape (N, new, new), dtype uint8.
        radiomap_array : ndarray
            Radiomap data, shape (N, new, new), dtype float32.
        source_names_array : ndarray
            Source block names, shape (N,), dtype string.
        """
        with h5py.File(output_path, "w") as f:
            # Write attributes
            f.attrs[H5Manager.ATTR_BLOCK_NAME] = "block"
            f.attrs[H5Manager.ATTR_BLOCK_SIZE] = new_block_size_m
            f.attrs[H5Manager.ATTR_RESOLUTION] = resolution_m
            f.attrs[H5Manager.ATTR_OVERLAP] = overlap_m
            f.attrs[H5Manager.ATTR_HAS_SIMULATION] = True
            f.attrs[H5Manager.ATTR_DATASET_SIZE] = np.uint16(total_samples)

            # Write compressed datasets
            f.create_dataset(
                H5Manager.DATASET_BUILDINGS,
                data=buildings_array,
                dtype="uint8",
                compression="gzip",
            )
            f.create_dataset(
                H5Manager.DATASET_TRANSMITTERS,
                data=transmitters_array,
                dtype="uint8",
                compression="gzip",
            )
            f.create_dataset(
                H5Manager.DATASET_RADIOMAP,
                data=radiomap_array,
                dtype="float32",
                compression="gzip",
            )

            f.create_dataset(
                H5Manager.DATASET_SOURCE_MAPPING,
                data=source_names_array,
                dtype=h5py.string_dtype(encoding='utf-8'),
                compression="gzip",
            )

    @staticmethod
    def merge_blocks(input_dir: Path, new_block_size_m: int, output_path: Path):
        """
        Merge multiple block HDF5 files from input_dir into a single output file.
        
        Processes each valid .h5 file, splits its datasets into smaller blocks
        of size new_block_size_m x new_block_size_m, filters out blocks where
        transmitters are all zeros, and writes the merged result to output_path.
        
        Parameters
        ----------
        input_dir : Path
            Directory containing block_*.h5 files.
        new_block_size_m : int
            Target block size in meters (must divide old sizes).
        output_path : Path
            Merged output HDF5 file.
        
        Raises
        ------
        ValueError
            If input_dir does not exist, resolution_m or overlap_m mismatch
            between files, or block_size_m divisibility condition fails.
        KeyError
            If a required attribute is missing from a source file.
        """
        if not input_dir.is_dir():
            raise ValueError(f"Input directory does not exist: {input_dir}")

        resolution_m_ref = None
        overlap_m_ref = None
        all_buildings = []
        all_transmitters = []
        all_radiomap = []
        all_source_names = []

        h5_files = sorted(input_dir.glob("*.h5"))

        for file_path in h5_files:
            print(f"Processing: {file_path.name}")

            # Validate and read source file
            result = H5Manager._read_and_validate_meta(
                file_path, resolution_m_ref, overlap_m_ref, new_block_size_m
            )
            if result is None:
                continue

            # Store reference values from the first valid file
            if resolution_m_ref is None:
                resolution_m_ref = result["resolution_m"]
                overlap_m_ref = result["overlap_m"]

            # Split into smaller blocks and filter all-zero transmitters
            b_list, t_list, r_list, s_list = H5Manager._split_and_filter_blocks(
                {
                    "buildings": result["buildings"],
                    "transmitters": result["transmitters"],
                    "radiomap": result["radiomap"],
                },
                result["old_block_size"],
                new_block_size_m,
                result["block_name"],
            )

            all_buildings.extend(b_list)
            all_transmitters.extend(t_list)
            all_radiomap.extend(r_list)
            all_source_names.extend(s_list)

        # --- Check for valid samples ---
        total_samples = len(all_buildings)
        if total_samples == 0:
            print("No valid samples found. Output file will not be created.")
            return

        # --- Stack into 3D arrays ---
        buildings_array = np.stack(all_buildings, axis=0)
        transmitters_array = np.stack(all_transmitters, axis=0)
        radiomap_array = np.stack(all_radiomap, axis=0)
        source_names_array = np.array(all_source_names, dtype=object)

        # --- Write merged output ---
        H5Manager._write_merged_file(
            output_path,
            new_block_size_m,
            resolution_m_ref,
            overlap_m_ref,
            total_samples,
            buildings_array,
            transmitters_array,
            radiomap_array,
            source_names_array,
        )

        print(
            f"Successfully merged {total_samples} samples into {output_path}"
        )

    @staticmethod
    def get_source_block_name(merged_file_path: Path, index: int) -> str:
        with h5py.File(merged_file_path, "r") as f:
            if H5Manager.DATASET_SOURCE_MAPPING not in f:
                raise KeyError(
                    f"Dataset '{H5Manager.DATASET_SOURCE_MAPPING}' not found in file."
                )
            source_mapping = f[H5Manager.DATASET_SOURCE_MAPPING]
            if index >= len(source_mapping):
                raise IndexError(
                    f"Index {index} out of range. Dataset size: {len(source_mapping)}"
                )
            return source_mapping[index].decode('utf-8')