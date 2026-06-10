import h5py
from utils.map_splitter import BlockMeta

class H5Manager:
    
    # --- Group/Dataset Name Constants ---
    DATASET_BUILDINGS = "buildings"
    DATASET_TRANSMITTERS = "transmitters"
    DATASET_RADIOMAP = "radiomap"
    
    # --- Attribute Key Constants ---
    ATTR_BLOCK_NAME = "block_name"
    ATTR_BLOCK_SIZE = "block_size_m"
    ATTR_RESOLUTION = "resolution_m"
    ATTR_OVERLAP = "overlap_m"
    ATTR_HAS_SIMULATION = "has_simulation"

    @staticmethod
    def init_block_file(file_path, meta:BlockMeta, resolution_m: float):
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
        H5Manager.update_simulation_status(file_path)

    @staticmethod
    def update_simulation_status(file_path):
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