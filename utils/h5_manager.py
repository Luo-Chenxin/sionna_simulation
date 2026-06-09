import h5py
from utils.map_splitter import BlockMeta

class H5Manager:
    @staticmethod
    def create_block_file(file_path, meta:BlockMeta, building_mask):
        """
        Create the HDF5 file and write the initial building mask.
        """
        with h5py.File(file_path, "w") as f:
            # Save building mask
            f.create_dataset("buildings", data=building_mask, dtype="uint8", compression="gzip")
            
            f.attrs["block_name"] = meta.name
            f.attrs["block_size_m"] = meta.block_size_m
            f.attrs["has_simulation"] = False

    @staticmethod
    def append_simulation_results(file_path, transmitters_mask, radiomap):
        """
        Append transmitters and radiomap to the SAME file after Sionna finishes.
        """
        with h5py.File(file_path, "a") as f:
            # Save transmitters mask
            if "transmitters" in f:
                del f["transmitters"] # Overwrite if exists
            f.create_dataset("transmitters", data=transmitters_mask, dtype="uint8", compression="gzip")
            
            # Save simulation target map
            if "radiomap" in f:
                del f["radiomap"]
            f.create_dataset("radiomap", data=radiomap, dtype="float32", compression="gzip")