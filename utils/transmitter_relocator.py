import h5py
import numpy as np
from pathlib import Path
from typing import Union


class TransmitterRelocator:
    """
    Relocates transmitters to the nearest unoccupied building
    within a square search range.

    For each transmitter not already on a building, the class searches
    for the nearest unoccupied building within ±d pixels (clipped to
    image boundaries). If multiple buildings are at the same distance,
    the one with the smallest x (row), then smallest y (column) is chosen.

    Buildings already hosting a transmitter are marked as occupied
    and cannot be selected by other transmitters.

    Parameters
    ----------
    search_range : int
        Half-size of the square search window. A transmitter at (i, j)
        will search within rows [i-d, i+d] and cols [j-d, j+d].
    """

    def __init__(self, search_range: int):
        if search_range < 0:
            raise ValueError("search_range must be non-negative")
        self.search_range = search_range

    def process_file(self, file_path: Union[str, Path]) -> None:
        """
        Process a single HDF5 file in-place.

        Reads 'transmitters' and 'buildings' datasets, relocates
        transmitters according to the rules, and writes the updated
        'transmitters' dataset back. All other datasets are left untouched.

        Parameters
        ----------
        file_path : str or Path
            Path to the HDF5 file to process.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with h5py.File(file_path, "r+") as f:
            if "transmitters" not in f or "buildings" not in f:
                raise KeyError(
                    f"File {file_path} must contain 'transmitters' and 'buildings' datasets"
                )

            transmitters = f["transmitters"][:]  # shape (M, M), uint8
            buildings = f["buildings"][:]        # shape (M, M), uint8

            # Perform relocation
            new_transmitters = self._relocate(transmitters, buildings)

            # Write back to file
            f["transmitters"][:] = new_transmitters

    def process_directory(self, dir_path: Union[str, Path]) -> list[Path]:
        """
        Process all HDF5 files (*.h5, *.hdf5) in a directory (non-recursive).

        Parameters
        ----------
        dir_path : str or Path
            Path to the directory containing HDF5 files.

        Returns
        -------
        list of Path
            Paths of files that were successfully processed.
        """
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        h5_files = sorted(
            list(dir_path.glob("*.h5")) + list(dir_path.glob("*.hdf5"))
        )
        # Remove duplicates (in case a file matches both patterns)
        h5_files = list(dict.fromkeys(h5_files))

        processed = []
        for file_path in h5_files:
            try:
                self.process_file(file_path)
                processed.append(file_path)
                print(f"Processed: {file_path}")
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

        return processed

    def _relocate(
        self,
        transmitters: np.ndarray,
        buildings: np.ndarray
    ) -> np.ndarray:
        """
        Core relocation logic for a single map.

        Parameters
        ----------
        transmitters : np.ndarray
            2D array of shape (M, M), dtype uint8.
            1 indicates a transmitter, 0 indicates none.
        buildings : np.ndarray
            2D array of shape (M, M), dtype uint8.
            1 indicates a building, 0 indicates none.

        Returns
        -------
        np.ndarray
            Updated transmitters array after relocation.
        """
        M = transmitters.shape[0]
        d = self.search_range

        # Copy transmitters to modify
        result = transmitters.copy()

        # Track which buildings are already occupied by a transmitter
        # Initially: buildings that already have a transmitter on them
        occupied = (buildings == 1) & (transmitters == 1)

        # Get coordinates of all transmitter positions (row, col order)
        tx_rows, tx_cols = np.where(transmitters == 1)

        for tx_r, tx_c in zip(tx_rows, tx_cols):
            # If transmitter is already on a building, it stays
            # (occupied was already marked during initialization)
            if buildings[tx_r, tx_c] == 1:
                continue

            # Search for the nearest unoccupied building within range
            best_coords = self._find_nearest_building(
                tx_r, tx_c, buildings, occupied, M, d
            )

            if best_coords is not None:
                best_r, best_c = best_coords
                # Move transmitter: clear old position, set new position
                result[tx_r, tx_c] = 0
                result[best_r, best_c] = 1
                # Mark the new building as occupied
                occupied[best_r, best_c] = True
            # else: no building found within range, transmitter stays in place

        return result

    def _find_nearest_building(
        self,
        tx_r: int,
        tx_c: int,
        buildings: np.ndarray,
        occupied: np.ndarray,
        M: int,
        d: int
    ) -> tuple[int, int] | None:
        """
        Find the nearest unoccupied building within ±d of (tx_r, tx_c).

        Tie-breaking: smallest row (x), then smallest column (y).

        Parameters
        ----------
        tx_r, tx_c : int
            Transmitter coordinates (row, column).
        buildings : np.ndarray
            Building map.
        occupied : np.ndarray
            Mask of buildings already occupied by a transmitter.
        M : int
            Size of the square map (M x M).
        d : int
            Search range half-width.

        Returns
        -------
        (int, int) or None
            Coordinates of the nearest building, or None if not found.
        """
        # Compute search boundaries, clipped to valid range
        r_min = max(0, tx_r - d)
        r_max = min(M - 1, tx_r + d)
        c_min = max(0, tx_c - d)
        c_max = min(M - 1, tx_c + d)

        best_dist_sq = float("inf")
        best_r = None
        best_c = None

        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                # Skip if no building here or building is already occupied
                if buildings[r, c] == 0 or occupied[r, c]:
                    continue

                # Compute squared Euclidean distance
                dr = r - tx_r
                dc = c - tx_c
                dist_sq = dr * dr + dc * dc

                if dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    best_r = r
                    best_c = c
                elif dist_sq == best_dist_sq:
                    # Tie-breaking: smaller row first, then smaller column
                    if r < best_r or (r == best_r and c < best_c):
                        best_r = r
                        best_c = c

        if best_r is not None:
            return best_r, best_c
        return None