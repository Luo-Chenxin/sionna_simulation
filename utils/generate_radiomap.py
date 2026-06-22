import numpy as np
import pandas as pd
from pathlib import Path
import sionna.rt as rt
import mitsuba as mi

from utils.scene_utils import add_txs_no_overlap, add_txs_from_grid
from utils.geo_coords import SceneCoordinateConverter
from utils.map_splitter import BlockMeta
from utils.material_factory import create_sionna_material


class RadioMapGenerator:
    """
    Generate summed RSS radio maps for Sionna scenes over rectangular blocks.

    The class handles:
      - Filtering transmitters that lie inside the core area of a block.
      - Loading and configuring a Sionna scene (materials, antenna array).
      - Instantiating transmitters via geographic coordinate conversion.
      - Computing the radio map with a ray‑tracing solver and summing the
        linear RSS contributions from all transmitters.
    """

    def __init__(self, converter: SceneCoordinateConverter, block_meta: BlockMeta, resolution_m: float):
        """
        Parameters
        ----------
        converter : SceneCoordinateConverter
            Converter that maps geographic coordinates to the scene's local
            coordinate system. The origin of this converter should be set to the
            center of the block's core area so that (0,0,0) in Sionna coincides
            with that center.
        block_meta : BlockMeta
        resolution_m : float
            Cell size in meters.
        """
        self.converter = converter
        self.block_meta = block_meta
        self.resolution_m = resolution_m
        self._setup_matrix_bounds()

    def generate(
        self,
        xml_path: Path,
        csv_path: Path,
        tx_array: rt.PlanarArray,
        frequency: float,
    ) -> np.ndarray | None:
        """
        Generate a summed RSS map for the given block.

        Parameters
        ----------
        xml_path : Path
            Sionna scene XML file. The corresponding ``mesh/`` directory must be
            located in the same folder as the XML file.
        csv_path : Path
            CSV file with transmitter columns: ``Latitude``, ``Longitude``,
            ``height``, ``Azimut``.
        tx_array : rt.PlanarArray
            Antenna array to be used for all transmitters.
        frequency : float
            Carrier frequency in Hz.

        Returns
        -------
        ndarray or None
            A 2D float array of summed linear RSS [W] with shape
            ``(num_rows, num_cols)`` where the row/column count is derived from
            ``block_meta.block_size_m / resolution_m``. Returns ``None`` if no
            transmitter falls inside the core area.
        """
        # Filter transmitters inside the block core
        df_tx_core = self._filter_tx_in_block(csv_path)
        if df_tx_core is None:
            return None

        # Load and configure the Sionna scene
        scene = self._load_and_setup_scene(xml_path, frequency, tx_array)

        # Modify radio materials
        self._modify_materials(scene)

        # Add transmitters to the scene
        geometry_scene = rt.load_scene(xml_path)
        add_txs_no_overlap(scene, df_tx_core, self.converter, geometry_scene)

        # Compute radio map and sum RSS
        rss_map = self._compute_radiomap(scene)

        return rss_map
    
    def generate_from_grid(
        self,
        xml_path: Path,
        tx_grid: np.ndarray,
        tx_array: rt.PlanarArray,
        frequency: float,
    ) -> np.ndarray | None:
        """
        Generate a summed RSS map using transmitter positions from a binary grid.

        Unlike ``generate()`` which reads transmitter locations from a CSV file,
        this method places transmitters wherever ``tx_grid[row][col] == 1``.
        The grid should already be corrected (e.g. relocated to nearest buildings)
        before calling this method.

        Parameters
        ----------
        xml_path : Path
            Sionna scene XML file. The corresponding ``mesh/`` directory must be
            located in the same folder as the XML file.
        tx_grid : np.ndarray
            2D uint8 array of shape (M, M) where 1 means place a transmitter.
            Typically read from an HDF5 file's ``transmitters`` dataset.
        tx_array : rt.PlanarArray
            Antenna array to be used for all transmitters.
        frequency : float
            Carrier frequency in Hz.

        Returns
        -------
        ndarray or None
            A 2D float array of summed linear RSS [W] with shape
            ``(num_rows, num_cols)``. Returns ``None`` if ``tx_grid`` has no
            active transmitters (all zeros).
        """
        # Skip if grid has no transmitters
        if not np.any(tx_grid):
            return None

        # Load and configure the Sionna scene
        scene = self._load_and_setup_scene(xml_path, frequency, tx_array)

        # Modify radio materials
        self._modify_materials(scene)

        # Load a separate geometry scene for height queries.
        # Kept independent from ``scene`` so that future extensions can modify
        # ``scene`` (e.g. adding custom objects) without affecting height lookups.
        geometry_scene = rt.load_scene(xml_path)

        # Place transmitters from grid
        add_txs_from_grid(scene, tx_grid, geometry_scene)

        # Compute radio map and sum RSS
        rss_map = self._compute_radiomap(scene)

        return rss_map

    def _filter_tx_in_block(
        self, csv_path: Path
    ) -> pd.DataFrame | None:
        """
        Load CSV and keep only transmitters inside the block's core area.

        Parameters
        ----------
        csv_path : Path
            Path to the transmitter CSV file.

        Returns
        -------
        pd.DataFrame or None
            Filtered transmitters, or ``None`` if none are in core.
        """
        df_tx = pd.read_csv(csv_path)

        tx_lon = df_tx["Longitude"].to_numpy()
        tx_lat = df_tx["Latitude"].to_numpy()

        in_core = (
            (tx_lon >= self.block_meta.lon_min_core) &
            (tx_lon <= self.block_meta.lon_max_core) &
            (tx_lat >= self.block_meta.lat_min_core) &
            (tx_lat <= self.block_meta.lat_max_core)
        )
        df_core = df_tx.loc[in_core].copy()

        if df_core.empty:
            return None
        return df_core

    def _load_and_setup_scene(
        self,
        xml_path: Path,
        frequency: float,
        tx_array: rt.PlanarArray
    ) -> rt.Scene:
        """
        Load the Sionna scene and configure the transmitter array and frequency.

        Parameters
        ----------
        xml_path : Path
            Sionna XML scene file.
        frequency : float
            Carrier frequency in Hz.
        tx_array : rt.PlanarArray
            Antenna array for all transmitters.

        Returns
        -------
        rt.Scene
            Configured Sionna scene.
        """
        scene = rt.load_scene(str(xml_path))
        scene.frequency = frequency
        scene.tx_array = tx_array
        return scene

    def _modify_materials(self, scene: rt.Scene) -> None:
        """
        Update radio materials for water and roads, and remove obsolete materials.

        Parameters
        ----------
        scene : rt.Scene
            The Sionna scene to modify.
        """
        if 'water' in scene.objects:
            scene.objects['water'].radio_material = create_sionna_material(
                'freshwater', scene.frequency
            )
            scene.remove('wet_ground')

        if 'roads' in scene.objects:
            scene.objects['roads'].radio_material = create_sionna_material(
                'asphalt_concrete', scene.frequency
            )
            scene.remove('chipboard')
    
    def _setup_matrix_bounds(self):
        """
        Calculate matrix boundaries from BlockMeta in projected coordinates.
        BlockMeta already has x_start, x_end, y_start, y_end in meters.
        """
        # Use BlockMeta's meter coordinates directly
        self.x_min = self.block_meta.x_start
        self.x_max = self.block_meta.x_end
        self.y_min = self.block_meta.y_start
        self.y_max = self.block_meta.y_end

        self.overlap_m =self.block_meta.overlap_m
        
        # Calculate matrix dimensions
        total_width = self.x_max - self.x_min
        total_height = self.y_max - self.y_min
        
        self.n_cols = int(np.ceil(total_width / self.resolution_m))
        self.n_rows = int(np.ceil(total_height / self.resolution_m))
        
        # Pre-calculate crop indices for overlap removal
        if self.overlap_m > 0:
            cells_to_crop = int(np.ceil(self.overlap_m / self.resolution_m))
            # Check if cropping is possible
            if cells_to_crop == 0:
                self.crop_slice = np.s_[:, :, :]
            elif cells_to_crop * 2 < self.n_rows and cells_to_crop * 2 < self.n_cols:
                # [tx, cols, rows]
                self.crop_slice = np.s_[:, cells_to_crop:-cells_to_crop, cells_to_crop:-cells_to_crop]
            else:
                # Overlap is too large relative to matrix size
                self.crop_slice = np.s_[:, :, :]
        else:
            self.crop_slice = np.s_[:, :, :]

    def _compute_radiomap(
        self,
        scene: rt.Scene,
    ) -> np.ndarray:
        """
        Compute a radio map centered at (0,0,0) and sum the RSS of all TXs.

        Parameters
        ----------
        scene : rt.Scene
            The Sionna scene with transmitters already added.

        Returns
        -------
        ndarray
            A 2D array of shape (num_rows, num_cols) with summed linear RSS [W].
        """
        solver = rt.RadioMapSolver()

        # Map extent in meters (square region)
        x_size_m = self.block_meta.x_end - self.block_meta.x_start
        y_size_m = self.block_meta.y_end - self.block_meta.y_start

        rm = solver(
            scene,
            max_depth=5,
            samples_per_tx=int(1e7),
            cell_size=mi.Point2f(self.resolution_m, self.resolution_m),
            center=mi.Point3f(0.0, 0.0, 0.0),
            size=mi.Point2f(x_size_m, y_size_m),
            orientation=mi.Point3f(0, 0, 0)
        )

        # rm.rss has shape [num_tx, num_cols, num_rows]
        rss_tensor = rm.rss.numpy()    # shape: (N_tx, n_cols, n_rows)

        # Remove overlop
        rss_tensor = rss_tensor[self.crop_slice]    # shape: (N, n_cols_cropped, n_rows_cropped) 

        # Sum contributions from all transmitters
        rss_map = rss_tensor.sum(axis=0)   # shape: (n_cols_cropped, n_rows_cropped) 

        return rss_map.astype(np.float32)