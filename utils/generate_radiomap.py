import numpy as np
import pandas as pd
from pathlib import Path
import sionna.rt as rt
import mitsuba as mi

from utils.scene_utils import add_txs
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

    def __init__(self, converter: SceneCoordinateConverter):
        """
        Parameters
        ----------
        converter : SceneCoordinateConverter
            Converter that maps geographic coordinates to the scene's local
            coordinate system. The origin of this converter should be set to the
            center of the block's core area so that (0,0,0) in Sionna coincides
            with that center.
        """
        self.converter = converter

    def generate(
        self,
        xml_path: Path,
        csv_path: Path,
        block_meta: BlockMeta,
        tx_array: rt.PlanarArray,
        frequency: float,
        resolution_m: float
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
        block_meta : BlockMeta
            Block metadata containing both the extended bounds (with overlap) and
            the core bounds (without overlap). Only transmitters inside the core
            area are used.
        tx_array : rt.PlanarArray
            Antenna array to be used for all transmitters.
        frequency : float
            Carrier frequency in Hz.
        resolution_m : float
            Cell size in meters.

        Returns
        -------
        ndarray or None
            A 2D float array of summed linear RSS [W] with shape
            ``(num_rows, num_cols)`` where the row/column count is derived from
            ``block_meta.block_size_m / resolution_m``. Returns ``None`` if no
            transmitter falls inside the core area.
        """
        # Filter transmitters inside the block core
        df_tx_core = self._filter_tx_in_core(block_meta, csv_path)
        if df_tx_core is None:
            return None

        # Load and configure the Sionna scene
        scene = self._load_and_setup_scene(xml_path, frequency, tx_array)

        # Modify radio materials
        self._modify_materials(scene)

        # Add transmitters to the scene
        add_txs(scene, df_tx_core, self.converter, terrain_scene=None)

        # Compute radio map and sum RSS
        rss_map = self._compute_radiomap(scene, block_meta, resolution_m)

        return rss_map

    def _filter_tx_in_core(
        self, block_meta: BlockMeta, csv_path: Path
    ) -> pd.DataFrame | None:
        """
        Load CSV and keep only transmitters inside the block's core area.

        Parameters
        ----------
        block_meta : BlockMeta
            Block metadata with core latitude/longitude bounds.
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
            (tx_lon >= block_meta.lon_min_core) &
            (tx_lon <= block_meta.lon_max_core) &
            (tx_lat >= block_meta.lat_min_core) &
            (tx_lat <= block_meta.lat_max_core)
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
        scene.objects["water"].radio_material = create_sionna_material(
            "freshwater", scene.frequency
        )
        scene.objects["roads"].radio_material = create_sionna_material(
            "asphalt_concrete", scene.frequency
        )

        # Clean up materials that were loaded with the scene but are not needed
        scene.remove("wet_ground")
        scene.remove("chipboard")

    def _compute_radiomap(
        self,
        scene: rt.Scene,
        block_meta: BlockMeta,
        resolution_m: float
    ) -> np.ndarray:
        """
        Compute a radio map centered at (0,0,0) and sum the RSS of all TXs.

        Parameters
        ----------
        scene : rt.Scene
            The Sionna scene with transmitters already added.
        block_meta : BlockMeta
            Block metadata used to determine the map size.
        resolution_m : float
            Cell size in meters.

        Returns
        -------
        ndarray
            A 2D array of shape (num_rows, num_cols) with summed linear RSS [W].
        """
        solver = rt.RadioMapSolver()

        # Map extent in meters (square region)
        map_size_m = float(block_meta.block_size_m)

        rm = solver(
            scene,
            max_depth=5,
            samples_per_tx=int(1e7),
            cell_size=mi.Point2f(resolution_m, resolution_m),
            center=mi.Point3f(0.0, 0.0, 0.0),
            size=mi.Point2f(map_size_m, map_size_m),
            orientation=mi.Point3f(0, 0, 0)
        )

        # rm.rss has shape [num_tx, num_cols, num_rows]
        rss_tensor = rm.rss.numpy()    # shape: (N_tx, n_cols, n_rows)

        # Sum contributions from all transmitters
        rss_sum = rss_tensor.sum(axis=0)   # shape: (n_cols, n_rows)

        # Transpose to obtain [row, column] layout
        rss_map = rss_sum.transpose()      # shape: (n_rows, n_cols)

        # Expected number of rows / columns
        expected_cells = int(round(map_size_m / resolution_m))
        if rss_map.shape != (expected_cells, expected_cells):
            raise RuntimeError(
                f"Expected RSS map shape ({expected_cells}, {expected_cells}), "
                f"got {rss_map.shape}"
            )

        return rss_map.astype(np.float32)