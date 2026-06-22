import sionna.rt as rt
import numpy as np
import pandas as pd
from utils.geo_coords import SceneCoordinateConverter, deflected_azimuth, _get_terrain_z_batch
from config import DISPLAY_RADIUS, DEFAULT_TRANSIMITTER_DBM, DEFAULT_PITCH, DEFAULT_ROLL

def get_tx_name(id):
    return f'tx_{id}'

def get_rx_name(id):
    return f'rx_{id}'

def add_tx(scene, name, position, orientation):
    tx = rt.Transmitter(
        name=name, 
        position=position, 
        orientation=orientation, 
        power_dbm=DEFAULT_TRANSIMITTER_DBM,
        display_radius=DISPLAY_RADIUS)
    scene.add(tx)

def add_txs(
        scene: rt.Scene, 
        df_tx: pd.DataFrame, 
        converter: SceneCoordinateConverter,
        terrain_scene: rt.Scene | None = None):
    """
    Batch map geographic Tx coordinates into the 3D scene and instantiate
    Transmitter objects.
    """
    xs, ys, zs = converter.latlonh_to_xyz_batch(
        df_tx["Latitude"], df_tx["Longitude"], df_tx["height"], terrain_scene
    )

    azimuths = deflected_azimuth(df_tx["Azimut"])

    for x, y, z, azimuth, row in zip(
        xs, ys, zs, azimuths, df_tx.itertuples(index=False)
    ):
        name = get_tx_name(row.ID)

        # Debug output to verify transmitter height alignment
        print(f"TX {name} placed at x={x:.2f}, y={y:.2f}, z={z:.2f}, self_height={row.height:.2f}, azimuth={azimuth:.2f}")

        # Explicitly cast to Python native float as expected by the scene renderer
        add_tx(
            scene,
            name,
            (float(x), float(y), float(z)),
            (float(azimuth), DEFAULT_PITCH, DEFAULT_ROLL)
        )

def add_txs_no_overlap(
        scene: rt.Scene,
        df_tx: pd.DataFrame,
        converter: SceneCoordinateConverter,
        geometry_scene: rt.Scene):
    """
    Batch map geographic Tx coordinates into the 3D scene and instantiate
    Transmitter objects, with Z raised above any scene geometry to avoid overlap.
    """
    xs, ys, zs = converter.latlonh_to_xyz_batch_no_overlap(
        df_tx["Latitude"], df_tx["Longitude"], df_tx["height"], geometry_scene
    )

    azimuths = deflected_azimuth(df_tx["Azimut"])

    for x, y, z, azimuth, row in zip(
        xs, ys, zs, azimuths, df_tx.itertuples(index=False)
    ):
        name = get_tx_name(row.ID)

        # Debug output to verify transmitter height alignment
        print(f"TX {name} placed at x={x:.2f}, y={y:.2f}, z={z:.2f}, self_height={row.height:.2f}, azimuth={azimuth:.2f}")

        # Explicitly cast to Python native float as expected by the scene renderer
        add_tx(
            scene,
            name,
            (float(x), float(y), float(z)),
            (float(azimuth), DEFAULT_PITCH, DEFAULT_ROLL)
        )

def add_txs_from_grid(
        scene: rt.Scene,
        grid: np.ndarray,
        geometry_scene: rt.Scene,
) -> None:
    """
    Place transmitters on a grid where grid[row][col] == 1.

    The grid is an m×m numpy array of 0s and 1s. Each cell with value 1
    gets a transmitter placed at its center.

    Coordinate mapping:
        - col index -> X axis (East), col increases -> X increases
        - row index -> Y axis (North), row increases -> Y increases
        - Grid center maps to scene origin (0, 0)
        - For even m, row=0 starts at Y = +(m-1)/2 (e.g. +1.5 for m=4)

    Grid spacing is 1 meter between adjacent cell centers.

    Z coordinate is set to the top of the scene geometry at each (X, Y)
    location plus 3 meters (to avoid overlap with buildings).

    All transmitters face azimuth = 0° (North).

    Transmitters are named "tx_r{row}_c{col}".

    Parameters
    ----------
    scene : rt.Scene
        The Sionna scene to add transmitters to.
    grid : np.ndarray
        m×m array of uint8, where 1 means place a transmitter.
    geometry_scene : rt.Scene
        Scene used to query geometry top heights for Z placement.
    """
    m = grid.shape[0]

    # Find all (row, col) indices where the grid value is 1
    rows, cols = np.where(grid == 1)

    if len(rows) == 0:
        return  # Nothing to place

    # Compute the grid center in index space.
    # For m odd,  cx is an integer (e.g. cx=1 for m=3).
    # For m even, cx is a half-integer (e.g. cx=1.5 for m=4).
    cx = (m - 1) / 2.0

    # Convert row/col to scene X, Y coordinates.
    # Spacing between adjacent cells is 1 meter.
    xs = (cols - cx) * 1.0
    ys = (rows - cx) * 1.0

    # Query scene geometry top heights at each (X, Y) location
    geometry_z = _get_terrain_z_batch(xs, ys, geometry_scene)

    # Place transmitters 3 meters above geometry to avoid overlap
    zs = geometry_z + 3.0

    # Create a transmitter for each grid cell with value 1
    for row, col, x, y, z in zip(rows, cols, xs, ys, zs):
        name = f"tx_r{row}_c{col}"

        print(f"TX {name} placed at x={x:.2f}, y={y:.2f}, z={z:.2f}")

        add_tx(
            scene,
            name,
            (float(x), float(y), float(z)),
            (0.0, DEFAULT_PITCH, DEFAULT_ROLL),
        )

def add_rx(scene, name, position):
    rx = rt.Receiver(name=name, position=position, display_radius=DISPLAY_RADIUS)
    scene.add(rx)

def add_rxs(
        scene, 
        df_rx: pd.DataFrame, 
        converter: SceneCoordinateConverter,
        terrain_scene: rt.Scene | None = None):
    """
    Batch map geographic Rx coordinates into the 3D scene and instantiate
    Receiver objects.
    """
    xs, ys, zs = converter.latlonh_to_xyz_batch(
        df_rx["Latitude"], df_rx["Longitude"], df_rx["height"], terrain_scene
    )  # All receivers are at the same height.

    for x, y, z, row in zip(xs, ys, zs, df_rx.itertuples(index=False)):
        name = get_rx_name(row.sensor_id)

        # Debug output to verify sensor height alignment
        print(f"Sensor {name} placed at x={x:.2f}, y={y:.2f}, z={z:.2f}")

        # Explicitly cast to standard Python floats for the scene renderer
        add_rx(scene, name, (float(x), float(y), float(z)))

def set_frequence(scene, frequence, unit:str):
    """
    Utility function to configure the carrier frequency of the simulation scene.
    """
    if unit.lower() == 'mhz':
        scene.frequency = frequence * 1e6
    elif unit.lower() == 'ghz':
        scene.frequency = frequence * 1e9
    elif unit.lower() == 'hz':
        scene.frequency = frequence
    else:
        raise ValueError(f"Unsupported frequency unit: {unit}")