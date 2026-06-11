import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from pathlib import Path
from config import LocalCRS


class OSMToPLY:
    """
    Convert osm geometries from a GeoDataFrame to 3D polygons and export as PLY file.
    
    This class assigns height values to polygon vertices, triangulates polygons,
    and exports to PLY format with LocalCRS.FRANCE_LAMBERT93 coordination system.
    """
    
    def __init__(self, gdf: gpd.GeoDataFrame, ply_path: Path, default_height: float):
        """
        Initialize the converter with input data and parameters.
        
        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            Input GeoDataFrame containing Polygon or MultiPolygon geometries
        ply_path : Path
            Output PLY file path
        default_height : float
            Default height value for polygons without height information
        """
        self.original_gdf = gdf.copy()
        self.ply_path = ply_path
        self.default_height = default_height
        
        # Convert to local CRS
        self.gdf = gdf.to_crs(LocalCRS.FRANCE_LAMBERT93.crs)
        
        # Process polygons: extract heights and convert to 3D
        if default_height == 0.0:
            self._process_polygons(handle_missing_height='drop')
        else:
            self._process_polygons(handle_missing_height='use_default')
        
        # Collect all 3D polygons
        self._collect_3d_polygons()
        
        # Build final MultiPolygon
        self._build_multi_polygon()
    
    def _process_polygons(self, handle_missing_height: str):
        """
        Process polygon geometries: extract heights and convert to 3D.
        
        Preserves original heights from attributes and handles cases where
        height information is missing.
        
        Parameters
        ----------
        handle_missing_height : str, optional
            Mode for handling polygons with missing height information.
            'use_default' : Use default_height for polygons without height
            'drop' : Remove polygons that don't have height information
        """
        # Filter to only polygon types
        mask = self.gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])
        polygon_gdf = self.gdf[mask].copy()
        
        if len(polygon_gdf) == 0:
            self.processed_geoms = []
            return
        
        processed_geoms = []
        
        for _, row in polygon_gdf.iterrows():
            geom = row.geometry
            
            if geom is None or geom.is_empty:
                continue
            
            # Try to get original height from attributes
            original_height = self._extract_height_from_row(row)
            
            if geom.has_z:
                # Preserve original 3D geometry
                processed_geoms.append(geom)
            elif original_height is not None:
                # Use original height from attributes
                polygon_3d = self._assign_height_to_polygon(geom, original_height)
                processed_geoms.append(polygon_3d)
            else:
                # No height information available
                if handle_missing_height == 'use_default':
                    polygon_3d = self._assign_height_to_polygon(geom, self.default_height)
                    processed_geoms.append(polygon_3d)
                elif handle_missing_height == 'drop':
                    # Skip this polygon
                    continue
                else:
                    raise ValueError(f"Invalid handle_missing_height mode: {handle_missing_height}. "
                                     f"Must be 'use_default' or 'drop'")
        
        self.processed_geoms = processed_geoms
        print(f"Processed {len(processed_geoms)} polygons with 3D coordinates")
    
    def _extract_height_from_row(self, row):
        """
        Extract height value from GeoDataFrame row attributes.
        
        Attempts to find height information from common attribute names
        (height, building:height, levels, etc.) and falls back to None.
        
        Parameters
        ----------
        row : pd.Series
            A row from the GeoDataFrame
            
        Returns
        -------
        float or None
            Extracted height value, or None if not found
        """
        # Common height-related column names in OSM data
        height_columns = ['height', 'building:height', 'levels', 'building:levels']
        
        for col in height_columns:
            if col in row.index and row[col] is not None:
                try:
                    value = float(row[col])
                    if col in ('levels', 'building:levels'):
                        # Convert number of levels to approximate height (3m per level)
                        return value * 3.0
                    return value
                except (ValueError, TypeError):
                    continue
        
        # No height information found
        return None
    
    def _assign_height_to_polygon(self, polygon, height):
        """
        Create a 3D polygon by assigning Z coordinate to all vertices.
        
        Parameters
        ----------
        polygon : Polygon or MultiPolygon
            2D polygon geometry to convert
        height : float
            Height value to assign as Z coordinate
            
        Returns
        -------
        Polygon or MultiPolygon
            3D polygon with Z coordinates
        """
        if polygon.geom_type == 'Polygon':
            # Process exterior ring
            exterior_coords = [(x, y, height) for x, y in polygon.exterior.coords]
            
            # Process interior rings (holes)
            interior_coords = []
            for interior in polygon.interiors:
                interior_coords.append([(x, y, height) for x, y in interior.coords])
            
            return Polygon(exterior_coords, interior_coords)
            
        elif polygon.geom_type == 'MultiPolygon':
            # Recursively process each polygon in the multipolygon
            polygons_3d = []
            for poly in polygon.geoms:
                polygons_3d.append(self._assign_height_to_polygon(poly, height))
            return MultiPolygon(polygons_3d)
        
        return polygon
    
    def _collect_3d_polygons(self):
        """
        Collect all individual 3D polygons while preserving their height information.
        
        Each polygon is stored separately to maintain its unique height attribute.
        Overlapping areas will contain multiple polygons at different heights.
        """
        self.individual_polygons = []
        
        for geom in self.processed_geoms:
            if geom.is_empty:
                continue
            
            # Extract individual polygons from MultiPolygon
            if geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    if not poly.is_empty:
                        height = self._get_polygon_height(poly)
                        self.individual_polygons.append({
                            'geometry': poly,
                            'height': height
                        })
            elif geom.geom_type == 'Polygon':
                height = self._get_polygon_height(geom)
                self.individual_polygons.append({
                    'geometry': geom,
                    'height': height
                })
        
        print(f"Collected {len(self.individual_polygons)} individual polygons with preserved heights")
    
    def _get_polygon_height(self, polygon):
        """
        Extract the Z coordinate from a polygon's vertices.
        
        Parameters
        ----------
        polygon : Polygon
            3D polygon geometry
            
        Returns
        -------
        float
            Height value (Z coordinate) of the polygon
        """
        if polygon.is_empty:
            return 0.0
        
        try:
            # Get Z coordinate from exterior ring
            coords = list(polygon.exterior.coords)
            if len(coords) > 0 and len(coords[0]) >= 3:
                return coords[0][2]
        except (IndexError, TypeError):
            pass
        
        return 0.0
    
    def _build_multi_polygon(self):
        """
        Build a MultiPolygon from all individual polygons.
        
        Keeps polygons as separate entities in a MultiPolygon structure.
        Overlapping areas are preserved with their respective heights.
        """
        if len(self.individual_polygons) == 0:
            self.merged_polygon = MultiPolygon()
            return
        
        # Collect all polygon geometries
        all_geoms = [item['geometry'] for item in self.individual_polygons]
        
        # Create a MultiPolygon preserving all individual geometries and their heights
        try:
            self.merged_polygon = MultiPolygon(all_geoms)
            print(f"Built MultiPolygon with {len(all_geoms)} components preserving individual heights")
        except Exception as e:
            print(f"Error building MultiPolygon: {e}")
            # Fallback: use first polygon
            if len(all_geoms) > 0:
                self.merged_polygon = all_geoms[0]
            else:
                self.merged_polygon = MultiPolygon()
    
    def _polygon_to_mesh(self, polygon):
        """
        Convert a 3D polygon to mesh vertices and faces for PLY export.
        
        Creates a fan triangulation of the polygon preserving 3D coordinates.
        
        Parameters
        ----------
        polygon : Polygon or MultiPolygon
            3D polygon geometry
            
        Returns
        -------
        tuple
            (vertices_array, faces_array) as numpy arrays
        """
        vertices = []
        faces = []
        
        if polygon.is_empty:
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
        
        def extract_polygon_data(poly):
            """Extract vertices and create faces for a single polygon."""
            verts = []
            face_list = []
            
            # Get exterior ring vertices
            exterior_coords = list(poly.exterior.coords)
            
            if len(exterior_coords) == 0:
                return verts, face_list
            
            # Handle both 2D and 3D coordinates
            if len(exterior_coords[0]) == 2:
                exterior_verts = [(x, y, 0.0) for x, y in exterior_coords]
            else:
                exterior_verts = [(x, y, z) for x, y, z in exterior_coords]
            
            start_idx = len(verts)
            # Exclude last point (same as first)
            verts.extend(exterior_verts[:-1])
            
            # Create fan triangulation from first vertex
            for i in range(1, len(exterior_verts) - 2):
                face_list.append([start_idx, start_idx + i, start_idx + i + 1])
            
            # Process interior rings (holes)
            for interior in poly.interiors:
                interior_coords = list(interior.coords)
                
                if len(interior_coords) == 0:
                    continue
                
                if len(interior_coords[0]) == 2:
                    interior_verts = [(x, y, 0.0) for x, y in interior_coords]
                else:
                    interior_verts = [(x, y, z) for x, y, z in interior_coords]
                
                hole_start_idx = len(verts)
                verts.extend(interior_verts[:-1])
                
                # Fan triangulation for holes (reversed for correct orientation)
                for i in range(1, len(interior_verts) - 2):
                    face_list.append([hole_start_idx, hole_start_idx + i + 1, hole_start_idx + i])
            
            return verts, face_list
        
        if polygon.geom_type == 'Polygon':
            vertices, faces = extract_polygon_data(polygon)
        elif polygon.geom_type == 'MultiPolygon':
            for poly in polygon.geoms:
                poly_verts, poly_faces = extract_polygon_data(poly)
                
                # Adjust face indices for existing vertices
                offset = len(vertices)
                adjusted_faces = [[f[0] + offset, f[1] + offset, f[2] + offset] 
                                for f in poly_faces]
                
                vertices.extend(poly_verts)
                faces.extend(adjusted_faces)
        
        return np.array(vertices), np.array(faces)
    
    def save_to_ply(self):
        """
        Export the merged polygon mesh to PLY file format.
        
        Converts the merged 3D polygon to a triangulated mesh and writes
        it in standard PLY format with vertex positions and face indices.
        Guarantees output is a valid PLY file even for empty geometries.
        
        Raises
        ------
        ValueError
            If merged polygon is None
        """
        if self.merged_polygon is None:
            raise ValueError("No merged polygon to save. Run processing first.")
        
        # Handle empty polygon case - create minimal valid PLY file
        if self.merged_polygon.is_empty:
            self._write_empty_ply()
            return
        
        # Convert polygon to mesh vertices and faces
        vertices, faces = self._polygon_to_mesh(self.merged_polygon)
        
        if len(vertices) == 0 or len(faces) == 0:
            self._write_empty_ply()
            return
        
        # Calculate height statistics for verification
        heights = vertices[:, 2] if vertices.shape[1] >= 3 else np.zeros(len(vertices))
        unique_heights = len(set(np.round(heights, 2)))
        
        # Ensure output directory exists
        self.ply_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write PLY file
        with open(self.ply_path, 'w') as f:
            # Header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            
            # Vertex data
            for vertex in vertices:
                f.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            
            # Face data
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        
        print(f"PLY file saved to: {self.ply_path}")
        print(f"Vertices: {len(vertices)}, Faces: {len(faces)}")
        print(f"Height range: {heights.min():.2f} to {heights.max():.2f} "
              f"(unique heights: {unique_heights})")
    
    def _write_empty_ply(self):
        """
        Write a minimal valid PLY file for empty geometry.
        
        Creates a PLY file with a single degenerate triangle to ensure
        the file is always valid PLY format.
        """
        # Ensure output directory exists
        self.ply_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.ply_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write("element vertex 1\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("element face 1\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            f.write("0.000000 0.000000 0.000000\n")
            f.write("3 0 0 0\n")
        
        print(f"Empty PLY file saved to: {self.ply_path}")

def generate_flat_terrain_ply(
    output_path: Path,
    x_min: int, 
    x_max: int, 
    y_min: int, 
    y_max: int, 
    resolution: float,
    height: float,
) -> Path:
    """
    Generate a flat terrain mesh as PLY file covering a given Lambert93 area.

    The terrain is a regular grid of triangles at a constant height.

    Parameters
    ----------
    output_path : Path
        Path where the PLY file will be saved
    x_min : int
        x_min in Lambert93 CRS
    x_max : int
        x_max in Lambert93 CRS
    y_min : int
        y_min in Lambert93 CRS
    y_max : int
        y_max in Lambert93 CRS
    resolution : float
        Grid cell size in meters
    height : float
        Constant Z height for all terrain vertices

    Returns
    -------
    Path
        Path to the generated PLY file

    Raises
    ------
    ValueError
        If resolution is not positive
    """
    if resolution <= 0:
        raise ValueError(f"Resolution must be positive, got {resolution}")

    # Calculate grid dimensions
    x_count = int((x_max - x_min) / resolution) + 1
    y_count = int((y_max - y_min) / resolution) + 1

    print(f"Generating terrain grid: {x_count} x {y_count} = {x_count * y_count} vertices")

    # Generate vertex grid
    vertices = []
    for j in range(y_count):
        y = y_min + j * resolution
        for i in range(x_count):
            x = x_min + i * resolution
            vertices.append((x, y, height))

    vertices = np.array(vertices)

    # Generate triangle faces (two triangles per grid cell)
    faces = []
    for j in range(y_count - 1):
        for i in range(x_count - 1):
            v00 = j * x_count + i
            v10 = j * x_count + i + 1
            v01 = (j + 1) * x_count + i
            v11 = (j + 1) * x_count + i + 1

            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    faces = np.array(faces)

    # Write PLY file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"comment Flat terrain in Lambert93\n")
        f.write(f"comment Bounds: X=[{x_min:.1f}, {x_max:.1f}], Y=[{y_min:.1f}, {y_max:.1f}]\n")
        f.write(f"comment Resolution: {resolution}m, Height: {height}m\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    print(f"PLY saved to: {output_path}")
    print(f"  Vertices: {len(vertices)}, Faces: {len(faces)}")
    print(f"  Area: {(x_max - x_min)}m x {(y_max - y_min)}m")

    return output_path