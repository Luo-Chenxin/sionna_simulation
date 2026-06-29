import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from pathlib import Path
from config import LocalCRS
from utils.map_splitter import BlockMeta
import mapbox_earcut as earcut

def extract_polygon_data(polygon, center_x, center_y, default_height):
    """
    Convert a 3D polygon to mesh vertices and faces including walls.
    Generates both the roof (top surface) and vertical walls (sides).
    
    Parameters
    ----------
    polygon : Polygon
        3D polygon geometry, possibly with holes
    center_x : float
        X coordinate of center point for translation
    center_y : float
        Y coordinate of center point for translation
        
    Returns
    -------
    tuple
        (vertices, faces) where vertices is list of [x, y, z] and faces is list of [i, j, k]
    """
    if polygon.is_empty:
        return [], []
    
    # Validate polygon
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
        if not polygon.is_valid:
            return [], []
    
    all_vertices = []
    all_faces = []
    
    # ============================================================
    # PART 1: Generate roof (top surface with triangulation)
    # ============================================================
    
    # Get the building height from first vertex
    exterior_coords = list(polygon.exterior.coords)
    first_coord = exterior_coords[0]
    height = first_coord[2] if len(first_coord) > 2 else default_height
    
    # Collect vertices for triangulation (2D for earcut)
    vertices_2d = []
    roof_vertex_map = {}  # Map from 2D coord index to 3D vertex index
    
    # Process exterior ring
    exterior_coords = exterior_coords[:-1]  # Remove closing point
    
    if len(exterior_coords) < 3:
        return [], []
    
    for x, y, *rest in exterior_coords:
        local_x = x - center_x
        local_y = y - center_y
        coord_2d = (round(local_x, 6), round(local_y, 6))
        
        if coord_2d not in roof_vertex_map:
            # Add 3D vertex at roof height
            vertex_idx = len(all_vertices)
            all_vertices.append([local_x, local_y, height])
            roof_vertex_map[coord_2d] = vertex_idx
        
        vertices_2d.append([local_x, local_y])
    
    # Track ring end indices for earcut
    ring_end_indices = [len(exterior_coords)]
    total_vertices = len(exterior_coords)
    
    # Process holes
    for interior in polygon.interiors:
        interior_coords = list(interior.coords)[:-1]
        
        if len(interior_coords) < 3:
            continue
        
        for x, y, *rest in interior_coords:
            local_x = x - center_x
            local_y = y - center_y
            coord_2d = (round(local_x, 6), round(local_y, 6))
            
            if coord_2d not in roof_vertex_map:
                vertex_idx = len(all_vertices)
                all_vertices.append([local_x, local_y, height])
                roof_vertex_map[coord_2d] = vertex_idx
            
            vertices_2d.append([local_x, local_y])
        
        total_vertices += len(interior_coords)
        ring_end_indices.append(total_vertices)
    
    # Triangulate roof
    vertices_array = np.array(vertices_2d, dtype=np.float32)
    ring_indices_array = np.array(ring_end_indices, dtype=np.uint32)
    
    try:
        triangle_indices = earcut.triangulate_float32(vertices_array, ring_indices_array)
    except Exception as e:
        print(f"  Roof earcut error: {e}")
        return [], []
    
    # Add roof faces
    for i in range(0, len(triangle_indices), 3):
        v1 = int(triangle_indices[i])
        v2 = int(triangle_indices[i + 1])
        v3 = int(triangle_indices[i + 2])
        
        if v1 != v2 and v2 != v3 and v1 != v3:
            all_faces.append([v1, v2, v3])
    
    # ============================================================
    # PART 2: Generate walls (vertical faces)
    # ============================================================
    
    # Walls are generated from exterior ring only (holes don't get walls)
    # For each edge of the exterior ring, create two triangles forming a quad
    
    num_exterior = len(exterior_coords)
    
    for i in range(num_exterior):
        # Get current and next vertex index (wrapping around)
        curr_2d_idx = i
        next_2d_idx = (i + 1) % num_exterior
        
        # Get 2D coordinates
        curr_2d = (round(vertices_2d[curr_2d_idx][0], 6), 
                   round(vertices_2d[curr_2d_idx][1], 6))
        next_2d = (round(vertices_2d[next_2d_idx][0], 6), 
                   round(vertices_2d[next_2d_idx][1], 6))
        
        # Get roof vertex indices
        curr_roof_idx = roof_vertex_map[curr_2d]
        next_roof_idx = roof_vertex_map[next_2d]
        
        curr_ground_x, curr_ground_y = vertices_2d[curr_2d_idx]
        next_ground_x, next_ground_y = vertices_2d[next_2d_idx]
        
        # Create ground level vertices (Z = 0)
        # Simple approach: always add ground vertices
        curr_ground_idx = len(all_vertices)
        all_vertices.append([curr_ground_x, curr_ground_y, 0.0])
        
        next_ground_idx = len(all_vertices)
        all_vertices.append([next_ground_x, next_ground_y, 0.0])
        
        # Create two triangles for the wall quad
        # Triangle 1: ground_curr -> roof_curr -> roof_next
        all_faces.append([curr_ground_idx, curr_roof_idx, next_roof_idx])
        # Triangle 2: ground_curr -> roof_next -> ground_next
        all_faces.append([curr_ground_idx, next_roof_idx, next_ground_idx])
    
    return all_vertices, all_faces

class OSMToPLY:
    """
    Convert osm geometries from a GeoDataFrame to 3D polygons and export as PLY file.
    
    This class assigns height values to polygon vertices, triangulates polygons,
    and exports to PLY format with LocalCRS.FRANCE_LAMBERT93 coordination system.
    """
    
    def __init__(self, gdf, ply_path, default_height, block_meta, handle_missing_height):
        """Initialize and process all polygons"""
        self.original_gdf = gdf.copy()
        self.ply_path = ply_path
        self.default_height = default_height
        
        self.gdf = gdf.to_crs(LocalCRS.FRANCE_LAMBERT93.crs)
        
        self.center_x = (block_meta.x_start + block_meta.x_end) / 2.0
        self.center_y = (block_meta.y_start + block_meta.y_end) / 2.0
        print(f"Center point: ({self.center_x:.2f}, {self.center_y:.2f})")
        
        # Process with explicit handle_missing_height parameter
        self._process_polygons(handle_missing_height=handle_missing_height)
        self._collect_3d_polygons()
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
        
        Attempts to find height information from 'height' and 'building:levels' attributes.
        
        Parameters
        ----------
        row : pd.Series
            A row from the GeoDataFrame
            
        Returns
        -------
        float or None
            Extracted height value, or None if not found
        """
        
        # Check 'height' attribute first
        if 'height' in row.index and row['height'] is not None:
            height_value = row['height']
            
            try:
                value = float(height_value)
                # Check if it's a meaningful number (not NaN)
                if not np.isnan(value):
                    return value
            except (ValueError, TypeError):
                pass
        
        # Check 'building:levels' attribute
        if 'building:levels' in row.index and row['building:levels'] is not None:
            levels_value = row['building:levels']
            
            try:
                value = float(levels_value)
                # Check if it's a meaningful number (not NaN)
                if not np.isnan(value):
                    # Convert number of levels to approximate height (3m per level)
                    return value * 3.0
            except (ValueError, TypeError):
                pass
        
        # No valid height information found
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
        
        if polygon.geom_type == 'Polygon':
            vertices, faces = extract_polygon_data(polygon, self.center_x, self.center_y, self.default_height)
        elif polygon.geom_type == 'MultiPolygon':
            for poly in polygon.geoms:
                poly_verts, poly_faces = extract_polygon_data(poly, self.center_x, self.center_y, self.default_height)
                
                # Adjust face indices for existing vertices
                offset = len(vertices)
                adjusted_faces = [[f[0] + offset, f[1] + offset, f[2] + offset] 
                                for f in poly_faces]
                
                vertices.extend(poly_verts)
                faces.extend(adjusted_faces)
        
        return np.array(vertices), np.array(faces)
    
    def save_to_ply(self):
        """
        Export the merged polygon mesh to binary PLY file format.
        
        Converts the merged 3D polygon to a triangulated mesh and writes
        it in binary PLY format for fast parsing. Guarantees output is 
        a valid PLY file even for empty geometries.
        
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
        
        # Convert to correct data types for binary writing
        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int32)
        
        # Calculate height statistics for verification
        heights = vertices[:, 2] if vertices.shape[1] >= 3 else np.zeros(len(vertices))
        unique_heights = len(set(np.round(heights, 2)))
        
        # Ensure output directory exists
        self.ply_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write binary PLY file
        with open(self.ply_path, 'wb') as f:
            # Write ASCII header
            header = f"""ply
format binary_little_endian 1.0
element vertex {len(vertices)}
property float x
property float y
property float z
element face {len(faces)}
property list uchar int vertex_indices
end_header
"""
            f.write(header.encode('ascii'))
            
            # Write vertex data as float32
            vertices.tofile(f)
            
            # Write face data: each face starts with vertex count (3)
            for face in faces:
                f.write(bytes([3]))
                face.tofile(f)
        
        print(f"Binary PLY file saved to: {self.ply_path}")
        print(f"Vertices: {len(vertices)}, Faces: {len(faces)}")
        print(f"Height range: {heights.min():.2f} to {heights.max():.2f} "
            f"(unique heights: {unique_heights})")
    
    def _write_empty_ply(self):
        """
        Write a minimal valid binary PLY file for empty geometry.
        
        Creates a PLY file with a single degenerate triangle to ensure
        the file is always valid PLY format.
        """
        # Ensure output directory exists
        self.ply_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.ply_path, 'wb') as f:
            # Write ASCII header
            header = """ply
format binary_little_endian 1.0
element vertex 1
property float x
property float y
property float z
element face 1
property list uchar int vertex_indices
end_header
"""
            f.write(header.encode('ascii'))
            
            # Write one vertex at origin
            vertex = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
            vertex.tofile(f)
            
            # Write one degenerate face
            f.write(bytes([3]))
            face = np.array([0, 0, 0], dtype=np.int32)
            face.tofile(f)
        
        print(f"Empty binary PLY file saved to: {self.ply_path}")

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
    Generate a flat terrain mesh as binary PLY file covering a given Lambert93 area.

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
    
    # Calculate center point for offset
    offset_x = (x_min + x_max) / 2.0
    offset_y = (y_min + y_max) / 2.0

    # Calculate grid dimensions
    x_count = int((x_max - x_min) / resolution)
    y_count = int((y_max - y_min) / resolution)

    # Generate vertex grid using meshgrid for efficiency
    x_coords = np.linspace(x_min, x_max, x_count, dtype=np.float32)
    y_coords = np.linspace(y_min, y_max, y_count, dtype=np.float32)
    xx, yy = np.meshgrid(x_coords, y_coords)
    
    # Build vertices array with offset
    total_vertices = x_count * y_count
    vertices = np.zeros((total_vertices, 3), dtype=np.float32)
    vertices[:, 0] = xx.ravel() - offset_x
    vertices[:, 1] = yy.ravel() - offset_y
    vertices[:, 2] = height

    # Generate triangle faces (two triangles per grid cell)
    total_faces = (x_count - 1) * (y_count - 1) * 2
    faces = np.zeros((total_faces, 3), dtype=np.int32)
    
    face_idx = 0
    for j in range(y_count - 1):
        row_start = j * x_count
        next_row_start = (j + 1) * x_count
        for i in range(x_count - 1):
            v00 = row_start + i
            v10 = v00 + 1
            v01 = next_row_start + i
            v11 = v01 + 1

            faces[face_idx] = [v00, v10, v11]
            faces[face_idx + 1] = [v00, v11, v01]
            face_idx += 2

    # Write binary PLY file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        # Write ASCII header
        header = f"""ply
format binary_little_endian 1.0
comment Flat terrain in Lambert93
comment Bounds: X=[{x_min:.1f}, {x_max:.1f}], Y=[{y_min:.1f}, {y_max:.1f}]
comment Resolution: {resolution}m, Height: {height}m
element vertex {len(vertices)}
property float x
property float y
property float z
element face {len(faces)}
property list uchar int vertex_indices
end_header
"""
        f.write(header.encode('ascii'))
        
        # Write all vertices at once
        vertices.tofile(f)
        
        # Write faces with vertex count prefix
        for face in faces:
            f.write(bytes([3]))
            face.tofile(f)

    print(f"Binary PLY saved to: {output_path}")
    print(f"  Vertices: {len(vertices)}, Faces: {len(faces)}")
    print(f"  Area: {(x_max - x_min)}m x {(y_max - y_min)}m")

    return output_path