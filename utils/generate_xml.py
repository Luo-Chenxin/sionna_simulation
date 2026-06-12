import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from typing import Dict

class SionnaXMLGenerator:
    """
    Generate Sionna 2.0.1 compatible XML scene files from PLY mesh directories.
    
    This class handles the creation of Mitsuba 3 XML scene files with proper
    material assignments and mesh references for Sionna 2.0.1 ray tracing.
    """
    
    # Fixed mesh filenames and their corresponding material assignments
    MESH_MATERIALS: Dict[str, Dict[str, str]] = {
        "terrain.ply": {
            "material_id": "mat-itu_medium_dry_ground",
            "material_type": "twosided",
            "color": "0.750000 0.650000 0.450000",  # Sandy brown for terrain
            "face_normals": "true"
        },
        "buildings.ply": {
            "material_id": "mat-itu_concrete",
            "material_type": "diffuse",
            "color": "0.700000 0.700000 0.750000",  # Light gray for concrete
            "face_normals": "true"
        },
        "railways.ply": {
            "material_id": "mat-itu_metal",
            "material_type": "diffuse",
            "color": "0.400000 0.350000 0.300000",  # Dark brown/gray for metal/rails
            "face_normals": "true"
        },
        "roads.ply": {
            "material_id": "mat-itu_chipboard",
            "material_type": "diffuse",
            "color": "0.250000 0.250000 0.280000",  # Dark gray for asphalt
            "face_normals": "true"
        },
        "forest.ply": {
            "material_id": "mat-itu_wood",
            "material_type": "diffuse",
            "color": "0.150000 0.550000 0.150000",  # Forest green for trees
            "face_normals": "true"
        },
        "water.ply": {
            "material_id": "mat-itu_wet_ground",
            "material_type": "diffuse",
            "color": "0.100000 0.300000 0.700000",  # Blue for water
            "face_normals": "true"
        }
    }
    
    def __init__(
        self, 
        mesh_dir: Path, 
        output_path: Path,
    ):
        """
        Initialize the XML generator with mesh directory and output settings.
        
        Parameters
        ----------
        mesh_dir : Path
            Directory containing PLY mesh files
        output_path : Path
            Output XML path
        
        Raises
        ------
        ValueError
            If mesh_dir does not exist or is not a directory
        """
        self.mesh_dir = Path(mesh_dir)
        if not self.mesh_dir.exists():
            raise ValueError(f"Mesh directory does not exist: {self.mesh_dir}")
        if not self.mesh_dir.is_dir():
            raise ValueError(f"Path is not a directory: {self.mesh_dir}")
        
        self.output_path = output_path
        
    def _create_integrator(self, parent: ET.Element) -> None:
        """
        Create integrator element for path tracing configuration.
        
        Parameters
        ----------
        parent : ET.Element
            Parent XML element to append to
        """
        integrator = ET.SubElement(parent, "integrator", {
            "type": "path",
            "id": "elm__0",
            "name": "elm__0"
        })
        ET.SubElement(integrator, "integer", {
            "name": "max_depth",
            "value": "12"
        })
    
    def _create_materials(self, parent: ET.Element) -> None:
        """
        Create material definitions for all mesh types.
        
        Parameters
        ----------
        parent : ET.Element
            Parent XML element to append to
        """
        for _, material_info in self.MESH_MATERIALS.items():
            bsdf = ET.SubElement(parent, "bsdf", {
                "type": material_info["material_type"],
                "id": material_info["material_id"],
                "name": material_info["material_id"]
            })
            ET.SubElement(bsdf, "rgb", {
                "value": material_info["color"],
                "name": "reflectance"
            })
    
    def _create_emitter(self, parent: ET.Element) -> None:
        """
        Create world emitter for constant environment lighting.
        
        Parameters
        ----------
        parent : ET.Element
            Parent XML element to append to
        """
        emitter = ET.SubElement(parent, "emitter", {
            "type": "constant",
            "id": "World",
            "name": "World"
        })
        ET.SubElement(emitter, "rgb", {
            "value": "1.000000 1.000000 1.000000",
            "name": "radiance"
        })
    
    def _create_shapes(self, parent: ET.Element) -> None:
        """
        Create shape elements referencing PLY mesh files.
        
        Parameters
        ----------
        parent : ET.Element
            Parent XML element to append to
        
        Raises
        ------
        FileNotFoundError
            If required mesh file is missing
        """
        for mesh_filename, material_info in self.MESH_MATERIALS.items():
            mesh_path = self.mesh_dir / mesh_filename
            
            if not mesh_path.exists():
                print(f"Mesh file not found: {mesh_path}. Skipping shape.")
                continue
            
            # Create shape element
            mesh_name = mesh_filename.replace(".ply", "")
            shape = ET.SubElement(parent, "shape", {
                "type": "ply",
                "id": f"mesh-{mesh_name}",
                "name": f"mesh-{mesh_name}"
            })
            
            # Add filename reference
            ET.SubElement(shape, "string", {
                "name": "filename",
                "value": f"{mesh_path}"
            })
            
            # Add face normals
            ET.SubElement(shape, "boolean", {
                "name": "face_normals",
                "value": "true"
            })
            
            # Add material reference
            ET.SubElement(shape, "ref", {
                "id": material_info["material_id"],
                "name": "bsdf"
            })
    
    def _build_xml_tree(self) -> ET.Element:
        """
        Build the complete XML element tree for the scene.
        
        Returns
        -------
        ET.Element
            Root element of the XML tree
        
        Notes
        -----
        This method constructs the complete scene XML with all required
        sections: integrator, materials, emitter, and shapes.
        """
        # Create root element
        root = ET.Element("scene", {"version": "2.1.0"})
        
        # Add comments for structure
        root.append(ET.Comment(" Defaults, these can be set via the command line: -Darg=value "))
        root.append(ET.Comment(" Camera and Rendering Parameters "))
        self._create_integrator(root)
        
        root.append(ET.Comment(" Materials "))
        self._create_materials(root)
        
        root.append(ET.Comment(" Emitters "))
        self._create_emitter(root)
        
        root.append(ET.Comment(" Shapes "))
        self._create_shapes(root)
        
        root.append(ET.Comment(" Volumes "))
        
        return root
    
    def _prettify_xml(self, elem: ET.Element) -> str:
        """
        Format XML element tree as a pretty-printed string.
        
        Parameters
        ----------
        elem : ET.Element
            Root XML element to format
        
        Returns
        -------
        str
            Formatted XML string with proper indentation
        """
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="\t")
    
    def generate(self, validate_meshes: bool = True) -> Path:
        """
        Generate the Sionna 2.0.1 XML scene file.
        
        Parameters
        ----------
        validate_meshes : bool, optional
            Whether to check if all mesh files exist, by default True
        
        Returns
        -------
        Path
            Path to the generated XML file
        
        Raises
        ------
        FileNotFoundError
            If validate_meshes is True and required meshes are missing
        IOError
            If unable to write the XML file
        """
        # Validate mesh files if requested
        if validate_meshes:
            missing_meshes = []
            for mesh_filename in self.MESH_MATERIALS.keys():
                if not (self.mesh_dir / mesh_filename).exists():
                    missing_meshes.append(mesh_filename)
            
            if missing_meshes:
                error_msg = f"Missing mesh files: {', '.join(missing_meshes)}"
                print(error_msg)
                raise FileNotFoundError(error_msg)
        
        # Build XML tree
        root = self._build_xml_tree()
        
        # Convert to pretty XML string
        xml_string = self._prettify_xml(root)
        
        # Write to file
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, 'w', encoding='utf-8') as f:
                f.write(xml_string)
            print(f"Successfully generated XML file: {self.output_path}")
        except IOError as e:
            print(f"Failed to write XML file: {e}")
            raise
        
        return self.output_path


def generate_xml(
    mesh_dir: Path,
    output_path: Path,
    validate_meshes: bool = True
) -> Path:
    """
    Convenience function to generate Sionna 2.0.1 XML from a mesh directory.
    
    This function creates an XML scene file compatible with Sionna 2.0.1
    that references PLY mesh files in the specified directory with
    appropriate material assignments.
    
    Parameters
    ----------
    mesh_dir : Path
        Directory containing PLY mesh files (terrain.ply, buildings.ply, etc.)
    output_path : Path
        Path for the output XML file
    validate_meshes : bool, optional
        Whether to check if all mesh files exist, by default True
    
    Returns
    -------
    Path
        Path to the generated XML file
    
    Raises
    ------
    ValueError
        If mesh_dir does not exist or is not a directory
    FileNotFoundError
        If validate_meshes is True and required meshes are missing
    """
    generator = SionnaXMLGenerator(mesh_dir, output_path)
    return generator.generate(validate_meshes)