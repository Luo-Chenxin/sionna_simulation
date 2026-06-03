from sionna.rt import RadioMaterial
from utils.material_properties import calculate_freshwater_permittivity, calculate_material_properties

# Material Database (Template)
# You can add your own materials here
# Note: "freshwater" does not use a,b,c,d parameters. It uses a specific function instead.
MATERIAL_DATABASE = {
    "asphalt_concrete": {
        "type": "itur_abcd",
        "a": 4.83, "b": 0.0, "c": 0.0108, "d": 1.3969,
        "f_min": 1.0, "f_max": 40.0,
        "color": (0.12, 0.12, 0.13), # Dark gray/asphalt
    },
    "freshwater": {
        "type": "itu_p527",
        "f_min": 0.1,  # Set your preferred minimum frequency in GHz
        "f_max": 1000.0, # Set your preferred maximum frequency in GHz
        "color": (0.00, 0.15, 0.75), # Deep blue water
    },
}

def _format_frequency(freq_hz):
    """
    Converts frequency (in Hz) to the most suitable unit with the fewest zeros, 
    and replaces the decimal point with an underscore.
    Supports input mi.Float, Python float, or numpy scalar.
    """
    # Safely extract Python native float from mi.Float or other array types
    if hasattr(freq_hz, "__len__") or hasattr(freq_hz, "numpy"):
        val = float(freq_hz[0])
    else:
        val = float(freq_hz)
        
    # Automatically selects the optimal unit (preferably keeping the integer part between 1 and 1000).
    if val >= 1e9:
        scaled = val / 1e9
        unit = "ghz"
    elif val >= 1e6:
        scaled = val / 1e6
        unit = "mhz"
    elif val >= 1e3:
        scaled = val / 1e3
        unit = "khz"
    else:
        scaled = val
        unit = "hz"
        
    # Format numbers: Keep one decimal place
    if scaled.is_integer():
        str_val = str(int(scaled))
    else:
        str_val = f"{scaled:.1f}".replace('.', '_')
        
    return f"{str_val}_{unit}"

def create_sionna_material(material_name, frequency_hz):
    """
    Get material parameters, check frequency limits, 
    and return a Sionna RadioMaterial instance.
    
    Parameters:
    - material_name (str): Key from MATERIAL_DATABASE.
    - frequency_hz (float): Frequency in Hz (Sionna default).
    - custom_f_range (tuple, optional): (f_min, f_max) in GHz to override limits.
    """
    # Check if the material exists
    if material_name not in MATERIAL_DATABASE:
        raise ValueError(f"Material '{material_name}' not found in database.")
    
    # Get parameters from database
    mat_data = MATERIAL_DATABASE[material_name]
    mat_type = mat_data["type"]
    mat_color = mat_data["color"]
    
    # Convert frequency from Hz to GHz
    f_ghz = frequency_hz / 1e9
    
    # Set frequency bounds
    f_min, f_max = mat_data["f_min"], mat_data["f_max"]
        
        
    # Check if frequency is in range
    if not (f_min <= f_ghz <= f_max):
        raise ValueError(
            f"Frequency error: {f_ghz:.2f} GHz is outside [{f_min}, {f_max}] GHz "
            f"for '{material_name}'."
        )
        
    # Calculate electrical properties based on material type
    if mat_type == "itur_abcd":
        a, b, c, d = mat_data["a"], mat_data["b"], mat_data["c"], mat_data["d"]
        eps_real, sigma = calculate_material_properties(f_ghz, a, b, c, d)
    elif mat_type == "itu_p527":
        eps_real, sigma = calculate_freshwater_permittivity(f_ghz)
    
    # Create Sionna RadioMaterial object
    # Append frequency to the name for clear tracking inside Sionna
    sionna_name = f"{material_name}_{_format_frequency(frequency_hz)}"
    return RadioMaterial(
        name=sionna_name, 
        relative_permittivity=eps_real, 
        conductivity=sigma,
        color=mat_color,
    )