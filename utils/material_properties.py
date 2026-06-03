import math

DEFAULT_SALINITY = 0.5   # Unit: g/kg or ppt
DEFAULT_TEMPERATURE = 20.0  # Unit: celsius

def calculate_freshwater_permittivity(f_ghz):
    """
    Calculates the real part of the relative permittivity and the equivalent conductivity 
    of surface water (pond/freshwater) based on the ITU-R P.527 Double-Debye model. 
    The default salinity is DEFAULT_SALINITY and the default temperature is DEFAULT_TEMPERATURE.

    Parameters:
        f_ghz (float): Target frequency in GHz.

    Returns:
        tuple: A tuple containing:
        - float: Real part of the relative permittivity.
        - float: Conductivity in S/m.
    """
    # Calculates the ITU variable Theta
    # Formula:：$$\Theta=\frac{300}{T+273.15}-1$$
    theta = (300.0 / (DEFAULT_TEMPERATURE + 273.15)) - 1.0
    
    # Calculates the Double-Debye model parameters
    # The following are the recommended temperature fitting coefficients for pure water
    # Formula: $$\varepsilon_s=77.66+103.3\Theta$$
    #          $$\varepsilon_1=0.0671\varepsilon_s$$
    #          $$\varepsilon_\infty=3.52-7.52\Theta$$
    eps_s = 77.66 - 103.3 * theta
    eps_1 = 0.0671 * eps_s
    eps_inf = 3.52 - 7.52 * theta
    
    # First and second Debye relaxation frequencies (Unit: GHz)
    # Formula: $$f_1=20.20-146.4\Theta+316\Theta^2$$
    #          $$f_2=39.8f_1$$
    f1 = 20.20 - 146.4 * theta + 316 * (theta ** 2)
    f2 = 39.8 * f1
    
    # Double-Debye equation calculation
    # Formula: $$\varepsilon_{ss}=\varepsilon_{s}\exp(-3.33330 \times 10^{-3}S + 4.74868 \times 10^{-6}S^2)$$
    #          $$f_{1s}=f_1(1+S(2.3232 \times 10^{-3}-7.9208 \times 10^{-5}T+3.6764 \times 10^{-6}T^{2}+3.5594 \times 10^{-7}T^{3}+8.9795 \times 10^{-9}T^{4})$$
    #          $$\varepsilon_{1s}=\varepsilon_1\exp(-6.28908 \times 10^{-3}S+1.76032 \times 10^{-4}S^2-9.22144 \times 10^{-5}TS)$$
    #          $$f_{2s}=f_2(1+S(-1.99723 \times 10^{-2}+1.81176 \times 10^{-4}T)$$
    #          $$\varepsilon_{\infty s} = \varepsilon_{\infty}(1+S(-2.04265 \times 10^{-3} + 1.57883 \times 10^{-4}T))$$
    eps_ss = eps_s * math.exp(-3.33330e-3 * DEFAULT_SALINITY + 4.74868e-6 * (DEFAULT_SALINITY ** 2))
    f_1s = f1 * (1 + DEFAULT_SALINITY * (2.3232e-3 - 7.9208e-5 * DEFAULT_TEMPERATURE + 3.6764e-6 * (DEFAULT_TEMPERATURE ** 2) + 3.5594e-7 * (DEFAULT_TEMPERATURE ** 3)  + 8.9795e-9 * (DEFAULT_TEMPERATURE ** 4)))
    eps_1s = eps_1 * math.exp(-6.28908e-3 * DEFAULT_SALINITY + 1.76032e-4 * (DEFAULT_SALINITY ** 2) - 9.22144e-5 * DEFAULT_TEMPERATURE * DEFAULT_SALINITY)
    f_2s = f2 * (1 +  DEFAULT_SALINITY * (-1.99723e-2 + 1.81176e-4 * DEFAULT_TEMPERATURE))
    eps_infs = eps_inf * (1 + DEFAULT_SALINITY * (-2.04265e-3 + 1.57883e-4 * DEFAULT_TEMPERATURE))

    # Extracts the common denominator terms to improve computational efficiency
    # Formula: $$1+(f/f_{1s})^2$$
    #          $$1+(f/f_{2s})^2$$
    denom1 = 1.0 + (f_ghz / f_1s) ** 2
    denom2 = 1.0 + (f_ghz / f_2s) ** 2
    
    # Calculates the real part of the relative permittivity
    # Formula: $$\varepsilon^\prime=\frac{\varepsilon_{ss}-\varepsilon_{1s}}{1+(f/f_{1s})^2}+\ \frac{\varepsilon_{1s}-\varepsilon_{\infty s}}{1+(f/f_{2s})^2}+\varepsilon_{\infty s}$$
    eps_real = ((eps_ss - eps_1s) / denom1) + ((eps_1s - eps_infs) / denom2) + eps_inf

    # Calculates the static conductivity sigma_s [S/m]
    # Formula: $$\sigma_{35}=2.903602+8.607\times{10}^{-2}T+4.738817\times{10}^{-4}T^2-2.991\times{10}^{-6}T^3+4.3047\times{10}^{-9}T^4$$
    #          $$R_{15}=S\frac{(37.5109+5.45216S+1.4409\times{10}^{-2}S^2)}{(1004.75+182.283S+S^2)}$$
    #          $$\alpha_0=\frac{(6.9431+3.2841S-9.9486\times{10}^{-2}S^2)}{(84.850+69.024S+S^2)}$$
    #          $$\alpha_1=49.843-0.2276S+0.198\times{10}^{-2}S^2$$
    #          $$R_{T15}=1+\frac{\alpha_0(T-15)}{(\alpha_1+T)}$$
    #          $$\sigma_{s}=\sigma_{35}R_{15}R_{T15}$$
    sigma_35 = 2.903602 + 8.607e-2 * DEFAULT_TEMPERATURE + 4.738817e-4 * (DEFAULT_TEMPERATURE ** 2) - 2.991e-6 * (DEFAULT_TEMPERATURE ** 3) + 4.3047e-9 * (DEFAULT_TEMPERATURE ** 4)
    R_15 = DEFAULT_SALINITY * (37.5109 + 5.45216 * DEFAULT_SALINITY + 1.4409e-2 * (DEFAULT_SALINITY ** 2)) / (1004.75 + 182.283 * DEFAULT_SALINITY + DEFAULT_SALINITY ** 2)
    alpha_0 = (6.9431 + 3.2841 * DEFAULT_SALINITY - 9.9486e-2 * (DEFAULT_SALINITY ** 2)) / (84.850 + 69.024 * DEFAULT_SALINITY + DEFAULT_SALINITY ** 2)
    alpha_1 = 49.843 - 0.2276 * DEFAULT_SALINITY + 0.198e-2 * (DEFAULT_SALINITY ** 2)
    R_T15 = 1 + alpha_0 * (DEFAULT_TEMPERATURE - 15) / (alpha_1 + DEFAULT_TEMPERATURE)
    sigma_s = sigma_35 * R_15 * R_T15
    
    # Calculates the imaginary part of the relative permittivity
    # Formula: $$\varepsilon^{\prime\prime}=\frac{(f/f_{1s})(\varepsilon_{ss}-\varepsilon_{1s})}{1+(f/f_{1s})^2}\ +\frac{(f/f_{2s})(\varepsilon_{1s}-\varepsilon_{\infty s})}{1+(f/f_{2s})^2}+\frac{18\sigma}{f}$$
    eps_imag = ((f_ghz/f_1s) * (eps_ss - eps_1s)) / denom1 + ((f_ghz/f_2s) * (eps_1s - eps_infs)) / denom2 + (18 * sigma_s)/f_ghz
    
    # Converts the imaginary part into equivalent conductivity sigma [S/m]
    # Formula: $$\sigma=0.05563f\varepsilon^{\prime\prime}$$
    sigma = 0.05563 * f_ghz * eps_imag
    
    return eps_real, sigma


def calculate_material_properties(f_ghz, a, b, c, d):
    """
    Calculate the real part of relative permittivity and conductivity of a material.
    Formula source: [ITURP20403]
    
    Parameters:
        f_ghz (float): Frequency in GHz.
        a, b, c, d (float): Material-specific constants.
    
    Returns:
        tuple: (eps_real, sigma)
            - eps_real: Real part of relative permittivity
            - sigma: Conductivity in [S/m]
    """
    # Calculate the real part of relative permittivity
    eps_real = a * (f_ghz ** b)
    
    # Calculate the conductivity [S/m]
    sigma = c * (f_ghz ** d)
    
    return eps_real, sigma