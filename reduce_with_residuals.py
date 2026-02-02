import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

def combine_frames(file_list, master_bias=None):
    """Stacks multiple FITS files using median combination."""
    if not file_list:
        raise ValueError("File list is empty!")
    data_stack = []
    for f in file_list:
        # Load data and ensure float type for math operations
        data = fits.getdata(f).astype(float)
        if master_bias is not None:
            data -= master_bias
        data_stack.append(data)
    return np.median(data_stack, axis=0)

def reduce_and_plot(sci_paths, bias_paths, dark_paths, flat_paths, arc_paths):
    print("--- Starting Data Reduction & Calibration ---")
    
    # --- 1. Master Calibration Generation ---
    print("Generating Master Frames...")
    master_bias = combine_frames(bias_paths)
    
    # Handle Dark Scaling
    try:
        with fits.open(sci_paths[0]) as h1, fits.open(dark_paths[0]) as h2:
            sci_exp = h1[0].header.get('EXPTIME', 1.0)
            dark_exp = h2[0].header.get('EXPTIME', 1.0)
            dark_scale = sci_exp / dark_exp
    except:
        print("Warning: Could not read exposure times. Assuming dark scale = 1.0")
        dark_scale = 1.0
        
    # Create Master Dark (Bias subtracted, then scaled)
    # Note: We subtract bias from dark frames before stacking
    master_dark = combine_frames(dark_paths, master_bias=master_bias) * dark_scale
    
    # Create Master Flat
    flat_data = combine_frames(flat_paths, master_bias=master_bias)
    master_flat = flat_data / np.median(flat_data)
    master_flat[master_flat < 0.05] = 1.0 # Avoid divide by zero/noise
    
    # --- 2. Extraction Parameters ---
    target_row = 1760 
    r = 5
    
    # --- 3. Wavelength Solution (The "Fitting" Section) ---
    print("Calculating Wavelength Solution...")
    
    # Your manual mapping
    mapping = {
        286: 5852.49, 
        752: 6143.06, 
        1175: 6402.25, 
        1606: 6677.28, 
        2064: 6965.43, 
        2226: 7067.22,
        2552: 7272.94,
        2934: 7503.87
    }
    
    pixels = np.array(list(mapping.keys()))
    wavelengths = np.array(list(mapping.values()))
    
    # Fit a 3rd degree polynomial (Pixel -> Wavelength)
    # 3rd degree handles slight optical distortions better than linear
    fit_coeffs = np.polyfit(pixels, wavelengths, deg=3)
    poly_func = np.poly1d(fit_coeffs)
    
    # Calculate residuals to check fit quality
    residuals = wavelengths - poly_func(pixels)
    print(f"Fit RMS Error: {np.std(residuals):.4f} Angstroms")
    
    # --- 4. Process Science Data ---
    print("Reducing Science Data...")
    sci_raw = combine_frames(sci_paths)
    
    # Full calibration: (Raw - Bias - Scaled_Dark) / Flat
    sci_calibrated = (sci_raw - master_bias - master_dark) / master_flat
    
    # Extract 1D Science Spectrum
    sci_1d = np.sum(sci_calibrated[target_row-r:target_row+r, :], axis=0)
    
    # Generate the wavelength array for every pixel in the CCD
    pixel_axis = np.arange(len(sci_1d))
    wavelength_axis = poly_func(pixel_axis)
    
    # --- 5. Plotting the Final Result ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 3]})
    
    # Subplot 1: Calibration Residuals (Quality Control)
    ax1.scatter(pixels, residuals, color='red', marker='x', s=100)
    ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax1.set_title(f"Wavelength Calibration Residuals (RMS: {np.std(residuals):.3f} $\AA$)")
    ax1.set_ylabel("Residual ($\AA$)")
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: The Final Linearized Spectrum
    ax2.plot(wavelength_axis, sci_1d, color='blue', lw=1.2, label='Science Spectrum')
    ax2.set_title("Final Linearized Spectrum")
    ax2.set_xlabel("Wavelength ($\AA$)")
    ax2.set_ylabel("Intensity (ADU)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Optional: Mark the known lines on the science plot to verify alignment
    for wave in wavelengths:
        ax2.axvline(x=wave, color='green', alpha=0.3, linestyle=':')

    plt.tight_layout()
    plt.show()

# --- Execution ---
sci_files  = ['test_data/' + 'Spectra_Trial-Calibration__60s.fits'] 
bias_files = ['test_data/' + 'Spectra_Trial_Bias_0s_20250910_162704-1.fits'] 
dark_files = ['test_data/' + 'Spectra_Trial-Dark_NEWW_20s_20251014_230009-1.fits'] 
flat_files = ['test_data/' + 'Whijee__Incandescent__20s.fits'] 
arc_files  = ['test_data/' + 'Spectra_Trial-Calibration__60s.fits']

reduce_and_plot(sci_files, bias_files, dark_files, flat_files, arc_files)