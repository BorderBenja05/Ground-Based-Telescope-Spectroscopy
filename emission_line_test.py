import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

def combine_frames(file_list, master_bias=None):
    """Stacks multiple FITS files using median combination."""
    if not file_list:
        raise ValueError("File list is empty!")
    data_stack = []
    for f in file_list:
        data = fits.getdata(f).astype(float)
        if master_bias is not None:
            data -= master_bias
        data_stack.append(data)
    return np.median(data_stack, axis=0)

def diagnostic_plot(sci_paths, bias_paths, dark_paths, flat_paths, arc_paths):
    print("--- Running Diagnostic Raw Mapping Check ---")
    
    # 1. Basic CCD Corrections
    master_bias = combine_frames(bias_paths)
    
    # Simple Dark scaling
    try:
        with fits.open(sci_paths[0]) as h1, fits.open(dark_paths[0]) as h2:
            sci_exp = h1[0].header.get('EXPTIME', 1.0)
            dark_exp = h2[0].header.get('EXPTIME', 1.0)
            dark_scale = sci_exp / dark_exp
    except:
        dark_scale = 1.0
        
    master_dark = combine_frames(dark_paths, master_bias=master_bias) * dark_scale
    
    # Flat normalization
    flat_data = combine_frames(flat_paths, master_bias=master_bias)
    master_flat = flat_data / np.median(flat_data)
    master_flat[master_flat < 0.05] = 1.0 

    # 2. Extract Arc Spectrum in Pixel Space
    arc_raw = combine_frames(arc_paths)
    arc_calibrated = (arc_raw - master_bias) / master_flat
    
    # Using your extraction parameters from the main script
    target_row = 1760 # Approximate center based on your search region
    r = 5
    arc_1d = np.sum(arc_calibrated[target_row-r:target_row+r, :], axis=0)

    # 3. Your Manual Mapping (Absolute Facts)
    mapping = {
        286: 5852.49, 
        752: 6143.06, 
        1423: 6402.25, 
        1606: 6677.28, 
        2064: 6965.43, 
        2226: 7067.22,
        2552: 7272.94
    }

    # 4. Plotting
    plt.figure(figsize=(15, 7))
    plt.plot(arc_1d, color='black', lw=1, label='Extracted Arc (Pixel Space)')
    
    # Overlay your calibration points
    for pix, wave in mapping.items():
        plt.axvline(x=pix, color='red', linestyle='--', alpha=0.6)
        plt.text(pix, np.max(arc_1d)*0.9, f"{wave}Å\n(Px {pix})", 
                 rotation=90, verticalalignment='top', fontsize=9, color='red')

    plt.title("Diagnostic: Raw Arc Spectrum vs. Manual Calibration Points")
    plt.xlabel("Pixel Coordinate (X-axis)")
    plt.ylabel("Intensity (ADU)")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

# --- Execution ---
# Using your exact file paths
sci_files  = ['Spectra_Trial-Calibration__60s.fits'] 
bias_files = ['Spectra_Trial_Bias_0s_20250910_162704-1.fits'] 
dark_files = ['Spectra_Trial-Dark_NEWW_20s_20251014_230009-1.fits'] 
flat_files = ['Whijee__Incandescent__20s.fits'] 
arc_files  = ['Spectra_Trial-Calibration__60s.fits']

diagnostic_plot(sci_files, bias_files, dark_files, flat_files, arc_files)