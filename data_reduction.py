import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import find_peaks
import glob

def combine_frames(file_list, master_bias=None):
    """Stacks multiple FITS files using median combination."""
    if not file_list:
        raise ValueError("File list is empty! Check your file paths/patterns.")
    data_stack = []
    for f in file_list:
        data = fits.getdata(f).astype(float)
        if master_bias is not None:
            data -= master_bias
        data_stack.append(data)
    return np.median(data_stack, axis=0)

def reduce_spectrum(sci_paths, bias_paths, dark_paths, flat_paths, arc_paths):
    print("--- Starting Full Reduction Pipeline ---")
    
    # 1. Master Bias
    print(f"1. Creating Master Bias...")
    master_bias = combine_frames(bias_paths)

    # 2. Master Dark
    print(f"2. Creating Master Dark...")
    # Attempt to read exposure times for scaling
    try:
        with fits.open(sci_paths[0]) as h1, fits.open(dark_paths[0]) as h2:
            sci_exp = h1[0].header.get('EXPTIME', 1.0)
            dark_exp = h2[0].header.get('EXPTIME', 1.0)
            dark_scale = sci_exp / dark_exp
    except:
        dark_scale = 1.0
        
    raw_master_dark = combine_frames(dark_paths, master_bias=master_bias)
    master_dark = raw_master_dark * dark_scale

    # 3. Master Flat
    print(f"3. Creating Master Flat...")
    flat_data = combine_frames(flat_paths, master_bias=master_bias)
    flat_norm = np.median(flat_data)
    master_flat = flat_data / (flat_norm if flat_norm > 0 else 1)
    master_flat[master_flat < 0.05] = 1.0 

    # 4. Science Reduction
    print(f"4. Reducing Science Images...")
    sci_raw = combine_frames(sci_paths)
    sci_calibrated = (sci_raw - master_bias - master_dark) / master_flat
    sci_calibrated = np.nan_to_num(sci_calibrated)

    # 5. Arc Reduction
    print(f"5. Processing Arc Calibration...")
    arc_raw = combine_frames(arc_paths)
    arc_calibrated = (arc_raw - master_bias) / master_flat

    # 6. Extraction
    print("6. Extracting 1D Spectrum...")
    
    # --- UPDATED: Search Region (Centered on 1760) ---
    search_center = 1760
    search_width = 50
    # Create a slice to find the star's vertical center
    search_slice = sci_calibrated[search_center-search_width : search_center+search_width, :]
    vertical_profile = np.median(search_slice, axis=1)
    
    # Find the row with the brightest pixels in this slice
    target_row_relative = np.argmax(vertical_profile)
    target_row = target_row_relative + (search_center - search_width)
    
    print(f"   -> Star detected at Row {target_row}")

    # --- UPDATED: Background Row ---
    bg_row = 1632

    # Extract Flux (Sum over +/- 5 pixels)
    r = 5
    sci_1d = np.sum(sci_calibrated[target_row-r:target_row+r, :], axis=0)
    bg_1d  = np.sum(sci_calibrated[bg_row-r:bg_row+r, :], axis=0)
    spectrum_1d = sci_1d - bg_1d
    
    # Extract Arc (same row as star)
    arc_1d = np.sum(arc_calibrated[target_row-r:target_row+r, :], axis=0)

    # 7. Smart Wavelength Calibration
    print("7. Calibrating Wavelengths (Auto-Refining)...")
    
    # Your approximate mapping
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
    
    pixel_coords = []
    wave_coords = []

    # --- AUTO-CORRECTION LOOP ---
    # This looks at your guess (e.g. 300) and finds the REAL peak nearby
    for guess_pix, wave in mapping.items():
        window = 30 # Search +/- 30 pixels around your guess
        start = int(max(0, guess_pix - window))
        end = int(min(len(arc_1d), guess_pix + window))
        
        # Find index of max value in this window
        local_peak_idx = np.argmax(arc_1d[start:end])
        real_pix = start + local_peak_idx
        
        pixel_coords.append(real_pix)
        wave_coords.append(wave)
        print(f"   Map: Guess {guess_pix} -> Snapped to {real_pix} -> Wave {wave}")

    # Fit polynomial to the REFINED pixels
    poly = np.polyfit(pixel_coords, wave_coords, 2)
    wavelengths = np.polyval(poly, np.arange(len(spectrum_1d)))

    # --- MANUAL SHIFT CORRECTION ---
    # If the line is still slightly off, change this value!
    # Positive = Shifts plot Right. Negative = Shifts plot Left.
    shift_correction = 0.0 
    wavelengths = wavelengths + shift_correction
    if shift_correction != 0:
        print(f"   -> Applied manual shift of {shift_correction} Angstroms")

    # 8. Plotting
    print("8. Generating Plots...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot Science
    ax1.plot(wavelengths, spectrum_1d, color='black', lw=1, label='Extracted Spectrum')
    ax1.set_title("Calibrated Science Spectrum")
    ax1.set_ylabel("Flux")
    ax1.grid(True, alpha=0.3)

    # Plot Arc
    ax2.step(wavelengths, arc_1d / np.max(arc_1d), color='midnightblue', where='mid')
    ax2.set_title("Calibrated Arc Spectrum (Reference)")
    ax2.set_xlabel("Wavelength (Å)")
    ax2.set_ylabel("Normalized Intensity")
    ax2.grid(True, alpha=0.3)
    
    # --- Vertical Line Check ---
    # H-alpha is 6562.8 Angstroms
    line_wave = 6562.8
    
    ax1.axvline(line_wave, color='red', linestyle='--', alpha=0.8, label='H-alpha')
    ax1.legend()
    ax2.axvline(line_wave, color='red', linestyle='--', alpha=0.8)
    
    # Optional: Plot the points used for calibration on the bottom graph
    ax2.scatter(wave_coords, np.ones_like(wave_coords)*0.9, color='red', marker='x', label='Calib Points')
    ax2.legend()

    plt.tight_layout()
    plt.show()

# --- Execution ---
data_dir = '/home/borderbenja/mlof/demo_data/'

# These match the file names you provided previously
sci_files  = ['Spectra_Trial-Calibration__60s.fits'] 
bias_files = ['Spectra_Trial_Bias_0s_20250910_162704-1.fits'] 
dark_files = ['Spectra_Trial-Dark_NEWW_20s_20251014_230009-1.fits'] 
flat_files = ['Whijee__Incandescent__20s.fits'] 
arc_files  = ['Spectra_Trial-Calibration__60s.fits']

reduce_spectrum(sci_files, bias_files, dark_files, flat_files, arc_files)