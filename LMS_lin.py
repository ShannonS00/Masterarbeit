# Detlin analysis functions 

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import glob
from scipy.ndimage import median_filter
from collections import defaultdict
import os 


# 1 Background subtraction and averaging for a set of observations (per DIT)
DETECTORS = ['DET1.DATA', 'DET2.DATA', 'DET3.DATA', 'DET4.DATA']

def process_observation_set(sci_files, dark_files, output_dir=None):
    # --- 1. Create Master Darks per detector (Group by DIT and Median) ---
    dark_groups = {det: defaultdict(list) for det in DETECTORS}
    for d_f in dark_files:
        with fits.open(d_f) as h:
            d_dit = h[0].header.get('HIERARCH ESO DET DIT')
            for det in DETECTORS:
                dark_groups[det][d_dit].append(h[det].data.astype(np.float64))

    master_darks = {
        det: {dit: np.median(frames, axis=0) for dit, frames in dark_groups[det].items()}
        for det in DETECTORS
    }

    # --- 2. Subtract Master Dark from Science (all detectors) ---
    calibrated_results = {det: defaultdict(list) for det in DETECTORS}
    for sci_f in sci_files:
        with fits.open(sci_f) as h_sci:
            dit = h_sci[0].header.get('HIERARCH ESO DET DIT')
            for det in DETECTORS:
                if dit in master_darks[det]:
                    sci_data = h_sci[det].data.astype(np.float64)
                    subtracted = sci_data - master_darks[det][dit]
                    calibrated_results[det][dit].append(subtracted)

    # --- 3. Average Observations, Sort by DIT & Save multi-extension FITS ---
    final_images = {det: [] for det in DETECTORS}
    final_times = []

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    for dit in sorted(calibrated_results[DETECTORS[0]].keys()):
        final_times.append(dit)
        ndit = len(calibrated_results[DETECTORS[0]][dit])

        if output_dir is not None:
            primary_hdu = fits.PrimaryHDU()
            primary_hdu.header['HIERARCH ESO DET DIT'] = (dit, 'Integration time [s]')
            primary_hdu.header['HIERARCH ESO DET NDIT'] = (ndit, 'Number of averaged exposures')
            primary_hdu.header['COMMENT'] = 'Dark-subtracted, averaged science frame'
            hdul = fits.HDUList([primary_hdu])

        for det in DETECTORS:
            combined_frame = np.mean(calibrated_results[det][dit], axis=0)
            final_images[det].append(combined_frame)

            if output_dir is not None:
                img_hdu = fits.ImageHDU(data=combined_frame.astype(np.float32), name=det)
                img_hdu.header['BUNIT'] = 'counts'
                hdul.append(img_hdu)

        if output_dir is not None:
            out_path = os.path.join(output_dir, f"dark_subtracted_DIT{dit:.3f}.fits")
            hdul.writeto(out_path, overwrite=True)

    final_images = {det: np.array(frames) for det, frames in final_images.items()}

    print(f"Processed {len(final_times)} unique DITs.")
    if output_dir is not None:
        print(f"Saved dark-subtracted frames ({len(DETECTORS)} detectors) to: {output_dir}/")
    return final_images, np.array(final_times)


def fit_linear_reference(science_ramps, times, sat_limit=49_900,
                          detector='DET1.DATA', illum_threshold=3000):
    """
    Fit a per-pixel linear reference through the low-signal (linear) regime
    and extrapolate it across all DITs.

    For each illuminated pixel the function:
      1. Selects only the DITs where that pixel's signal < sat_limit
         (the regime where the detector is still linear).
      2. Fits  S_ideal(t) = slope * t + intercept  by least squares.
      3. Extrapolates the line to ALL DITs to produce the ideal linear signal.

    The mask is applied per-pixel, so pixels that saturate earlier still get
    a reliable linear fit from their own low-signal points.

    Parameters
    ----------
    science_ramps : dict {detector: ndarray} or ndarray (n_dits, h, w)
        Dark-subtracted, averaged ramp cube from process_observation_set.
    times : array-like, shape (n_dits,)
        DIT values in seconds, in any order.
    sat_limit : float
        Signal threshold in ADU below which the detector is treated as linear.
        Default 50 000.
    detector : str
        Key to use when science_ramps is a dict.
    illum_threshold : float
        Pixels whose median signal is below this value are treated as dark /
        unilluminated and skipped (their ideal cube entries stay 0).

    Returns
    -------
    ideal_cube : ndarray (n_dits, h, w)
        Ideal linear signal at every DIT for every pixel.
        Unilluminated pixels are 0.
    slopes : ndarray (h, w)
        Per-pixel fitted slope [ADU / s].  0 for unilluminated pixels.
    intercepts : ndarray (h, w)
        Per-pixel fitted intercept [ADU].  0 for unilluminated pixels.
    """
    if isinstance(science_ramps, dict):
        science_ramps = science_ramps[detector]
    science_ramps = np.asarray(science_ramps)

    t = np.array(times, dtype=float)
    sort_idx = np.argsort(t)
    t = t[sort_idx]
    ramps_sorted = science_ramps[sort_idx].astype(float)   # (n_dits, h, w)
    n_dits, h, w = ramps_sorted.shape

    # --- Illuminated pixel mask ---
    illum_mask = np.median(ramps_sorted, axis=0) > illum_threshold  # (h, w)
    ramps_px   = ramps_sorted[:, illum_mask]                        # (n_dits, n_pixels)
    n_pixels   = ramps_px.shape[1]
    print(f"Illuminated pixels : {n_pixels:,}  /  {h * w:,}")

    # --- Per-pixel linear regime mask: signal < sat_limit ---
    lin_mask = ramps_px < sat_limit                                  # (n_dits, n_pixels)
    n_lin    = lin_mask.sum(axis=0)                                  # (n_pixels,)
    print(f"Linear-regime DITs : min {n_lin.min()}  max {n_lin.max()}  "
          f"median {int(np.median(n_lin))}  (per pixel, out of {n_dits})")

    # --- Vectorised per-pixel least-squares via normal equations ---
    # For pixel p with mask m_p:
    #   slope_p = (N*sum(t*S) - sum(t)*sum(S)) / (N*sum(t^2) - sum(t)^2)
    #   intercept_p = (sum(S) - slope_p*sum(t)) / N
    t_bc  = t[:, np.newaxis]                                         # (n_dits, 1) broadcast

    sum_t  = (t_bc  * lin_mask).sum(axis=0)                         # (n_pixels,)
    sum_t2 = (t_bc**2 * lin_mask).sum(axis=0)
    sum_s  = (ramps_px * lin_mask).sum(axis=0)
    sum_ts = (t_bc * ramps_px * lin_mask).sum(axis=0)

    denom      = n_lin * sum_t2 - sum_t ** 2
    safe_denom = np.where(denom == 0, 1, denom)                     # avoid div-by-zero

    slopes_px     = (n_lin * sum_ts - sum_t * sum_s) / safe_denom   # (n_pixels,)
    intercepts_px = (sum_s - slopes_px * sum_t) / np.where(n_lin == 0, 1, n_lin)

    # Zero out pixels with fewer than 2 linear-regime points (can't fit a line)
    bad = n_lin < 2
    slopes_px[bad]     = 0.0
    intercepts_px[bad] = 0.0
    if bad.any():
        print(f"Warning: {bad.sum()} pixels had < 2 linear-regime DITs — set to 0.")

    # --- Extrapolate to ALL DITs ---
    ideal_px = slopes_px[np.newaxis, :] * t[:, np.newaxis] + intercepts_px[np.newaxis, :]
    #                                                          (n_dits, n_pixels)

    # --- Map back to full detector arrays ---
    ideal_cube  = np.zeros((n_dits, h, w), dtype=float)
    slopes_map  = np.zeros((h, w),         dtype=float)
    intercepts_map = np.zeros((h, w),      dtype=float)

    ideal_cube[:, illum_mask]  = ideal_px
    slopes_map[illum_mask]     = slopes_px
    intercepts_map[illum_mask] = intercepts_px

    print("Linear reference fit complete.")
    return ideal_cube, slopes_map, intercepts_map


def compute_deviation(science_ramps, ideal_cube, illum_threshold=3000,
                      detector='DET1.DATA', plot=True, sat_limit=None):
    """
    Compute per-pixel deviation from the ideal linear reference.

    For every DIT and every illuminated pixel:
        ratio      = S_obs / S_ideal          (1.0 = perfectly linear)
        difference = S_obs - S_ideal  [ADU]   (0   = perfectly linear)

    Both are returned as cubes of shape (n_dits, h, w) so they can be used
    directly for polynomial fitting (x = S_obs, y = ratio or difference).

    Parameters
    ----------
    science_ramps   : dict {detector: ndarray} or ndarray (n_dits, h, w)
    ideal_cube      : ndarray (n_dits, h, w)
        Output of fit_linear_reference — the extrapolated ideal linear signal.
    illum_threshold : float
        Pixels whose median signal is below this are masked out (set to NaN).
    detector        : str
        Key used when science_ramps is a dict.
    plot            : bool
        If True, show a scatter plot of ratio vs S_obs for all illuminated pixels.
    sat_limit       : float or None
        If given, draws a vertical line on the plot at the saturation threshold.

    Returns
    -------
    ratio_cube : ndarray (n_dits, h, w)
        S_obs / S_ideal.  NaN for unilluminated pixels and where S_ideal == 0.
    diff_cube  : ndarray (n_dits, h, w)
        S_obs - S_ideal.  NaN for unilluminated pixels.
    """
    if isinstance(science_ramps, dict):
        science_ramps = science_ramps[detector]
    science_ramps = np.asarray(science_ramps)

    ramps = science_ramps.astype(float)              # (n_dits, h, w)
    n_dits, h, w = ramps.shape

    # --- Illumination mask ---
    illum_mask = np.median(ramps, axis=0) > illum_threshold   # (h, w)

    # --- Ratio and difference, NaN outside illuminated area ---
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_cube = np.where(
            illum_mask[np.newaxis] & (ideal_cube != 0),
            ramps / ideal_cube,
            np.nan
        )

    diff_cube = np.where(
        illum_mask[np.newaxis],
        ramps - ideal_cube,
        np.nan
    )

    # --- Diagnostic stats ---
    valid_ratio = ratio_cube[np.isfinite(ratio_cube)]
    print(f"Ratio  — median: {np.median(valid_ratio):.4f}  "
          f"std: {np.std(valid_ratio):.4f}  "
          f"min: {np.min(valid_ratio):.4f}  "
          f"max: {np.max(valid_ratio):.4f}")

    # --- Optional plot: ratio and difference vs S_obs ---
    if plot:
        signal_flat = ramps[:, illum_mask].ravel()
        ratio_flat  = ratio_cube[:, illum_mask].ravel()
        diff_flat   = diff_cube[:, illum_mask].ravel()

        valid = np.isfinite(ratio_flat) & (signal_flat > 0)
        rng = np.random.default_rng(0)
        idx = rng.choice(np.sum(valid),
                         size=min(40_000, int(np.sum(valid))), replace=False)

        median_signal = np.nanmedian(ramps[:, illum_mask], axis=1)   # one per DIT
        median_ratio  = np.nanmedian(ratio_cube[:, illum_mask], axis=1)
        median_diff   = np.nanmedian(diff_cube[:, illum_mask], axis=1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Ratio panel
        axes[0].scatter(signal_flat[valid][idx], ratio_flat[valid][idx],
                        s=1, alpha=0.15, color='steelblue')
        axes[0].plot(median_signal, median_ratio, 'o-', color='red',
                     markersize=5, linewidth=1.5, label='Median per DIT')
        axes[0].axhline(1.0, color='gray', linestyle=':', linewidth=1)
        if sat_limit is not None:
            axes[0].axvline(sat_limit, color='orange', linestyle='--',
                            alpha=0.7, label=f'sat_limit = {sat_limit:,}')
        axes[0].set_xlabel('Observed signal  S_obs  [ADU]')
        axes[0].set_ylabel('S_obs / S_ideal')
        axes[0].set_title('Deviation from linearity — ratio')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # Difference panel
        axes[1].scatter(signal_flat[valid][idx], diff_flat[valid][idx],
                        s=1, alpha=0.15, color='steelblue')
        axes[1].plot(median_signal, median_diff, 'o-', color='red',
                     markersize=5, linewidth=1.5, label='Median per DIT')
        axes[1].axhline(0.0, color='gray', linestyle=':', linewidth=1)
        if sat_limit is not None:
            axes[1].axvline(sat_limit, color='orange', linestyle='--',
                            alpha=0.7, label=f'sat_limit = {sat_limit:,}')
        axes[1].set_xlabel('Observed signal  S_obs  [ADU]')
        axes[1].set_ylabel('S_obs − S_ideal  [ADU]')
        axes[1].set_title('Deviation from linearity — difference')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return ratio_cube, diff_cube


def fit_nonlinearity_poly(science_ramps, ratio_cube, s_ref=None,
                           poly_order=3, illum_threshold=3000,
                           detector='DET1.DATA', plot=True,
                           output_path=None, header=None):
    """
    Fit a per-pixel polynomial to the linearity correction factor.

    Following the JWST convention, the constant term is fixed at 1.0 to
    enforce C(0) = 1 (no correction at zero signal).  Only the deviation
    delta(u) = C(u) - 1 is fitted — poly_order free parameters, no a0:

        C(u) = 1  +  c_n*u^n  +  ...  +  c_1*u

    where  u = S_obs / s_ref  (dimensionless, values in [0, 1])
           C = S_ideal / S_obs  (>1 where detector compresses signal)

    Fixing a0 = 1 avoids the large alternating-sign coefficients that arise
    when the constant term is left free.

    To correct science data:
        S_corrected = S_obs  *  C(S_obs / s_ref)
        i.e.  S_corrected = S_obs  *  polyval(coeffs_cube[:, y, x], u)

    Parameters
    ----------
    science_ramps   : dict {detector: ndarray} or ndarray (n_dits, h, w)
        Observed signal ramps (same as passed to compute_deviation).
    ratio_cube      : ndarray (n_dits, h, w)
        S_obs / S_ideal from compute_deviation.  NaN for unilluminated pixels.
    s_ref           : float or None
        Normalisation reference signal [ADU].  If None, the maximum finite
        observed signal across all illuminated pixels is used automatically.
        Recommended: use a fixed physical value (e.g. well capacity ~50 000)
        so coefficients are comparable across datasets.
    poly_order      : int
        Polynomial degree (default 3).  poly_order free parameters are fit
        (the a0 = 1 constraint removes one degree of freedom).
    illum_threshold : float
        Minimum median signal to treat a pixel as illuminated.
    detector        : str
        Key used when science_ramps is a dict.
    plot            : bool
        If True, show scatter of C vs u and per-pixel fit for the median pixel.
    output_path     : str or None
        If given, write coefficient cube to a multi-extension FITS file.
    header          : fits.Header or None
        Base header written to the FITS primary HDU.

    Returns
    -------
    coeffs_cube : ndarray (poly_order+1, h, w)
        Per-pixel polynomial coefficients [c_n, ..., c_1, 1.0].
        coeffs_cube[:, y, x] can be passed directly to np.polyval.
        The last plane (a0) is 1.0 for illuminated pixels, 0.0 for dark.
    s_ref : float
        The normalisation reference actually used (stored in FITS header).
    """
    if isinstance(science_ramps, dict):
        science_ramps = science_ramps[detector]
    science_ramps = np.asarray(science_ramps)

    ramps = science_ramps.astype(float)             # (n_dits, h, w)
    n_dits, h, w = ramps.shape

    # --- Illumination mask ---
    illum_mask = np.median(ramps, axis=0) > illum_threshold
    ramps_px   = ramps[:, illum_mask]               # (n_dits, n_pixels)
    ratio_px   = ratio_cube[:, illum_mask]          # (n_dits, n_pixels)
    n_pixels   = ramps_px.shape[1]
    print(f"Illuminated pixels : {n_pixels:,}  /  {h * w:,}")

    # --- Correction factor  C = S_ideal / S_obs = 1 / ratio ---
    with np.errstate(divide='ignore', invalid='ignore'):
        corr_px = np.where(ratio_px > 0, 1.0 / ratio_px, np.nan)
    # (n_dits, n_pixels)

    # --- Normalised signal  u = S_obs / s_ref ---
    if s_ref is None:
        s_ref = float(np.nanmax(ramps_px[np.isfinite(ratio_px)]))
        print(f"s_ref (auto) = {s_ref:,.0f} ADU  "
              f"— pass s_ref=<value> to fix it (e.g. well capacity)")
    else:
        print(f"s_ref = {s_ref:,.0f} ADU")

    u_px = ramps_px / s_ref                         # (n_dits, n_pixels), values in [0, ~1]

    # --- Valid data points: both u and C must be finite ---
    valid = np.isfinite(u_px) & np.isfinite(corr_px)   # (n_dits, n_pixels)
    n_valid = valid.sum(axis=0)                          # (n_pixels,)
    print(f"Valid (signal, correction) pairs per pixel: "
          f"min {n_valid.min()}  max {n_valid.max()}  "
          f"median {int(np.median(n_valid))}")

    # --- Constrained fit: delta(u) = C(u) - 1  (no constant term) ---
    # Vandermonde WITHOUT the u^0 column: powers = [poly_order, ..., 1]
    # This enforces C(0) = 1 exactly, following the JWST convention.
    delta_px = corr_px - 1.0                            # (n_dits, n_pixels)
    powers   = np.arange(poly_order, 0, -1)             # [3, 2, 1]
    V = u_px[:, :, np.newaxis] ** powers                # (n_dits, n_pixels, poly_order)

    V_masked     = V        * valid[:, :, np.newaxis]   # (n_dits, n_pixels, poly_order)
    delta_masked = delta_px * valid                      # (n_dits, n_pixels)

    VTV = np.einsum('npi,npj->pij', V_masked, V_masked)     # (n_pixels, M, M)
    VTy = np.einsum('npi,np->pi',   V_masked, delta_masked)  # (n_pixels, M)

    # Guard against pixels with too few valid points
    bad = n_valid < poly_order
    if bad.any():
        print(f"Warning: {bad.sum()} pixels have fewer valid points than "
              f"polynomial terms — deviation coefficients set to 0.")
        VTV[bad] += np.eye(poly_order) * 1e-30

    delta_coeffs_px = np.linalg.solve(VTV, VTy[:, :, np.newaxis]).squeeze(-1)
    # (n_pixels, poly_order)  — coefficients for u^poly_order ... u^1
    delta_coeffs_px[bad] = 0.0

    # --- Assemble full coeffs_cube: [c_n, ..., c_1, a0=1.0] ---
    # Shape (poly_order+1, h, w) — compatible with np.polyval and
    # apply_nonlinearity_correction without any changes downstream.
    coeffs_cube = np.zeros((poly_order + 1, h, w), dtype=float)
    coeffs_cube[:poly_order, illum_mask] = delta_coeffs_px.T   # c_n ... c_1
    coeffs_cube[poly_order,  illum_mask] = 1.0                  # a0 fixed at 1

    # --- Optional diagnostic plot ---
    if plot:
        u_flat    = u_px[valid].ravel()
        corr_flat = corr_px[valid].ravel()

        rng = np.random.default_rng(0)
        idx = rng.choice(len(u_flat), size=min(40_000, len(u_flat)), replace=False)

        # Full coefficients for the median pixel: [c_n, ..., c_1, 1.0]
        med_px    = int(np.median(np.where(~bad)[0])) if (~bad).any() else 0
        u_line    = np.linspace(0, 1, 500)
        full_coeffs_med = np.append(delta_coeffs_px[med_px], 1.0)
        fit_curve = np.polyval(full_coeffs_med, u_line)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(u_flat[idx], corr_flat[idx],
                   s=1, alpha=0.15, color='steelblue',
                   label=f'All illuminated pixels (N={len(u_flat):,})')
        ax.plot(u_line, fit_curve, '-', color='red', linewidth=2,
                label=f'Degree-{poly_order} fit, a0=1 fixed (median pixel)')
        ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)
        ax.set_xlabel(f'Normalised signal  u = S_obs / s_ref  (s_ref = {s_ref:,.0f} ADU)')
        ax.set_ylabel('Correction factor  C = S_ideal / S_obs')
        ax.set_title('Non-linearity polynomial fit  —  C(u) = 1 + delta(u)  per pixel')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # --- Optional FITS output ---
    if output_path is not None:
        hdr = header.copy() if header is not None else fits.Header()
        hdr['POLYDEG']  = (poly_order, 'Polynomial degree (a0 fixed at 1)')
        hdr['S_REF']    = (s_ref,      'Normalisation reference signal [ADU]')
        hdr['DETECTOR'] = (detector,   'Detector identifier')
        hdr['COMMENT']  = ('Non-linearity: S_corr = S_obs * polyval(coeffs, S_obs/S_REF)')
        hdr['COMMENT']  = ('a0 = 1.0 fixed (JWST convention: no correction at 0 signal)')

        primary = fits.PrimaryHDU(header=hdr)
        hdul    = fits.HDUList([primary])
        for k, degree in enumerate(range(poly_order, -1, -1)):
            img = fits.ImageHDU(data=coeffs_cube[k].astype(np.float32),
                                name=f'COEFF_A{degree}')
            img.header['DEGREE']  = (degree, f'Coefficient for u^{degree} term')
            img.header['S_REF']   = (s_ref,  'Normalisation reference [ADU]')
            if degree == 0:
                img.header['COMMENT'] = ('a0 fixed at 1.0 — not fitted')
            hdul.append(img)
        hdul.writeto(output_path, overwrite=True)
        print(f"Saved non-linearity coefficients to: {output_path}")

    print("Non-linearity polynomial fit complete.")
    return coeffs_cube, s_ref


def apply_nonlinearity_correction(science_ramps, coeffs_cube, s_ref,
                                   detector='DET1.DATA'):
    """
    Apply the non-linearity correction.

        S_corrected = S_obs  *  C(S_obs / s_ref)

    where  C(u) = np.polyval(coeffs_cube[:, y, x], u)  and  u = S_obs / s_ref.

    Parameters
    ----------
    science_ramps : dict or ndarray (h, w) or (n_dits, h, w)
    coeffs_cube   : ndarray (poly_order+1, h, w)  from fit_nonlinearity_poly
    s_ref         : float  — same value used during fitting
    detector      : str

    Returns
    -------
    corrected : ndarray, same shape as input ramps
    """
    if isinstance(science_ramps, dict):
        science_ramps = science_ramps[detector]
    science_ramps = np.asarray(science_ramps)

    single = science_ramps.ndim == 2
    ramps  = science_ramps[np.newaxis].astype(float) if single else science_ramps.astype(float)

    u = ramps / s_ref                               # normalised signal, (n_dits, h, w)

    # Horner's method per DIT: C(u) = ((a_n*u + a_{n-1})*u + ...)*u + a_0
    for k in range(ramps.shape[0]):
        Cu = coeffs_cube[0].astype(float)           # start with highest-degree coeff
        for coef in coeffs_cube[1:]:
            Cu = Cu * u[k] + coef                   # accumulate Horner step
        ramps[k] = ramps[k] * Cu                    # S_corrected = S_obs * C(u)

    return ramps[0] if single else ramps


def validate_correction(science_ramps, ideal_cube, coeffs_cube, s_ref,
                         illum_threshold=3000, detector='DET1.DATA',
                         sat_limit=None, title=''):
    """
    Validate the non-linearity correction by comparing before and after.

    Applies the correction polynomial to the raw ramps, then computes:
        before  :  S_obs       / S_ideal   (raw deviation)
        after   :  S_corrected / S_ideal   (residual — should be ~1.0 everywhere)

    Three-panel plot:
        Left   — before: S_obs / S_ideal vs S_obs
        Centre — after:  S_corrected / S_ideal vs S_obs
        Right  — residual std per DIT (quantifies improvement)

    Parameters
    ----------
    science_ramps : dict or ndarray (n_dits, h, w)
    ideal_cube    : ndarray (n_dits, h, w)  from fit_linear_reference
    coeffs_cube   : ndarray (poly_order+1, h, w)  from fit_nonlinearity_poly
    s_ref         : float  — normalisation reference used during fitting
    illum_threshold : float
    detector      : str
    sat_limit     : float or None  — draws a vertical line if given
    title         : str  — optional suffix for the plot title

    Returns
    -------
    residual_cube : ndarray (n_dits, h, w)
        S_corrected / S_ideal.  NaN for unilluminated pixels.
    """
    if isinstance(science_ramps, dict):
        science_ramps = science_ramps[detector]
    science_ramps = np.asarray(science_ramps)

    ramps = science_ramps.astype(float)             # (n_dits, h, w)
    n_dits, h, w = ramps.shape

    illum_mask = np.median(ramps, axis=0) > illum_threshold

    # --- Apply correction ---
    corrected = apply_nonlinearity_correction(ramps, coeffs_cube, s_ref)

    # --- Ratios ---
    with np.errstate(divide='ignore', invalid='ignore'):
        before = np.where(illum_mask[np.newaxis] & (ideal_cube != 0),
                          ramps     / ideal_cube, np.nan)
        after  = np.where(illum_mask[np.newaxis] & (ideal_cube != 0),
                          corrected / ideal_cube, np.nan)

    # --- Per-DIT statistics ---
    before_px     = before[:, illum_mask]            # (n_dits, n_pixels)
    after_px      = after[:, illum_mask]
    median_signal = np.nanmedian(ramps[:, illum_mask], axis=1)
    median_before = np.nanmedian(before_px, axis=1)
    median_after  = np.nanmedian(after_px,  axis=1)
    std_before    = np.nanstd(before_px, axis=1)
    std_after     = np.nanstd(after_px,  axis=1)

    print("Validation summary (per DIT, illuminated pixels):")
    print(f"{'Signal':>10}  {'Before median':>14}  {'Before std':>11}  "
          f"{'After median':>13}  {'After std':>10}")
    for i in range(n_dits):
        print(f"{median_signal[i]:10.0f}  {median_before[i]:14.4f}  "
              f"{std_before[i]:11.4f}  {median_after[i]:13.4f}  {std_after[i]:10.4f}")

    # --- Scatter data ---
    signal_flat = ramps[:, illum_mask].ravel()
    before_flat = before_px.ravel()
    after_flat  = after_px.ravel()
    valid = np.isfinite(before_flat) & np.isfinite(after_flat) & (signal_flat > 0)
    rng = np.random.default_rng(0)
    idx = rng.choice(np.sum(valid), size=min(40_000, int(np.sum(valid))), replace=False)

    # Shared y-axis range clipped to 0.5–99.5 percentile of both panels
    all_vals = np.concatenate([before_flat[valid], after_flat[valid]])
    ylo, yhi = np.nanpercentile(all_vals, [0.5, 99.5])
    vline_kw = dict(color='orange', linestyle='--', alpha=0.7)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Linearity correction validation'
                 f'{" — " + title if title else ""}', fontsize=12)

    # Left — before
    axes[0].scatter(signal_flat[valid][idx], before_flat[valid][idx],
                    s=1, alpha=0.15, color='steelblue')
    axes[0].plot(median_signal, median_before, 'o-', color='red',
                 markersize=5, linewidth=1.5, label='Median per DIT')
    axes[0].axhline(1.0, color='gray', linestyle=':', linewidth=1)
    if sat_limit is not None:
        axes[0].axvline(sat_limit, label=f'sat_limit={sat_limit:,}', **vline_kw)
    axes[0].set_xlabel('S_obs  [ADU]')
    axes[0].set_ylabel('S_obs / S_ideal')
    axes[0].set_title('Before correction')
    axes[0].set_ylim(ylo, yhi)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Centre — after
    axes[1].scatter(signal_flat[valid][idx], after_flat[valid][idx],
                    s=1, alpha=0.15, color='mediumseagreen')
    axes[1].plot(median_signal, median_after, 'o-', color='darkgreen',
                 markersize=5, linewidth=1.5, label='Median per DIT')
    axes[1].axhline(1.0, color='gray', linestyle=':', linewidth=1)
    if sat_limit is not None:
        axes[1].axvline(sat_limit, label=f'sat_limit={sat_limit:,}', **vline_kw)
    axes[1].set_xlabel('S_obs  [ADU]')
    axes[1].set_ylabel('S_corrected / S_ideal')
    axes[1].set_title('After correction')
    axes[1].set_ylim(ylo, yhi)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Right — residual std per DIT
    axes[2].plot(median_signal, std_before * 100, 'o-', color='steelblue',
                 markersize=5, linewidth=1.5, label='Before')
    axes[2].plot(median_signal, std_after  * 100, 's-', color='darkgreen',
                 markersize=5, linewidth=1.5, label='After')
    if sat_limit is not None:
        axes[2].axvline(sat_limit, **vline_kw)
    axes[2].set_xlabel('Median signal per DIT  [ADU]')
    axes[2].set_ylabel('Std of ratio  [%]')
    axes[2].set_title('Residual scatter per DIT')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return after   # S_corrected / S_ideal  (n_dits, h, w)


def make_saturation_mask(science_ramps, ideal_cube, threshold_pct=5.0,
                          illum_threshold=3000, detector='DET1.DATA', plot=True):
    """
    Build a per-pixel saturation / validity mask.

    A pixel-DIT sample is flagged as INVALID (saturated) when the raw
    deviation from linearity exceeds a threshold:

        |S_obs / S_ideal  -  1|  >  threshold_pct / 100

    This identifies the signal level at which the non-linearity is so large
    that the correction polynomial can no longer be trusted.  Samples beyond
    this point are masked out and should not be used in science reductions.

    Parameters
    ----------
    science_ramps   : dict or ndarray (n_dits, h, w)
    ideal_cube      : ndarray (n_dits, h, w)  from fit_linear_reference
    threshold_pct   : float
        Deviation threshold in percent (default 5 %).  Samples with
        |S_obs/S_ideal - 1| > threshold_pct/100 are flagged.
    illum_threshold : float
        Pixels with median signal below this are treated as dark (always valid).
    detector        : str
    plot            : bool
        If True, show maps of the per-pixel saturation signal and valid fraction.

    Returns
    -------
    valid_cube      : ndarray bool (n_dits, h, w)
        True  = sample is in the linear regime, safe to use.
        False = sample is saturated / correction unreliable.
        Unilluminated pixels are always True (not flagged).
    sat_signal_map  : ndarray (h, w)
        Signal [ADU] at which each pixel first exceeds the threshold.
        NaN for pixels that never saturate within the ramp.
    valid_frac_map  : ndarray (h, w)
        Fraction of DITs that are valid per pixel (0–1).
        Useful for identifying persistently hot or poorly illuminated pixels.
    """
    if isinstance(science_ramps, dict):
        science_ramps = science_ramps[detector]
    science_ramps = np.asarray(science_ramps)

    ramps = science_ramps.astype(float)             # (n_dits, h, w)
    n_dits, h, w = ramps.shape

    illum_mask = np.median(ramps, axis=0) > illum_threshold   # (h, w)

    # --- Per-sample deviation |ratio - 1| ---
    with np.errstate(divide='ignore', invalid='ignore'):
        deviation = np.where(
            illum_mask[np.newaxis] & (ideal_cube != 0),
            np.abs(ramps / ideal_cube - 1.0),
            0.0          # dark / unilluminated → treat as zero deviation
        )

    # --- Valid mask: deviation < threshold ---
    valid_cube = deviation < (threshold_pct / 100.0)          # (n_dits, h, w)
    # Unilluminated pixels are always valid
    valid_cube[:, ~illum_mask] = True

    # --- Per-pixel saturation signal: signal at first invalid DIT ---
    # Sort ramps by DIT (already sorted if fit_linear_reference was called,
    # but we sort defensively here too)
    sat_signal_map = np.full((h, w), np.nan, dtype=float)

    # For each pixel find the first DIT where it goes invalid
    # (deviation array is ordered the same as ramps)
    first_bad = (~valid_cube).argmax(axis=0)           # (h, w) — index of first bad DIT
    has_bad   = (~valid_cube).any(axis=0)              # (h, w) — True if any DIT is bad

    # Signal at that DIT for each pixel
    row_idx, col_idx = np.where(has_bad & illum_mask)
    for r, c in zip(row_idx, col_idx):
        k = first_bad[r, c]
        sat_signal_map[r, c] = ramps[k, r, c]

    # --- Valid fraction per pixel ---
    valid_frac_map = valid_cube.sum(axis=0) / n_dits   # (h, w)

    # --- Summary ---
    n_illum = illum_mask.sum()
    n_ever_bad = (has_bad & illum_mask).sum()
    print(f"Threshold          : {threshold_pct:.1f} %")
    print(f"Illuminated pixels : {n_illum:,}")
    print(f"Pixels with ≥1 saturated DIT : {n_ever_bad:,}  "
          f"({100 * n_ever_bad / max(n_illum, 1):.1f} %)")
    finite_sat = sat_signal_map[np.isfinite(sat_signal_map)]
    if len(finite_sat) > 0:
        print(f"Saturation signal  : median {np.median(finite_sat):,.0f}  "
              f"min {np.min(finite_sat):,.0f}  max {np.max(finite_sat):,.0f}  ADU")

    # --- Optional plot ---
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f'Saturation mask  (threshold = {threshold_pct:.1f} %)', fontsize=11)

        # Saturation signal map
        im0 = axes[0].imshow(sat_signal_map, origin='lower', cmap='plasma',
                             vmin=np.nanpercentile(sat_signal_map, 2),
                             vmax=np.nanpercentile(sat_signal_map, 98))
        plt.colorbar(im0, ax=axes[0], label='Saturation signal [ADU]')
        axes[0].set_title('Per-pixel saturation signal')
        axes[0].set_xlabel('x  [pixel]')
        axes[0].set_ylabel('y  [pixel]')

        # Valid fraction map
        im1 = axes[1].imshow(valid_frac_map, origin='lower', cmap='RdYlGn',
                             vmin=0, vmax=1)
        plt.colorbar(im1, ax=axes[1], label='Fraction of valid DITs')
        axes[1].set_title('Valid fraction per pixel')
        axes[1].set_xlabel('x  [pixel]')
        axes[1].set_ylabel('y  [pixel]')

        plt.tight_layout()
        plt.show()

    return valid_cube, sat_signal_map, valid_frac_map


def write_calibration_fits(coeffs_cube, s_ref, output_path,
                            sat_signal_map=None, valid_frac_map=None,
                            header=None, poly_order=3,
                            detector='DET1.DATA', sat_limit=None,
                            threshold_pct=None):
    """
    Write the full linearity calibration to a FITS file.

    This is the deliverable that the pipeline reads at the start of each
    science reduction, before any other calibration step.

    Extensions
    ----------
    PRIMARY        — metadata header (no image data)
    COEFF_A{n}     — polynomial coefficient map for u^n term, shape (h, w)
                     Apply: S_corr = S_obs * polyval(coeffs[:, y, x], S_obs/S_REF)
    SAT_SIGNAL     — (optional) per-pixel saturation signal [ADU]
    VALID_FRAC     — (optional) fraction of calibration DITs that were valid

    Parameters
    ----------
    coeffs_cube    : ndarray (poly_order+1, h, w)  from fit_nonlinearity_poly
    s_ref          : float  — normalisation reference [ADU] (stored in header)
    output_path    : str    — path for the output FITS file
    sat_signal_map : ndarray (h, w) or None  from make_saturation_mask
    valid_frac_map : ndarray (h, w) or None  from make_saturation_mask
    header         : fits.Header or None  — base header (e.g. original science header)
    poly_order     : int
    detector       : str
    sat_limit      : float or None  — signal threshold used for linear fit
    threshold_pct  : float or None  — deviation threshold used for saturation mask
    """
    hdr = header.copy() if header is not None else fits.Header()

    # --- Pipeline-essential keywords ---
    hdr['HIERARCH LIN POLYDEG'] = (poly_order, 'Linearity polynomial degree')
    hdr['HIERARCH LIN S_REF']   = (float(s_ref),
                                    'Normalisation reference signal [ADU]')
    hdr['HIERARCH LIN CORRECT'] = ('S_obs * polyval(coeffs, S_obs/S_REF)',
                                    'How to apply this calibration')
    hdr['DETECTOR'] = (detector, 'Detector identifier')
    if sat_limit is not None:
        hdr['HIERARCH LIN SATLIM'] = (float(sat_limit),
                                       'Linear-regime threshold used in fit [ADU]')
    if threshold_pct is not None:
        hdr['HIERARCH LIN SATPCT'] = (float(threshold_pct),
                                       'Saturation deviation threshold [%]')
    hdr['COMMENT'] = ('Linearity calibration: apply before flat-field, '
                      'dark subtraction assumed done.')

    primary = fits.PrimaryHDU(header=hdr)
    hdul    = fits.HDUList([primary])

    # --- Coefficient extensions ---
    for k, degree in enumerate(range(poly_order, -1, -1)):
        img = fits.ImageHDU(data=coeffs_cube[k].astype(np.float32),
                            name=f'COEFF_A{degree}')
        img.header['DEGREE']             = (degree, f'u^{degree} coefficient')
        img.header['HIERARCH LIN S_REF'] = (float(s_ref),
                                             'Normalisation reference [ADU]')
        hdul.append(img)

    # --- Optional saturation extensions ---
    if sat_signal_map is not None:
        sat_hdu = fits.ImageHDU(data=sat_signal_map.astype(np.float32),
                                name='SAT_SIGNAL')
        sat_hdu.header['BUNIT']   = ('ADU', 'Signal at saturation onset')
        sat_hdu.header['COMMENT'] = ('NaN = pixel never saturated in calibration ramp')
        if threshold_pct is not None:
            sat_hdu.header['HIERARCH LIN SATPCT'] = (
                float(threshold_pct), 'Deviation threshold used [%]')
        hdul.append(sat_hdu)

    if valid_frac_map is not None:
        vf_hdu = fits.ImageHDU(data=valid_frac_map.astype(np.float32),
                               name='VALID_FRAC')
        vf_hdu.header['BUNIT']   = ('', 'Fraction of calibration DITs in linear regime')
        vf_hdu.header['COMMENT'] = ('1.0 = all DITs valid, 0.0 = always saturated')
        hdul.append(vf_hdu)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    hdul.writeto(output_path, overwrite=True)
    print(f"Calibration FITS written to: {output_path}")
    print(f"  Extensions: {[h.name for h in hdul]}")


def find_nonlinear_onset(science_ramps, ideal_cube, times,
                          sat_limit=50_000, n_sigma=5,
                          illum_threshold=3000, detector='DET1.DATA',
                          plot=True, title=''):
    """
    Find the DIT at which a detector setting leaves the linear regime.

    Method (sigma-clip on detector median):
      1. At each DIT, compute the median deviation across all illuminated pixels:
             dev(t) = median_pixels[ S_obs(t) / S_ideal(t) - 1 ]
      2. Use only the linear-regime DITs (signal < sat_limit) to estimate the
         noise floor:  σ = std( dev[linear DITs] )
      3. The onset DIT is the first DIT where  |dev(t)| > n_sigma * σ.

    This gives one number per (setting, detector) — the DIT and signal level
    where non-linearity becomes statistically detectable.

    Parameters
    ----------
    science_ramps   : dict or ndarray (n_dits, h, w)
    ideal_cube      : ndarray (n_dits, h, w)  from fit_linear_reference
    times           : array-like (n_dits,)  DIT values [s]
    sat_limit       : float
        Signal threshold that defines the "known linear" regime for σ estimation.
    n_sigma         : float
        Detection threshold in units of σ (default 5).
    illum_threshold : float
    detector        : str
    plot            : bool
        If True, show deviation vs DIT with σ bands and onset marked.
    title           : str

    Returns
    -------
    result : dict with keys
        'onset_dit'    : float or None — DIT [s] at onset (None if never detected)
        'onset_signal' : float or None — median detector signal at onset [ADU]
        'sigma_pct'    : float         — noise floor σ [%] in linear regime
        'dev_pct'      : ndarray       — median deviation [%] per DIT (sorted by DIT)
        'times_sorted' : ndarray       — DIT values sorted ascending
    """
    if isinstance(science_ramps, dict):
        science_ramps = science_ramps[detector]
    science_ramps = np.asarray(science_ramps)

    t = np.array(times, dtype=float)
    sort_idx = np.argsort(t)
    t        = t[sort_idx]
    ramps    = science_ramps[sort_idx].astype(float)   # (n_dits, h, w)
    ideal    = ideal_cube[sort_idx]
    n_dits   = len(t)

    illum_mask = np.median(ramps, axis=0) > illum_threshold

    # --- Per-DIT median deviation across all illuminated pixels ---
    # dev(t) = median_px[ S_obs/S_ideal - 1 ]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_px = np.where(
            illum_mask[np.newaxis] & (ideal != 0),
            ramps / ideal,
            np.nan
        )[:, illum_mask]                               # (n_dits, n_pixels)

    dev_per_dit = np.nanmedian(ratio_px - 1.0, axis=1) * 100   # (n_dits,) in %
    sig_per_dit = np.nanmedian(ramps[:, illum_mask], axis=1)    # (n_dits,) median signal

    # --- Linear-regime DITs: median signal < sat_limit ---
    lin_mask = sig_per_dit < sat_limit
    if lin_mask.sum() < 2:
        print("Warning: fewer than 2 DITs in linear regime — cannot estimate σ.")
        return {'onset_dit': None, 'onset_signal': None,
                'sigma_pct': np.nan, 'dev_pct': dev_per_dit, 'times_sorted': t}

    sigma_pct = float(np.std(dev_per_dit[lin_mask]))
    threshold = n_sigma * sigma_pct

    # --- Find onset: first DIT where |dev| > n_sigma * σ ---
    exceeded = np.abs(dev_per_dit) > threshold
    onset_idx = int(np.argmax(exceeded)) if exceeded.any() else None

    if onset_idx is not None and exceeded[onset_idx]:
        onset_dit    = float(t[onset_idx])
        onset_signal = float(sig_per_dit[onset_idx])
        print(f"Non-linear onset  : DIT = {onset_dit:.2f} s  "
              f"(median signal = {onset_signal:,.0f} ADU)  "
              f"[{n_sigma}σ, σ = {sigma_pct:.3f} %]")
    else:
        onset_dit    = None
        onset_signal = None
        print(f"No onset detected within ramp at {n_sigma}σ  "
              f"(σ = {sigma_pct:.3f} %,  threshold = {threshold:.3f} %)")

    if title:
        print(f"  Setting: {title}")

    # --- Plot ---
    if plot:
        fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(sig_per_dit, dev_per_dit, 'o-', color='steelblue',
                markersize=6, linewidth=1.5, label='Median deviation per DIT')
        ax.axhline(0,          color='gray',  linestyle=':', linewidth=1)
        ax.axhline(+threshold, color='red',   linestyle='--', linewidth=1,
                   label=f'+{n_sigma}σ = +{threshold:.2f} %')
        ax.axhline(-threshold, color='red',   linestyle='--', linewidth=1,
                   label=f'−{n_sigma}σ = −{threshold:.2f} %')
        ax.axhspan(-threshold, +threshold, alpha=0.08, color='green',
                   label=f'Linear regime (σ = {sigma_pct:.3f} %)')

        if onset_signal is not None:
            ax.axvline(onset_signal, color='darkorange', linestyle='-',
                       linewidth=1.5, label=f'Onset: {onset_signal:,.0f} ADU')
            ax.scatter([onset_signal], [dev_per_dit[onset_idx]],
                       color='darkorange', s=80, zorder=5)

        if sat_limit is not None:
            ax.axvline(sat_limit, color='gray', linestyle=':', linewidth=1,
                       alpha=0.6, label=f'sat_limit = {sat_limit:,}')

        # Annotate each point with its DIT value
        for i in range(n_dits):
            ax.annotate(f'{t[i]:.1f}s', (sig_per_dit[i], dev_per_dit[i]),
                        textcoords='offset points', xytext=(4, 4), fontsize=7,
                        color='steelblue')

        ax.set_xlabel('Median detector signal  [ADU]')
        ax.set_ylabel('Median deviation  (S_obs/S_ideal − 1)  [%]')
        ax.set_title(f'Non-linearity onset{" — " + title if title else ""}')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return {
        'onset_dit':    onset_dit,
        'onset_signal': onset_signal,
        'sigma_pct':    sigma_pct,
        'dev_pct':      dev_per_dit,
        'times_sorted': t,
    }


def summarise_nonlinear_onset(datasets, ideal_cubes, n_sigma=5,
                               sat_limit=50_000, illum_threshold=3000):
    """
    Run find_nonlinear_onset for all settings × all detectors and print a table.

    Parameters
    ----------
    datasets   : dict {name: (images_dict, times)}
    ideal_cubes: dict {name: {detector: ideal_cube}}
                 Pre-computed by fit_linear_reference for each setting × detector.
    n_sigma    : float
    sat_limit  : float
    illum_threshold : float

    Returns
    -------
    results : dict {name: {detector: result_dict}}
    """
    DETECTORS = ['DET1.DATA', 'DET2.DATA', 'DET3.DATA', 'DET4.DATA']
    results = {}

    print(f"\n{'Setting':<12} {'Detector':<12} {'Onset DIT [s]':>14} "
          f"{'Onset signal [ADU]':>20} {'σ [%]':>8}")
    print('-' * 70)

    for name, (imgs, times) in datasets.items():
        results[name] = {}
        for det in DETECTORS:
            r = find_nonlinear_onset(
                imgs, ideal_cubes[name][det], times,
                sat_limit=sat_limit, n_sigma=n_sigma,
                illum_threshold=illum_threshold,
                detector=det, plot=False)
            results[name][det] = r

            onset_dit_str = f"{r['onset_dit']:.2f}" if r['onset_dit'] else '—'
            onset_sig_str = f"{r['onset_signal']:,.0f}" if r['onset_signal'] else '—'
            print(f"{name:<12} {det:<12} {onset_dit_str:>14} "
                  f"{onset_sig_str:>20} {r['sigma_pct']:>8.3f}")

    return results


# Usage

# Usage
'''ideal_cube, slopes, intercepts = LMS_lin.fit_linear_reference(
    images_21_750, times_21_750,
    sat_limit=50_000,
    detector='DET1.DATA'
)

# Ratio (non-linearity) at every DIT and pixel:
ratio_cube = images_21_750['DET1.DATA'] / ideal_cube  # (n_dits, h, w)
'''
'''
import LMS_lin

# Step 1 — linear reference
ideal_cube, slopes, intercepts = LMS_lin.fit_linear_reference(
    images_21_750, times_21_750, sat_limit=50_000, detector='DET1.DATA'
)

# Step 2 — deviation
ratio_cube, diff_cube = LMS_lin.compute_deviation(
    images_21_750, ideal_cube,
    detector='DET1.DATA',
    sat_limit=50_000,      # draws vertical line on plot
    plot=True
)
'''


'''# Step 1
ideal_cube, slopes, intercepts = LMS_lin.fit_linear_reference(
    images_21_750, times_21_750, sat_limit=50_000, detector='DET1.DATA')

# Step 2
ratio_cube, diff_cube = LMS_lin.compute_deviation(
    images_21_750, ideal_cube, detector='DET1.DATA', plot=False)

# Step 3 — fit the correction polynomial
coeffs_cube, s_ref = LMS_lin.fit_nonlinearity_poly(
    images_21_750, ratio_cube,
    s_ref=50_000,           # fix this to well capacity for portability
    poly_order=3,
    detector='DET1.DATA',
    output_path='coeffs/21_750/nonlin_DET1.fits')

# Step 4 — apply
corrected = LMS_lin.apply_nonlinearity_correction(
    images_21_750, coeffs_cube, s_ref, detector='DET1.DATA')

    
    residual_cube = LMS_lin.validate_correction(
    images_21_750, ideal_cube, coeffs_cube, s_ref,
    detector='DET1.DATA',
    sat_limit=50_000,
    title='Order 21 / BBT 750'
)


# Steps 1–3 as before...

# Step 5a — saturation mask
valid_cube, sat_signal_map, valid_frac_map = LMS_lin.make_saturation_mask(
    images_21_750, ideal_cube, threshold_pct=5.0, detector='DET1.DATA')

# Step 5b — write calibration file
LMS_lin.write_calibration_fits(
    coeffs_cube, s_ref,
    output_path='calib/nonlin_21_750_DET1.fits',
    sat_signal_map=sat_signal_map,
    valid_frac_map=valid_frac_map,
    header=orig_header,
    sat_limit=50_000,
    threshold_pct=5.0,
    detector='DET1.DATA')

'''