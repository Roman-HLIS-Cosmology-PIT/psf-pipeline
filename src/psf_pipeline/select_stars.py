import time
import requests
import numpy as np
from astroquery.gaia import Gaia

from detect import get_cat


def query_gaia_stars_cone(
    ra_deg, dec_deg, radius_arcmin,
    *, g_max=21.0, ruwe_max=1.4, g_snr_min=10.0,
    top=200000,          # <- SERVER-SIDE cap
    async_job=True,
    retries=3
):
    radius_deg = radius_arcmin / 60.0

    where = [
        f"1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_deg}))",
        f"phot_g_mean_mag < {g_max}",
        f"phot_g_mean_flux_over_error > {g_snr_min}",
        f"ruwe < {ruwe_max}",
    ]

    query = f"""
    SELECT TOP {int(top)}
      source_id, ra, dec, ref_epoch,
      phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
      phot_g_mean_flux_over_error, ruwe,
      pmra, pmdec, parallax
    FROM gaiadr3.gaia_source
    WHERE {" AND ".join(where)}
    """

    last_err = None
    for k in range(retries + 1):
        try:
            job = Gaia.launch_job_async(query) if async_job else Gaia.launch_job(query)
            return job.get_results()
        except requests.exceptions.HTTPError as e:
            last_err = e
            # quick backoff; often succeeds on the next attempt if it was transient
            time.sleep(1.5 * (k + 1))

    raise last_err


def _mad(x):
    """Median absolute deviation scaled to ~1-sigma for a normal distribution."""
    x = np.asarray(x)
    med = np.nanmedian(x)
    return 1.4826 * np.nanmedian(np.abs(x - med))

def get_stars(
    mag,
    size,
    flags=None,
    flag_max=None,
    mag_range=None,          # (min, max) to consider for fitting/selection
    size_range=None,         # (min, max) sanity range
    nbins=30,
    min_per_bin=50,
    q_locus=0.04,            # quantile used to trace the stellar locus (lower envelope)
    poly_order=3,
    use_log_size=True,       # usually helps linearize the locus
    nsig_low=3.0,            # allow below-locus outliers (cosmics, bad fits) if too large
    nsig_high=3.0,           # main separator vs galaxies (galaxies have positive residuals)
    max_abs_low_resid=None,  # optional hard cap on how far below locus you allow (in size units)
    refine=True,             # 2-pass refinement using preliminary star set
    return_model=False,
):
    """
    Select stars using a stellar-locus band in size–magnitude space.

    Parameters
    ----------
    mag : array-like
        Magnitude (e.g., MAG_AUTO).
    size : array-like
        Size metric (e.g., FLUX_RADIUS, FWHM_IMAGE, or PSF-like size proxy).
    flags : array-like, optional
        Quality flags (e.g., SExtractor FLAGS).
    flag_max : int, optional
        Keep objects with flags <= flag_max.
    mag_range, size_range : tuple, optional
        Ranges to restrict fitting/selection.
    nbins : int
        Number of magnitude bins for locus estimation.
    min_per_bin : int
        Minimum objects per bin to use that bin for the locus fit.
    q_locus : float
        Quantile of the size distribution per mag bin used to trace the locus.
        Typical: 0.1–0.3. Lower if galaxy contamination is heavy.
    poly_order : int
        Polynomial order for locus fit.
    use_log_size : bool
        Fit/select in log10(size) space (recommended).
    nsig_low, nsig_high : float
        Band width in robust sigma (MAD) around the locus.
        Galaxies mostly sit at positive residuals.
    max_abs_low_resid : float, optional
        If set, clips how far below the locus you allow (in *linear size* units).
    refine : bool
        If True, refit locus using a preliminary star selection.
    return_model : bool
        If True, return (mask, model_dict). Else just mask.

    Returns
    -------
    mask : ndarray(bool)
        True for objects classified as stars.
    model_dict : dict (optional)
        Contains polynomial coeffs, sigma, and useful diagnostics.
    """
    mag = np.asarray(mag)
    size = np.asarray(size)

    n = mag.size
    if size.size != n:
        raise ValueError("mag and size must have the same length")

    # --- base valid mask
    valid = np.isfinite(mag) & np.isfinite(size)
    if flags is not None and flag_max is not None:
        flags = np.asarray(flags)
        valid &= np.isfinite(flags) & (flags <= flag_max)

    if mag_range is not None:
        valid &= (mag >= mag_range[0]) & (mag <= mag_range[1])

    if size_range is not None:
        valid &= (size >= size_range[0]) & (size <= size_range[1])

    # Need positive sizes for log-space
    if use_log_size:
        valid &= (size > 0)

    if valid.sum() < max(200, poly_order + 2):
        mask = np.zeros(n, dtype=bool)
        return (mask, {"status": "too_few_valid"}) if return_model else mask

    # Work vectors
    m = mag[valid]
    s = size[valid]
    y = np.log10(s) if use_log_size else s

    # --- helper: fit locus from binned quantiles
    def fit_locus(mm, yy):
        # magnitude bins
        lo, hi = np.nanpercentile(mm, [1, 99])
        edges = np.linspace(lo, hi, nbins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

        x_pts = []
        y_pts = []
        n_pts = []

        for i in range(nbins):
            inb = (mm >= edges[i]) & (mm < edges[i + 1])
            if inb.sum() < min_per_bin:
                continue

            # lower-envelope tracer: quantile in each bin
            yq = np.nanquantile(yy[inb], q_locus)
            x_pts.append(centers[i])
            y_pts.append(yq)
            n_pts.append(inb.sum())

        x_pts = np.array(x_pts)
        y_pts = np.array(y_pts)
        n_pts = np.array(n_pts)

        if x_pts.size < max(poly_order + 2, 5):
            # fallback: constant locus at global lower quantile
            const = np.nanquantile(yy, q_locus)
            coeff = np.array([const])  # poly0
            return coeff, x_pts, y_pts, n_pts, "fallback_constant"

        # polynomial fit (weighted by sqrt(N) to favor well-populated bins)
        w = np.sqrt(n_pts)
        coeff = np.polyfit(x_pts, y_pts, deg=poly_order, w=w)
        return coeff, x_pts, y_pts, n_pts, "ok"

    # --- pass 1 locus
    coeff1, x1, y1, n1, status1 = fit_locus(m, y)
    yhat1 = np.polyval(coeff1, m)
    resid1 = y - yhat1

    # robust sigma: use the "lower half" to focus on stars (galaxies inflate +resid tail)
    cut = np.nanpercentile(resid1, 50)
    sig1 = _mad(resid1[resid1 <= cut])
    if not np.isfinite(sig1) or sig1 <= 0:
        sig1 = _mad(resid1)
    if not np.isfinite(sig1) or sig1 <= 0:
        sig1 = np.nanstd(resid1)

    # preliminary star band
    prelim = (resid1 >= -nsig_low * sig1) & (resid1 <= nsig_high * sig1)

    # Optional: hard-cap too-far-below points in linear-size units
    if max_abs_low_resid is not None:
        if use_log_size:
            # convert (y - yhat) to multiplicative factor in size
            # below-locus by dlog => size_ratio = 10**dlog
            size_ratio = 10.0 ** (resid1)
            # resid too negative means ratio too small: enforce s >= s_locus - cap
            s_locus = 10.0 ** yhat1
            prelim &= (s >= (s_locus - max_abs_low_resid))
        else:
            prelim &= (resid1 >= -max_abs_low_resid)

    # --- refinement: refit locus on prelim stars, recompute sigma and final mask
    if refine and prelim.sum() >= max(200, poly_order + 2):
        coeff2, x2, y2, n2, status2 = fit_locus(m[prelim], y[prelim])
        yhat2 = np.polyval(coeff2, m)
        resid2 = y - yhat2

        cut2 = np.nanpercentile(resid2, 60)  # slightly above median to keep enough stars
        sig2 = _mad(resid2[resid2 <= cut2])
        if not np.isfinite(sig2) or sig2 <= 0:
            sig2 = _mad(resid2)
        if not np.isfinite(sig2) or sig2 <= 0:
            sig2 = np.nanstd(resid2)

        final = (resid2 >= -nsig_low * sig2) & (resid2 <= nsig_high * sig2)

        if max_abs_low_resid is not None:
            if use_log_size:
                s_locus = 10.0 ** yhat2
                final &= (s >= (s_locus - max_abs_low_resid))
            else:
                final &= (resid2 >= -max_abs_low_resid)

        mask_valid = final
        model = {
            "status": "ok_refined",
            "coeff": coeff2,
            "sigma": float(sig2),
            "use_log_size": bool(use_log_size),
            "q_locus": float(q_locus),
            "poly_order": int(poly_order),
            "nbins": int(nbins),
            "min_per_bin": int(min_per_bin),
        }
    else:
        mask_valid = prelim
        model = {
            "status": f"ok_singlepass_{status1}",
            "coeff": coeff1,
            "sigma": float(sig1),
            "use_log_size": bool(use_log_size),
            "q_locus": float(q_locus),
            "poly_order": int(poly_order),
            "nbins": int(nbins),
            "min_per_bin": int(min_per_bin),
        }

    # map back to full length
    mask = np.zeros(n, dtype=bool)
    mask[np.where(valid)[0]] = mask_valid

    return (mask, model) if return_model else mask

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    out, seg = get_cat('', 'detect_config.yaml')
    out = out[np.where(out['flux_radius']>0)]
    
    star_mask = get_stars(30-2.5*np.log10(out['flux']), out['flux_radius'])

    plt.scatter(out['flux_radius'], 30-2.5*np.log10(out['flux']), alpha=0.3)
    plt.scatter(out['flux_radius'][star_mask], 30-2.5*np.log10(out['flux'][star_mask]), alpha=0.3, c='g')

    plt.xlim(0, 6)
    plt.show()
    plt.close()

    ra0  = 9.5012560332761 # deg
    dec0 = -43.829285353832 # deg
    radius_arcmin = 5  # arcmin

    gaia_tab = query_gaia_stars_cone(
        ra0, dec0, radius_arcmin,
        g_max=20.5,        # keep Gaia G < 20.5
        ruwe_max=1.4,      # quality cut
        g_snr_min=10.0,    # quality cut
    )

    print(len(gaia_tab))
    print(gaia_tab.colnames)
    #print(gaia_tab[:5])
    print(gaia_tab[:])
    print (len(gaia_tab))


