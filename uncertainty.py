"""
uncertainty.py

Small shared utilities for reporting a value with an uncertainty at a
sensible number of significant figures, used by both
04_make_figures_6_7_8.py (Table I) and 06_make_figureA1.py (Table II)
so the two tables format uncertainties the same way.
"""
import math
import numpy as np


def round_to_uncertainty(value, uncertainty, sig_figs_unc=2):
    """Round `uncertainty` to sig_figs_unc significant figures, and round
    `value` to the same decimal place. Returns (value_rounded,
    uncertainty_rounded, n_decimals)."""
    if uncertainty <= 0 or not np.isfinite(uncertainty):
        return value, uncertainty, 6
    exponent = math.floor(math.log10(abs(uncertainty)))
    decimal_place = exponent - (sig_figs_unc - 1)
    factor = 10.0 ** (-decimal_place)
    unc_r = round(uncertainty * factor) / factor
    val_r = round(value * factor) / factor
    ndec = max(0, -decimal_place)
    return val_r, unc_r, ndec


def format_linear(value, uncertainty):
    """'0.822', '0.039' style, decimal places set by the uncertainty."""
    v, u, ndec = round_to_uncertainty(value, uncertainty)
    return f"{v:.{ndec}f}", f"{u:.{ndec}f}"


def format_scientific(value, uncertainty):
    """('a.aa', 'b.bb', exponent) with value/uncertainty as mantissas of a
    shared power of ten, uncertainty rounded to 2 significant figures."""
    if value == 0 or not np.isfinite(value):
        return "0", "0", 0
    exponent = math.floor(math.log10(abs(value)))
    mant_v = value / 10 ** exponent
    mant_u = uncertainty / 10 ** exponent
    mant_v_r, mant_u_r, ndec = round_to_uncertainty(mant_v, mant_u)
    return f"{mant_v_r:.{ndec}f}", f"{mant_u_r:.{ndec}f}", exponent
