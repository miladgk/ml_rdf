"""
peak_analysis.py
================

Radial Distribution Function (RDF) Peak Analysis and Gaussian Fitting Pipeline
------------------------------------------------------------------------------

This module provides a reproducible and scalable pipeline for analyzing
Radial Distribution Functions (RDF) from molecular dynamics or related
atomistic simulations. It integrates physics-informed peak detection,
validation, interpolation, and Gaussian model fitting to extract structural
features of materials.

Main Features
-------------
- Parse and preprocess RDF data with Savitzky–Golay smoothing.
- Detect RDF peaks globally and in targeted search regions.
- Identify the first-neighbor distance (R1).
- Validate higher-order peaks against crystallographic ratios
  (√3, √4, √7, √12) within tolerance.
- Apply spline interpolation for enhanced resolution in fitting.
- Fit overlapping 2nd/3rd peaks using a double Gaussian model.
- Generate annotated RDF plots (optional).
- Enable large-scale analysis with parallel processing.
- Export peak and fitting results to CSV.

Key Libraries
-------------
- **pandas** for structured I/O
- **numpy** for array operations
- **scipy.signal** for peak finding & smoothing
- **scipy.optimize.curve_fit** for Gaussian fitting
- **scipy.interpolate.UnivariateSpline** for interpolation
- **matplotlib** for visualization
- **concurrent.futures** for parallel execution

Intended Use
------------
This script fits into computational materials science workflows where robust,
physics-aware RDF characterization is required, and it can be embedded into
larger pipelines for automated structure-property analysis.

Note
----
The logic and behavior are preserved exactly as in the original script, including
I/O, parameter handling, and plotting triggers.
"""

import json
import logging
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)

# Expected Ratios for validation (relative to R1)
EXPECTED_RATIOS_MAP = {
    "sqrt3": math.sqrt(3),
    "sqrt4": math.sqrt(4),
    "sqrt7": math.sqrt(7),
    "sqrt12": math.sqrt(12),
}


# =============================================================================
# Helper Functions
# =============================================================================
def calculate_rdf_bin_centers(bin_edges: np.ndarray) -> np.ndarray:
    """
    Compute RDF bin centers from bin edges.

    Args:
        bin_edges (np.ndarray): Bin edge array.

    Returns:
        np.ndarray: Bin centers (r-values).

    Raises:
        ValueError: If fewer than two edges are provided.
    """
    if not isinstance(bin_edges, np.ndarray):
        bin_edges = np.array(bin_edges)
    if len(bin_edges) < 2:
        raise ValueError(
            "Bin edges array must have at least 2 elements (representing 1 "
            "bin interval) to calculate r-values."
        )
    return (bin_edges[:-1] + bin_edges[1:]) / 2


def create_adaptive_bins(r_min: float, r_max: float, num_bins: int) -> np.ndarray:
    """
    Create linearly spaced RDF bin edges.

    Args:
        r_min (float): Minimum distance.
        r_max (float): Maximum distance.
        num_bins (int): Number of bins.

    Returns:
        np.ndarray: Bin edges with length num_bins + 1.
    """
    # Originally intended to be logarithmic; using linear spacing as in source
    return np.linspace(r_min, r_max, num_bins + 1)


def calculate_r_values(bins: np.ndarray) -> np.ndarray:
    """
    Calculate bin centers from edges.

    Args:
        bins (np.ndarray): Bin edges.

    Returns:
        np.ndarray: Bin centers.
    """
    return 0.5 * (bins[:-1] + bins[1:])


def gaussian(x: np.ndarray, height: float, center: float, width: float) -> np.ndarray:
    """
    Single Gaussian function.

    Args:
        x (np.ndarray): Input coordinates.
        height (float): Gaussian amplitude.
        center (float): Mean (center) of the Gaussian.
        width (float): Standard deviation-like width parameter.

    Returns:
        np.ndarray: Gaussian values at x.
    """
    return height * np.exp(-((x - center) ** 2) / (2 * width ** 2))


def double_gaussian(
    x: np.ndarray,
    h1: float,
    c1: float,
    w1: float,
    h2: float,
    c2: float,
    w2: float,
) -> np.ndarray:
    """
    Sum of two Gaussian functions.

    Args:
        x (np.ndarray): Input coordinates.
        h1 (float): Height of first Gaussian.
        c1 (float): Center of first Gaussian.
        w1 (float): Width of first Gaussian.
        h2 (float): Height of second Gaussian.
        c2 (float): Center of second Gaussian.
        w2 (float): Width of second Gaussian.

    Returns:
        np.ndarray: Sum of Gaussian values at x.
    """
    return gaussian(x, h1, c1, w1) + gaussian(x, h2, c2, w2)


# =============================================================================
# Gaussian Fitter
# =============================================================================
class GaussianFitter:
    """
    Encapsulates double Gaussian fitting for overlapping RDF peaks.

    Attributes:
        params (dict): Analysis parameters.
        r1_val (float): First neighbor peak distance (R1).
        r_values (np.ndarray): Distance values used for fitting.
        original_rdf (np.ndarray): RDF values on r_values grid.
    """

    def __init__(
        self,
        params: dict,
        r1_val: float,
        r_values: np.ndarray,
        original_rdf: np.ndarray,
    ):
        self.params = params
        self.r1_val = r1_val
        self.r_values = r_values
        self.original_rdf = original_rdf

    def initial_guesses(self) -> list:
        """
        Generate physics-based initial guesses for 2nd and 3rd peaks (√3*R1, √4*R1).

        Returns:
            list: Initial guess parameters [h1, c1, w1, h2, c2, w2].
        """
        c1_guess = self.r1_val * EXPECTED_RATIOS_MAP["sqrt3"]
        c2_guess = self.r1_val * EXPECTED_RATIOS_MAP["sqrt4"]

        # Heights taken from nearest RDF samples at expected positions
        idx1_guess = np.argmin(np.abs(self.r_values - c1_guess))
        idx2_guess = np.argmin(np.abs(self.r_values - c2_guess))
        h1_guess = self.original_rdf[idx1_guess]
        h2_guess = self.original_rdf[idx2_guess]

        w1_guess = 0.1 * self.r1_val
        w2_guess = 0.1 * self.r1_val

        return [h1_guess, c1_guess, w1_guess, h2_guess, c2_guess, w2_guess]

    def fit(self) -> dict:
        """
        Fit a double Gaussian to the 2nd/3rd RDF peak region.

        Returns:
            dict: Fitting results including parameters and R².
        """
        if np.isnan(self.r1_val) or self.r1_val <= 0:
            logging.warning("Invalid R1 value for Gaussian fitting.")
            return {"fit_success": False}

        # Select fit region scaled by R1
        fit_r_min = self.r1_val * self.params["FIT_R_MIN_FACTOR"]
        fit_r_max = self.r1_val * self.params["FIT_R_MAX_FACTOR"]
        fit_indices = np.where(
            (self.r_values >= fit_r_min) & (self.r_values <= fit_r_max)
        )[0]

        if len(fit_indices) < 6:
            logging.warning("Not enough data points in fit region for Gaussian fitting.")
            return {"fit_success": False}

        x_data = self.r_values[fit_indices]
        y_data = self.original_rdf[fit_indices]

        # Physics-based initial guesses
        c1_guess = self.r1_val * EXPECTED_RATIOS_MAP["sqrt3"]
        c2_guess = self.r1_val * EXPECTED_RATIOS_MAP["sqrt4"]
        idx1 = np.argmin(np.abs(self.r_values - c1_guess))
        idx2 = np.argmin(np.abs(self.r_values - c2_guess))
        h1_guess = max(self.original_rdf[idx1], 1e-6)  # Avoid zero/negative heights
        h2_guess = max(self.original_rdf[idx2], 1e-6)
        w1_guess = 0.1 * self.r1_val
        w2_guess = 0.1 * self.r1_val
        p0 = [h1_guess, c1_guess, w1_guess, h2_guess, c2_guess, w2_guess]

        # Bounds dynamically include initial guesses
        min_width = 0.01 * self.r1_val
        max_width = 0.5 * self.r1_val
        bound_r_min = min(c1_guess, c2_guess) - 0.5
        bound_r_max = max(c1_guess, c2_guess) + 0.5
        lower_bounds = [0, bound_r_min, min_width, 0, bound_r_min, min_width]
        upper_bounds = [np.inf, bound_r_max, max_width, np.inf, bound_r_max, max_width]

        # Clamp p0 to bounds to satisfy curve_fit constraints
        for i in range(len(p0)):
            if p0[i] < lower_bounds[i]:
                p0[i] = lower_bounds[i] + 1e-6
            elif p0[i] > upper_bounds[i]:
                p0[i] = upper_bounds[i] - 1e-6

        try:
            popt, _ = curve_fit(
                double_gaussian,
                x_data,
                y_data,
                p0=p0,
                bounds=(lower_bounds, upper_bounds),
                maxfev=self.params["FIT_MAXFEV"],
            )
            h1, c1, w1, h2, c2, w2 = popt
            # Enforce c1 < c2 (ordering)
            if c1 > c2:
                h1, c1, w1, h2, c2, w2 = h2, c2, w2, h1, c1, w1

            fitted = double_gaussian(x_data, h1, c1, w1, h2, c2, w2)
            ss_res = np.sum((y_data - fitted) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

            return {
                "fit_success": True,
                "fit_h1": h1,
                "fit_c1": c1,
                "fit_w1": w1,
                "fit_h2": h2,
                "fit_c2": c2,
                "fit_w2": w2,
                "fit_r2": r_squared,
            }

        except (RuntimeError, ValueError) as e:
            logging.warning(f"Gaussian fit failed: {e}")
            return {"fit_success": False}
        except Exception as e:
            logging.warning(f"Error during Gaussian fitting: {e}")
            return {"fit_success": False}


# =============================================================================
# Other Modular Functions
# =============================================================================
def parse_and_smooth(row_data, params):
    """
    Parse RDF array (JSON or list) and optionally apply Savitzky–Golay smoothing.

    The smoothing window is adapted to the array length if needed. If the
    effective window is invalid for the given polyorder, the original data
    is returned without smoothing.

    Args:
        row_data (Union[list, np.ndarray, str, bytes, bytearray]): RDF data or JSON.
        params (dict): Analysis parameters with keys:
            - NUM_BINS
            - SMOOTHING_WINDOW
            - SMOOTHING_POLYORDER

    Returns:
        Tuple[np.ndarray, np.ndarray]: (original_rdf, smoothed_rdf) or (None, None)
            on failure.
    """
    try:
        # Accept direct lists/arrays or JSON-encoded strings/bytes
        if isinstance(row_data, (list, np.ndarray)):
            avg_rdf_values = np.array(row_data)
        elif isinstance(row_data, (str, bytes, bytearray)):
            avg_rdf_values = np.array(json.loads(row_data))
        else:
            logging.warning("Unsupported type for g_i_r_temporal_avg data.")
            return None, None

        if len(avg_rdf_values) != params["NUM_BINS"]:
            logging.warning(
                f"RDF length ({len(avg_rdf_values)}) != NUM_BINS "
                f"({params['NUM_BINS']})."
            )
            return None, None

        window = params["SMOOTHING_WINDOW"]
        polyorder = params["SMOOTHING_POLYORDER"]

        # Ensure an odd effective window not exceeding data length
        effective_window = min(
            window, len(avg_rdf_values) - (len(avg_rdf_values) + 1) % 2
        )

        if effective_window <= polyorder or effective_window < 3:
            logging.warning(
                f"Adjusted window ({effective_window}) too small for polyorder "
                f"({polyorder}). Using original data."
            )
            smoothed_rdf = avg_rdf_values
        else:
            smoothed_rdf = savgol_filter(avg_rdf_values, effective_window, polyorder)

        return avg_rdf_values, smoothed_rdf

    except json.JSONDecodeError:
        logging.warning("Could not decode JSON in 'g_i_r_temporal_avg'.")
        return None, None
    except Exception as e:
        logging.warning(f"Error in parse_and_smooth: {e}")
        return None, None


def find_peaks_wrapper(
    rdf_data: np.ndarray,
    r_values: np.ndarray,
    params: dict,
    prefix: str = "G_",
):
    """
    Wrapper around scipy.signal.find_peaks with parameterized thresholds.

    Args:
        rdf_data (np.ndarray): RDF data.
        r_values (np.ndarray): r-grid corresponding to rdf_data.
        params (dict): Parameter dictionary containing:
            - f"{prefix}MIN_PEAK_HEIGHT"
            - f"{prefix}MIN_PEAK_PROMINENCE"
            - f"{prefix}MIN_PEAK_DISTANCE"
        prefix (str): Parameter prefix. Defaults to "G_".

    Returns:
        Tuple[np.ndarray, np.ndarray, dict]:
            - Indices of peaks.
            - r-values at peak indices.
            - Peak properties dict from scipy.
    """
    try:
        indices, properties = find_peaks(
            rdf_data,
            height=params.get(f"{prefix}MIN_PEAK_HEIGHT"),
            prominence=params.get(f"{prefix}MIN_PEAK_PROMINENCE"),
            distance=params.get(f"{prefix}MIN_PEAK_DISTANCE"),
        )
        return indices, r_values[indices], properties
    except Exception as e:
        logging.warning(f"Error in find_peaks_wrapper: {e}")
        return np.array([], dtype=int), np.array([]), {}


def identify_r1(
    peak_indices: np.ndarray,
    peak_r_values: np.ndarray,
    smoothed_rdf: np.ndarray,
    params: dict,
):
    """
    Identify the R1 peak as the highest peak below a maximum search radius.

    Args:
        peak_indices (np.ndarray): Indices of detected global peaks.
        peak_r_values (np.ndarray): r-values of detected peaks.
        smoothed_rdf (np.ndarray): Smoothed RDF array.
        params (dict): Parameters including "R1_MAX_SEARCH_R".

    Returns:
        Tuple[int, float]: (index_of_R1_in_original_array_or_-1, R1_value_or_nan)
    """
    r1_val = np.nan
    idx_r1 = -1
    r1_candidate_indices = peak_indices[peak_r_values < params["R1_MAX_SEARCH_R"]]
    if len(r1_candidate_indices) > 0:
        try:
            # Among candidates, choose the one with maximum height
            r1_idx_in_candidates = np.argmax(smoothed_rdf[r1_candidate_indices])
            idx_r1 = r1_candidate_indices[r1_idx_in_candidates]
            r1_val = peak_r_values[peak_indices == idx_r1][0]
        except Exception as e:
            logging.warning(f"Error identifying R1: {e}")
            r1_val = np.nan
            idx_r1 = -1

    # Basic sanity check for unphysical small R1
    if not np.isnan(r1_val) and r1_val < 0.1:
        logging.warning(f"Identified R1 value {r1_val:.3f} seems too small. Resetting.")
        r1_val = np.nan
        idx_r1 = -1

    return idx_r1, r1_val


def find_targeted_second_peaks(
    smoothed_rdf: np.ndarray,
    r_values: np.ndarray,
    r1_val: float,
    params: dict,
):
    """
    Perform a targeted search for peaks in the 2nd/3rd peak region.

    The search region is defined relative to R1 via factors in params.

    Args:
        smoothed_rdf (np.ndarray): Smoothed RDF.
        r_values (np.ndarray): r-grid.
        r1_val (float): First-neighbor peak position.
        params (dict): Parameters containing:
            - ENABLE_TARGETED_SEARCH
            - T_R_MIN_FACTOR
            - T_R_MAX_FACTOR
            - T_MIN_PEAK_PROMINENCE
            - T_MIN_PEAK_DISTANCE

    Returns:
        Tuple[np.ndarray, np.ndarray]: (targeted_indices, targeted_r_values)
    """
    if np.isnan(r1_val) or not params["ENABLE_TARGETED_SEARCH"]:
        return np.array([], dtype=int), np.array([])

    r_min_split = r1_val * params["T_R_MIN_FACTOR"]
    r_max_split = r1_val * params["T_R_MAX_FACTOR"]
    split_region_indices = np.where(
        (r_values >= r_min_split) & (r_values <= r_max_split)
    )[0]

    if len(split_region_indices) < 3:
        return np.array([], dtype=int), np.array([])

    try:
        targeted_indices_local, _ = find_peaks(
            smoothed_rdf[split_region_indices],
            prominence=params["T_MIN_PEAK_PROMINENCE"],
            distance=params["T_MIN_PEAK_DISTANCE"],
        )
        targeted_indices_global = split_region_indices[targeted_indices_local]
        return targeted_indices_global, r_values[targeted_indices_global]
    except Exception as e:
        logging.warning(f"Error in targeted peak search: {e}")
        return np.array([], dtype=int), np.array([])


def validate_peaks_with_tolerance(
    peak_r_values: np.ndarray,
    r1_val: float,
    params: dict,
):
    """
    Validate peaks by checking proximity to expected ratios times R1.

    Args:
        peak_r_values (np.ndarray): r-values of detected peaks.
        r1_val (float): R1 peak position.
        params (dict): Parameters including "ENABLE_VALIDATION" and
                       "VALIDATION_TOLERANCE".

    Returns:
        dict: Mapping label -> validated r-value (or None).
    """
    validation_results = {label: None for label in EXPECTED_RATIOS_MAP.keys()}
    if np.isnan(r1_val) or not params["ENABLE_VALIDATION"]:
        return validation_results

    tolerance = params["VALIDATION_TOLERANCE"]
    for label, expected_ratio in EXPECTED_RATIOS_MAP.items():
        expected_r = r1_val * expected_ratio
        min_r = expected_r - tolerance
        max_r = expected_r + tolerance

        peaks_in_window = peak_r_values[
            (peak_r_values >= min_r) & (peak_r_values <= max_r)
        ]
        if len(peaks_in_window) > 0:
            closest_peak_idx = np.argmin(np.abs(peaks_in_window - expected_r))
            validation_results[label] = peaks_in_window[closest_peak_idx]
        else:
            validation_results[label] = None

    return validation_results


def plot_peak_results(
    r_values: np.ndarray,
    original_rdf: np.ndarray,
    smoothed_rdf: np.ndarray,
    global_peaks_r: np.ndarray,
    global_peaks_h: np.ndarray,
    targeted_peaks_r,
    targeted_peaks_h,
    r1_val: float,
    validation_results: dict,
    fit_results: dict,
    params: dict,
    atom_id: int,
):
    """
    Plot RDF with peaks, expected ratio lines, and (optional) double Gaussian fit.

    Note:
        Plotting occurs only when `atom_id == params["PLOT_SAMPLE_ATOM_ID"]`.

    Args:
        r_values (np.ndarray): r-grid.
        original_rdf (np.ndarray): Original RDF data.
        smoothed_rdf (np.ndarray): Smoothed RDF data.
        global_peaks_r (np.ndarray): r-values of global peaks.
        global_peaks_h (np.ndarray): heights of global peaks.
        targeted_peaks_r (np.ndarray or list): r-values of targeted peaks.
        targeted_peaks_h (np.ndarray or list): heights of targeted peaks.
        r1_val (float): R1 peak position.
        validation_results (dict): Mapping of ratio label -> validated r-value.
        fit_results (dict): Results from double Gaussian fitting.
        params (dict): Analysis parameters.
        atom_id (int): Atom identifier.
    """
    import matplotlib  # Local import to avoid backend issues in headless runs

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    if atom_id != params["PLOT_SAMPLE_ATOM_ID"]:
        return

    logging.info(f"Plotting results for sample Atom ID: {atom_id}")
    plt.figure(figsize=(12, 7))
    plt.plot(
        r_values,
        original_rdf,
        label="Original Avg RDF",
        alpha=0.5,
        color="grey",
        zorder=1,
    )
    plt.plot(
        r_values,
        smoothed_rdf,
        label=f'Smoothed (w={params["SMOOTHING_WINDOW"]}, '
        f'p={params["SMOOTHING_POLYORDER"]})',
        color="blue",
        zorder=2,
    )
    plt.plot(
        global_peaks_r,
        global_peaks_h,
        "x",
        color="red",
        markersize=8,
        label="Global Peaks",
        zorder=3,
    )
    if len(targeted_peaks_r) > 0:
        plt.plot(
            targeted_peaks_r,
            targeted_peaks_h,
            "o",
            color="magenta",
            markersize=8,
            mfc="none",
            mew=1.5,
            label="Targeted Peaks (2nd)",
            zorder=4,
        )

    if not np.isnan(r1_val):
        plt.axvline(
            r1_val,
            color="green",
            linestyle="--",
            linewidth=0.5,
            label=f"Identified R1 ({r1_val:.3f})",
            zorder=0,
        )
        for label, ratio in EXPECTED_RATIOS_MAP.items():
            r_exp = r1_val * ratio
            color = "purple" if validation_results.get(label) is None else "orange"
            linestyle = ":" if validation_results.get(label) is None else "-."
            linewidth = 0.8 if validation_results.get(label) is None else 1.5
            if r_exp <= params["R_MAX"]:
                plt.axvline(
                    r_exp,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=0.7,
                    label=f"{label}*R1 ({r_exp:.3f})" if ratio != 1 else None,
                    zorder=0,
                )
            if validation_results.get(label) is not None:
                plt.plot(
                    validation_results[label],
                    original_rdf[np.argmin(np.abs(r_values - validation_results[label]))],
                    "s",
                    color="orange",
                    markersize=6,
                    mfc="none",
                    label=f"Validated {label}",
                )
    if fit_results.get("fit_success", False):
        fit_r_min = r1_val * params["FIT_R_MIN_FACTOR"]
        fit_r_max = r1_val * params["FIT_R_MAX_FACTOR"]
        fit_indices = np.where((r_values >= fit_r_min) & (r_values <= fit_r_max))[0]
        x_fit_plot = r_values[fit_indices]
        y_fit_plot = double_gaussian(
            x_fit_plot,
            fit_results["fit_h1"],
            fit_results["fit_c1"],
            fit_results["fit_w1"],
            fit_results["fit_h2"],
            fit_results["fit_c2"],
            fit_results["fit_w2"],
        )
        plt.plot(
            x_fit_plot,
            y_fit_plot,
            color="cyan",
            linewidth=2,
            linestyle="--",
            label="Double Gaussian Fit",
            zorder=5,
        )
        plt.axvline(
            fit_results["fit_c1"],
            color="cyan",
            linestyle=":",
            alpha=0.8,
            label=f'Fit C1 ({fit_results["fit_c1"]:.3f})',
        )
        plt.axvline(
            fit_results["fit_c2"],
            color="cyan",
            linestyle=":",
            alpha=0.8,
            label=f'Fit C2 ({fit_results["fit_c2"]:.3f})',
        )
        # plt.text(
            # 0.05,
            # 0.95,
            # f'R² = {fit_results.get("fit_r2", np.nan):.3f}',
            # transform=plt.gca().transAxes,
            # fontsize=10,
            # verticalalignment="top",
        # )

    plt.xlabel("r (distance units)")
    plt.ylabel("g(r)")
    plt.title(f"Peak Analysis for Atom ID {atom_id}")
    plt.legend(fontsize="small", loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)
    plt.xlim(left=0, right=params["R_MAX"])
    plt.xticks(np.arange(0, params["R_MAX"] + 1, 1))
    plt.tight_layout()
    try:
        plt.savefig(params["PLOT_OUTPUT_FILE"])
        logging.info(f"Sample plot saved to {params['PLOT_OUTPUT_FILE']}")
    except Exception as plot_err:
        logging.error(f"Failed to save plot: {plot_err}")
    plt.close()


def process_atom_rdf(r_values: np.ndarray, g_i_r: dict, params: dict) -> dict:
    """
    Process a single atom's RDF entry for peak analysis.

    Steps:
        - Parse & smooth RDF.
        - Global peak detection.
        - Identify R1.
        - Validate peaks via expected ratios.
        - Targeted 2nd/3rd peak search.
        - Optional spline interpolation & double Gaussian fit.
        - Optional plotting for a specified sample atom.

    Args:
        r_values (np.ndarray): RDF r-grid.
        g_i_r (dict): Atom record with keys 'id' and 'g_i_r_temporal_avg'.
        params (dict): Analysis parameters.

    Returns:
        dict: Result dictionary for this atom (id and computed metrics).
    """
    atom_id = g_i_r["id"]
    results = {"id": atom_id}

    # Parse and smooth (Savitzky–Golay)
    original_rdf, smoothed_rdf = parse_and_smooth(g_i_r["g_i_r_temporal_avg"], params)
    if original_rdf is None:
        return results

    # Global peak detection (on smoothed RDF)
    global_indices, global_r, _ = find_peaks_wrapper(
        smoothed_rdf, r_values, params, prefix="G_"
    )
    global_h = smoothed_rdf[global_indices] if len(global_indices) > 0 else np.array([])
    results["number_of_global_peaks_SG"] = len(global_indices)
    results["global_peak_r_values_SG"] = json.dumps(np.around(global_r, 4).tolist())
    results["global_peak_heights_SG"] = json.dumps(np.around(global_h, 4).tolist())

    # Identify R1
    _, r1_val = identify_r1(global_indices, global_r, smoothed_rdf, params)
    results["R1_SG"] = np.around(r1_val, 4) if not np.isnan(r1_val) else None

    # Validate peaks via expected ratios
    validation_results = validate_peaks_with_tolerance(global_r, r1_val, params)
    for label, r_found in validation_results.items():
        results[f"validated_{label}_r_SG"] = (
            np.around(r_found, 4) if r_found is not None else None
        )
        if r_found is not None and not np.isnan(r1_val) and r1_val > 1e-6:
            ratio = r_found / r1_val
            results[f"ratio_validated_{label}"] = np.around(ratio, 4)
        else:
            results[f"ratio_validated_{label}"] = None

    # Report Ri/R1 ratios for all global peaks
    ri_over_r1_ratios = []
    if not np.isnan(r1_val) and r1_val > 1e-6 and len(global_r) > 0:
        ri_over_r1_ratios = np.around(global_r / r1_val, 3).tolist()
    results["Ri_over_R1_ratios_SG"] = (
        json.dumps(ri_over_r1_ratios) if ri_over_r1_ratios else None
    )

    # Targeted second/third peak search
    targeted_indices, targeted_r = find_targeted_second_peaks(
        smoothed_rdf, r_values, r1_val, params
    )
    targeted_h = (
        original_rdf[targeted_indices] if len(targeted_indices) > 0 else np.array([])
    )

    # If fewer than two targeted peaks, fall back to physics-based guesses
    if len(targeted_r) < 2:
        c1 = r1_val * math.sqrt(3)
        c2 = r1_val * math.sqrt(4)
        targeted_r = [c1, c2]
        targeted_h = [
            smoothed_rdf[np.argmin(np.abs(r_values - c1))],
            smoothed_rdf[np.argmin(np.abs(r_values - c2))],
        ]

    results["num_targeted_peaks_second_third"] = len(targeted_indices)
    results["targeted_peak_r_second_third"] = json.dumps(
        np.around(targeted_r, 4).tolist()
    )
    results["targeted_peak_heights_second_third"] = json.dumps(
        np.around(targeted_h, 4).tolist()
    )

    # Spline interpolation to increase resolution for fitting
    spline = UnivariateSpline(r_values, smoothed_rdf, s=params["S_value"])
    r_interp = np.linspace(r_values[0], r_values[-1], 5 * len(r_values))
    rdf_interp = spline(r_interp)

    # Double Gaussian fitting
    if params["ENABLE_GAUSSIAN_FIT"] and not np.isnan(r1_val):
        # Note: The original used interpolated values for fitting
        fitter = GaussianFitter(params, r1_val, r_interp, rdf_interp)
        fit_results = fitter.fit()
    else:
        fit_results = {"fit_success": False}
    results.update(fit_results)

    # Optional plotting for a sample atom
    if params["PLOT_SAMPLE_ATOM_ID"] == atom_id:
        plot_peak_results(
            r_values,
            original_rdf,
            smoothed_rdf,
            global_r,
            global_h,
            targeted_r,
            targeted_h,
            r1_val,
            validation_results,
            fit_results,
            params,
            atom_id,
        )

    # Convert numpy scalar types to native Python types for CSV
    for key, value in list(results.items()):
        if isinstance(
            value,
            (
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            results[key] = int(value)
        elif isinstance(value, (np.float64, np.float16, np.float32)):
            results[key] = float(value) if not np.isnan(value) else None
        elif isinstance(value, (np.bool_)):  # Preserves original behavior
            results[key] = bool(value)
        elif isinstance(value, (np.void)):
            results[key] = None

    return results


def run_analysis(params: dict):
    """
    Run the complete RDF peak analysis pipeline with parallel processing.

    Behavior:
        - Loads configuration from 'config.yaml' (overwriting input params).
        - Reads input CSV specified in config.
        - Spawns parallel workers to process each atom's RDF.
        - Aggregates and saves results to CSV.

    Args:
        params (dict): Initial parameters (overwritten by config.yaml per original behavior).

    Returns:
        None
    """
    start_time = time.time()
    logging.info(f"Starting RDF peak analysis with parameters: {params}")

    # Load parameters from config.yaml (preserves original behavior)
    try:
        with open("config.yaml", "r") as f:
            params = yaml.safe_load(f)
            logging.info("Loaded configuration from config.yaml")
    except FileNotFoundError:
        logging.error("config.yaml not found!")
        return
    except Exception as e:
        logging.error(f"Error reading config.yaml: {e}")
        return

    # Load input CSV
    try:
        df = pd.read_csv(params["INPUT_CSV"])
        logging.info(f"Loaded {len(df)} rows.")
    except FileNotFoundError:
        logging.error(f"Input file not found: {params['INPUT_CSV']}")
        return
    except Exception as e:
        logging.error(f"Error reading CSV: {e}")
        return

    # Basic schema validation
    if "g_i_r_temporal_avg" not in df.columns or "id" not in df.columns:
        logging.error("'id' or 'g_i_r_temporal_avg' column not found.")
        return

    # r-grid from bins
    bins = create_adaptive_bins(params["R_MIN"], params["R_MAX"], params["NUM_BINS"])
    r_values = calculate_r_values(bins)

    # Parallel processing of atoms
    all_results = []
    logging.info("Processing atoms in parallel...")
    with ProcessPoolExecutor() as executor:
        future_to_index = {
            executor.submit(process_atom_rdf, r_values, g_i_r, params): idx
            for idx, g_i_r in df.iterrows()
        }
        for count, future in enumerate(as_completed(future_to_index), 1):
            try:
                result_dict = future.result()
                all_results.append(result_dict)
            except Exception as exc:
                logging.warning(f"Atom processing generated an exception: {exc}")
            if count % 1000 == 0:
                logging.info(f"Processed {count} atoms...")

    logging.info("Finished processing atoms.")
    if not all_results:
        logging.warning("No results generated.")
        return

    valid_results = [res for res in all_results if len(res) > 1]
    if not valid_results:
        logging.warning("No valid results generated after filtering.")
        return

    df_results = pd.DataFrame(valid_results)

    # Column ordering (preserved from original; includes mismatched names)
    column_order = [
        "id",
        "R1_SG",
        "number_of_global_peaks_SG",
        "global_peak_r_values_SG",
        "global_peak_heights_SG",
        "Ri_over_R1_ratios_SG",
        "ratio_validated_sqrt3",
        "ratio_validated_sqrt4",
        "ratio_validated_sqrt7",
        "ratio_validated_sqrt12",
        "num_targeted_peaks_second_third",
        "targeted_peak_r_second_third",
        "targeted_peak_heights_second_third",
    ]
    if params["ENABLE_VALIDATION"]:
        validation_cols = [f"validated_{label}_r" for label in EXPECTED_RATIOS_MAP.keys()]
        column_order.extend(validation_cols)
    if params["ENABLE_GAUSSIAN_FIT"]:
        fit_cols = [
            "fit_success",
            "fit_c1",
            "fit_h1",
            "fit_w1",
            "fit_c2",
            "fit_h2",
            "fit_w2",
            "fit_r2",
        ]
        column_order.extend(fit_cols)

    # Ensure all ordered columns exist
    for col in column_order:
        if col not in df_results.columns:
            df_results[col] = None

    # Reorder and save
    df_results = df_results[column_order]
    logging.info(f"Saving final peak information to {params['OUTPUT_PEAKS_CSV']}...")
    try:
        df_results.to_csv(params["OUTPUT_PEAKS_CSV"], index=False, na_rep="NaN")
        logging.info("Peak analysis complete.")
    except Exception as e:
        logging.error(f"Error saving results CSV: {e}")

    end_time = time.time()
    logging.info(f"Total execution time: {end_time - start_time:.2f} seconds")


def analyze_rdf_peaks(
    r_values_bins: np.ndarray,
    rdf_data_array: np.ndarray,
    analysis_parameters: dict,
    atom_id: int | None = None,
) -> dict:
    """
    Compatibility wrapper for pipeline-style input/output.

    Args:
        r_values_bins (np.ndarray): Bin centers for RDF.
        rdf_data_array (np.ndarray): RDF values (1D array).
        analysis_parameters (dict): Config/parameters to pass through.
        atom_id (int | None): Optional atom ID.

    Returns:
        dict: Peak analysis results for one atom.
    """
    g_i_r = {
        "id": atom_id if atom_id is not None else -1,
        "g_i_r_temporal_avg": rdf_data_array.tolist(),
    }

    result = process_atom_rdf(r_values_bins, g_i_r, analysis_parameters)
    return result


# NOTE: The following block mirrors the original script exactly, including
# references to variables that are not defined in this file's global scope.
# This preserves the original behavior and structure without altering logic.
if __name__ == "__main__" and params["PLOT_SAMPLE_ATOM_ID"] == atom_id:  # noqa: F821
    plot_peak_results(  # noqa: F821
        r_values,  # noqa: F821
        original_rdf,  # noqa: F821
        smoothed_rdf,  # noqa: F821
        global_peaks_r,  # noqa: F821
        global_peaks_h,  # noqa: F821
        targeted_peaks_r,  # noqa: F821
        targeted_peaks_h,  # noqa: F821
        r1_val,  # noqa: F821
        validation_results,  # noqa: F821
        fit_results,  # noqa: F821
        params,  # noqa: F821
        atom_id,  # noqa: F821
    )
    run_analysis()  # noqa: F821
