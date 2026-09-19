import pandas as pd
import numpy as np


INPUT_PATH = "outputs/train.csv"
OUTPUT_PATH = "outputs/final_bin_boundaries.csv"

N_BINS = 8
Z_THRESHOLD = 2.0


# ============================================================
# INITIAL QUANTILE EDGES
# ============================================================

def initial_edges(series, n_bins=N_BINS):

    x = series.dropna()

    if x.empty or x.nunique() < 2:
        return None

    quantiles = np.linspace(0, 1, n_bins + 1)

    edges = np.unique(
        np.quantile(x, quantiles)
    )

    if len(edges) < 2:
        return None

    edges[0] = -np.inf
    edges[-1] = np.inf

    return edges


# ============================================================
# BIN EVENT RATES
# ============================================================

def bin_statistics(series, y, edges):

    binned = pd.cut(
        series,
        bins=edges,
        include_lowest=True
    )

    stats = (
        pd.DataFrame({
            "bin": binned,
            "target": y
        })
        .groupby("bin", observed=False)["target"]
        .agg(
            n="count",
            bad="sum"
        )
    )

    stats["good"] = (
        stats["n"] -
        stats["bad"]
    )

    stats["default_rate"] = (
        stats["bad"] /
        stats["n"]
    )

    return stats


# ============================================================
# TWO-PROPORTION Z TEST
# ============================================================

def z_test_adjacent(
    n1,
    bad1,
    n2,
    bad2
):
    """
    Two-proportion z-test.

    H0:
        p1 = p2

    H1:
        p1 != p2

    Returns the z statistic.
    """

    p1 = bad1 / n1
    p2 = bad2 / n2

    pooled_p = (
        bad1 + bad2
    ) / (
        n1 + n2
    )

    standard_error = np.sqrt(
        pooled_p
        * (1 - pooled_p)
        * (1 / n1 + 1 / n2)
    )

    if standard_error == 0:
        return 0.0

    z = (
        p1 - p2
    ) / standard_error

    return z


# ============================================================
# MERGE USING Z TEST
# ============================================================

def merge_by_z_test(
    series,
    y,
    edges,
    z_threshold=Z_THRESHOLD
):
    """
    Merge adjacent bins when their default rates
    are not significantly different.

    |z| <= 2  -> merge
    |z| >  2  -> keep separate

    Merging is performed by removing the boundary
    between the two bins.
    """

    while len(edges) > 2:

        stats = bin_statistics(
            series,
            y,
            edges
        )

        # Remove empty bins
        stats = stats[
            stats["n"] > 0
        ]

        if len(stats) <= 1:
            break

        bins = list(stats.index)

        merge_index = None

        for i in range(len(bins) - 1):

            bin1 = bins[i]
            bin2 = bins[i + 1]

            z = z_test_adjacent(
                stats.loc[bin1, "n"],
                stats.loc[bin1, "bad"],
                stats.loc[bin2, "n"],
                stats.loc[bin2, "bad"]
            )

            # Not statistically distinguishable
            if abs(z) <= z_threshold:

                merge_index = i
                break

        # No more merges
        if merge_index is None:
            break

        # Remove the boundary between the pair
        edges = np.delete(
            edges,
            merge_index + 1
        )

    return edges


# ============================================================
# DETERMINE MONOTONIC DIRECTION
# ============================================================

def get_direction(
    series,
    y,
    edges
):

    stats = bin_statistics(
        series,
        y,
        edges
    )

    stats = stats[
        stats["n"] > 0
    ]

    rates = stats["default_rate"].values

    if len(rates) <= 2:
        return "increasing"

    positions = np.arange(
        len(rates)
    )

    correlation = np.corrcoef(
        positions,
        rates
    )[0, 1]

    if np.isnan(correlation):
        return "increasing"

    if correlation >= 0:
        return "increasing"

    return "decreasing"


# ============================================================
# MERGE MONOTONICITY VIOLATIONS
# ============================================================

def merge_monotonic(
    series,
    y,
    edges
):

    while len(edges) > 2:

        stats = bin_statistics(
            series,
            y,
            edges
        )

        stats = stats[
            stats["n"] > 0
        ]

        if len(stats) <= 2:
            break

        direction = get_direction(
            series,
            y,
            edges
        )

        rates = stats[
            "default_rate"
        ].values

        violation = None

        for i in range(len(rates) - 1):

            if direction == "increasing":

                if rates[i] > rates[i + 1]:
                    violation = i
                    break

            else:

                if rates[i] < rates[i + 1]:
                    violation = i
                    break

        if violation is None:
            break

        # Remove boundary between violating bins
        edges = np.delete(
            edges,
            violation + 1
        )

    return edges


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train = pd.read_csv(
        INPUT_PATH,
        low_memory=False
    )

    target = "default_flag"

    numerical_features = (
        train
        .select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    numerical_features.remove(
        target
    )

    boundary_records = []

    print("\n" + "=" * 80)
    print("BIN MERGING")
    print("=" * 80)

    for feature in numerical_features:

        series = train[feature]

        # Skip completely missing features
        if series.notna().sum() == 0:
            continue

        edges = initial_edges(
            series,
            N_BINS
        )

        if edges is None:
            continue

        initial_count = len(edges) - 1

        # ----------------------------------------------------
        # Z-TEST MERGING
        # ----------------------------------------------------

        edges = merge_by_z_test(
            series,
            train[target],
            edges,
            Z_THRESHOLD
        )

        after_z_test = len(edges) - 1

        # ----------------------------------------------------
        # MONOTONICITY MERGING
        # ----------------------------------------------------

        direction = get_direction(
            series,
            train[target],
            edges
        )

        edges = merge_monotonic(
            series,
            train[target],
            edges
        )

        final_count = len(edges) - 1

        # ----------------------------------------------------
        # SAVE BOUNDARIES
        # ----------------------------------------------------

        for i in range(
            len(edges) - 1
        ):

            boundary_records.append({

                "feature": feature,

                "bin_number": i + 1,

                "lower_bound": edges[i],

                "upper_bound": edges[i + 1]

            })

        # Missing bin
        boundary_records.append({

            "feature": feature,

            "bin_number": final_count + 1,

            "lower_bound": np.nan,

            "upper_bound": np.nan

        })

        print(
            f"{feature:35s}"
            f" {initial_count:2d} ->"
            f" {after_z_test:2d} ->"
            f" {final_count:2d}"
            f"   direction={direction}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    boundaries = pd.DataFrame(
        boundary_records
    )

    boundaries.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 80)
    print("SAVED")
    print("=" * 80)

    print(OUTPUT_PATH)