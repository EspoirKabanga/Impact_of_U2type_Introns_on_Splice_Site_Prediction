import os
import re
import math
import logging
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from config import Config  # noqa: E402


logger = logging.getLogger("plot_sensitivity")


def setup_logging(level: str = "INFO") -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


DECILE_RE = re.compile(r"^decile_metrics_(?P<model>[^_]+)_(?P<category>[^_]+)_(?P<filetype>donor|acceptor)\.csv$")
SMOOTH_RE = re.compile(r"^smooth_trend_(?P<model>[^_]+)_(?P<category>[^_]+)_(?P<filetype>donor|acceptor)\.csv$")


def find_result_files(results_dir: Path) -> Tuple[List[Path], List[Path]]:
    decile_files: List[Path] = []
    smooth_files: List[Path] = []
    for p in results_dir.glob("*.csv"):
        name = p.name
        if DECILE_RE.match(name):
            decile_files.append(p)
        elif SMOOTH_RE.match(name):
            smooth_files.append(p)
    return decile_files, smooth_files


def parse_meta(path: Path, pattern: re.Pattern) -> Tuple[str, str, str]:
    m = pattern.match(path.name)
    if not m:
        raise ValueError(f"Unrecognized filename format: {path}")
    return m.group("model"), m.group("category"), m.group("filetype")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_bin_center(bin_str: str) -> float:
    try:
        low_s, high_s = str(bin_str).split("-")
        return (float(low_s) + float(high_s)) / 2.0
    except Exception:
        return float("nan")


def style_for(model: str, filetype: str) -> Dict[str, str]:
    color_map = {
        "IntSplicer": "#1f77b4",
        "SpliceRover": "#2ca02c",
        "SpliceFinder": "#ff7f0e",
        "DeepSplicer": "#d62728",
    }
    # All solid lines per request
    linestyle = "-"
    return {"color": color_map.get(model, "#333333"), "linestyle": linestyle}


def _compute_ticks(values: List[float]) -> Tuple[float, List[float]]:
    clean_vals = [v for v in values if not pd.isna(v)]
    if not clean_vals:
        return 0.0, [0.0, 0.5, 1.0]
    vmin = min(clean_vals)
    # floor to nearest 0.05 step below or equal to vmin
    floor = math.floor(vmin * 20.0) / 20.0
    # ensure floor not equal to 1.0
    floor = min(floor, 0.95)
    mid = (floor + 1.0) / 2.0
    ticks = [round(floor, 2), round(mid, 2), 1.00]
    return floor, ticks


def plot_deciles(decile_files: List[Path], output_dir: Path) -> None:
    # Group by category then filetype
    by_category_ft: Dict[str, Dict[str, List[Tuple[str, pd.DataFrame]]]] = {}
    for f in decile_files:
        model, category, filetype = parse_meta(f, DECILE_RE)
        df = pd.read_csv(f)
        df["bin_center"] = df["Length_Bin"].apply(to_bin_center)
        by_category_ft.setdefault(category, {}).setdefault(filetype, []).append((model, df))

    for category, by_ft in by_category_ft.items():
        for filetype, entries in by_ft.items():
            if not entries:
                continue

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=False)

            # Collect mins for tick computation
            f1_all_vals: List[float] = []
            mcc_all_vals: List[float] = []

            # F1 subplot
            for model, df in entries:
                st = style_for(model, filetype)
                label = f"{model}"
                x = df["bin_center"].values
                y = pd.to_numeric(df["F1"], errors="coerce").values
                y_low = pd.to_numeric(df["F1_CI_low"], errors="coerce").values
                y_high = pd.to_numeric(df["F1_CI_high"], errors="coerce").values
                f1_all_vals.extend(y_low.tolist())
                f1_all_vals.extend(y.tolist())
                ax1.plot(x, y, label=label, marker="o", markersize=4, linewidth=2, **st)
                ax1.fill_between(x, y_low, y_high, alpha=0.15, color=st["color"])

            f1_floor, _ = _compute_ticks(f1_all_vals)
            ax1.set_title(f"Decile F1-score — ({filetype})", fontsize=16)
            ax1.set_xlabel("Intron length (bin center)", fontsize=14)
            ax1.set_ylabel("F1-score", fontsize=14)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='both', labelsize=14)

            # MCC subplot
            for model, df in entries:
                st = style_for(model, filetype)
                label = f"{model}"
                x = df["bin_center"].values
                y = pd.to_numeric(df["MCC"], errors="coerce").values
                y_low = pd.to_numeric(df["MCC_CI_low"], errors="coerce").values
                y_high = pd.to_numeric(df["MCC_CI_high"], errors="coerce").values
                mcc_all_vals.extend(y_low.tolist())
                mcc_all_vals.extend(y.tolist())
                ax2.plot(x, y, label=label, marker="o", markersize=4, linewidth=2, **st)
                ax2.fill_between(x, y_low, y_high, alpha=0.15, color=st["color"])

            mcc_floor, _ = _compute_ticks(mcc_all_vals)

            # Enforce same y-axis range and ticks for F1-score and MCC
            common_floor = min(f1_floor, mcc_floor)
            common_mid = round((common_floor + 1.0) / 2.0, 2)
            common_ticks = [round(common_floor, 2), common_mid, 1.00]

            ax1.set_ylim(common_floor, 1.0)
            ax1.set_yticks(common_ticks)
            ax2.set_ylim(common_floor, 1.0)
            ax2.set_yticks(common_ticks)
            ax2.set_title(f"Decile MCC — ({filetype})", fontsize=16)
            ax2.set_xlabel("Intron length (bin center)", fontsize=14)
            ax2.set_ylabel("MCC", fontsize=14)
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='both', labelsize=14)

            # Shared legend
            handles, labels = ax1.get_legend_handles_labels()
            fig.legend(handles, labels, fontsize=12, ncol=4, loc='upper center')
            fig.tight_layout(rect=[0, 0, 1, 0.92])
            fpath = output_dir / f"deciles_{category}_{filetype}.pdf"
            fig.savefig(fpath, dpi=150)
            plt.close(fig)


def plot_smooth(smooth_files: List[Path], output_dir: Path) -> None:
    # Group by category then filetype
    by_category_ft: Dict[str, Dict[str, List[Tuple[str, pd.DataFrame]]]] = {}
    for f in smooth_files:
        model, category, filetype = parse_meta(f, SMOOTH_RE)
        df = pd.read_csv(f)
        df["Length"] = pd.to_numeric(df["Length"], errors="coerce")
        df["F1_smoothed"] = pd.to_numeric(df["F1_smoothed"], errors="coerce")
        df["MCC_smoothed"] = pd.to_numeric(df["MCC_smoothed"], errors="coerce")
        by_category_ft.setdefault(category, {}).setdefault(filetype, []).append((model, df))

    for category, by_ft in by_category_ft.items():
        for filetype, entries in by_ft.items():
            if not entries:
                continue

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=False)

            f1_all_vals: List[float] = []
            mcc_all_vals: List[float] = []

            # F1 subplot
            for model, df in entries:
                st = style_for(model, filetype)
                label = f"{model}"
                x = df["Length"].values
                y = df["F1_smoothed"].values
                f1_all_vals.extend(pd.to_numeric(y, errors="coerce").tolist())
                ax1.plot(x, y, label=label, marker="o", markersize=4, linewidth=2, **st)

            f1_floor, _ = _compute_ticks(f1_all_vals)
            ax1.set_title(f"Smoothed F1-score — ({filetype})", fontsize=16)
            ax1.set_xlabel("Intron length", fontsize=14)
            ax1.set_ylabel("F1-score (smoothed)", fontsize=14)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='both', labelsize=14)

            # MCC subplot
            for model, df in entries:
                st = style_for(model, filetype)
                label = f"{model}"
                x = df["Length"].values
                y = df["MCC_smoothed"].values
                mcc_all_vals.extend(pd.to_numeric(y, errors="coerce").tolist())
                ax2.plot(x, y, label=label, marker="o", markersize=4, linewidth=2, **st)

            mcc_floor, _ = _compute_ticks(mcc_all_vals)

            # Enforce same y-axis range and ticks for F1-score and MCC
            common_floor = min(f1_floor, mcc_floor)
            common_mid = round((common_floor + 1.0) / 2.0, 2)
            common_ticks = [round(common_floor, 2), common_mid, 1.00]

            ax1.set_ylim(common_floor, 1.0)
            ax1.set_yticks(common_ticks)
            ax2.set_ylim(common_floor, 1.0)
            ax2.set_yticks(common_ticks)
            ax2.set_title(f"Smoothed MCC — ({filetype})", fontsize=16)
            ax2.set_xlabel("Intron length", fontsize=14)
            ax2.set_ylabel("MCC (smoothed)", fontsize=14)
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='both', labelsize=14)

            # Shared legend
            handles, labels = ax1.get_legend_handles_labels()
            fig.legend(handles, labels, fontsize=12, ncol=4, loc='upper center')
            fig.tight_layout(rect=[0, 0, 1, 0.92])
            fpath = output_dir / f"smooth_{category}_{filetype}.pdf"
            fig.savefig(fpath, dpi=150)
            plt.close(fig)


def main() -> None:
    setup_logging("INFO")
    results_dir = Config.PROJECT_ROOT / "sensitivity_results"
    ensure_dir(results_dir)

    decile_files, smooth_files = find_result_files(results_dir)
    if not decile_files and not smooth_files:
        logger.warning(f"No sensitivity CSVs found in {results_dir}")
        return

    if decile_files:
        logger.info(f"Found {len(decile_files)} decile CSVs. Plotting...")
        plot_deciles(decile_files, results_dir)

    if smooth_files:
        logger.info(f"Found {len(smooth_files)} smooth CSVs. Plotting...")
        plot_smooth(smooth_files, results_dir)

    logger.info("All figures saved to sensitivity_results")


if __name__ == "__main__":
    main()


