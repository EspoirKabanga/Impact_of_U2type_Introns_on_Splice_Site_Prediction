import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# Reduce TensorFlow verbosity BEFORE any TF import (via downstream modules)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # Errors and Warnings only
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')

from config import Config
from collections import OrderedDict
from data_preprocessing import DataLoader, DNASequenceProcessor
from models import ModelFactory, BaseModel
from metrics import MetricsCalculator
from utils import read_sequences_from_file


logger = logging.getLogger("sensitivity")


def setup_logging(level: str = "INFO") -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def read_introns_file(introns_file: Path) -> List[str]:
    with open(introns_file, "r") as f:
        return [line.strip().upper() for line in f if line.strip()]


def build_introns_indices(intron_sequences: List[str], kmer_size: int = 12) -> Tuple[Dict[int, set], Dict[str, set]]:
    """Build indices to accelerate intron lookup inside a longer sequence.
    - introns_by_length: length -> set(intron_sequence)
    - kmer_to_lengths: k-mer -> set(lengths that contain this k-mer)
    """
    introns_by_length: Dict[int, set] = {}
    kmer_to_lengths: Dict[str, set] = {}

    for seq in intron_sequences:
        L = len(seq)
        if L not in introns_by_length:
            introns_by_length[L] = set()
        introns_by_length[L].add(seq)

        if L >= kmer_size:
            for i in range(L - kmer_size + 1):
                kmer = seq[i : i + kmer_size]
                bucket = kmer_to_lengths.get(kmer)
                if bucket is None:
                    bucket = set()
                    kmer_to_lengths[kmer] = bucket
                bucket.add(L)

    return introns_by_length, kmer_to_lengths


def find_intron_length_in_sequence(
    sequence: str,
    introns_by_length: Dict[int, set],
    kmer_to_lengths: Dict[str, set],
    kmer_size: int = 12,
) -> Optional[int]:
    """Best-effort detection of embedded intron length in a positive sequence.
    Strategy:
      1) Gather candidate lengths from any k-mer hits
      2) For candidate lengths, check exact substring membership against all introns of that length
      3) Return the first length found; if multiple, return the maximum length (more conservative)
    """
    seq = sequence.upper()
    candidate_lengths: set = set()
    if len(seq) >= kmer_size:
        for i in range(len(seq) - kmer_size + 1):
            kmer = seq[i : i + kmer_size]
            lengths_bucket = kmer_to_lengths.get(kmer)
            if lengths_bucket:
                candidate_lengths.update(lengths_bucket)

    if not candidate_lengths:
        return None

    # Prefer checking longer lengths first; cheaper to reduce false positives
    for L in sorted(candidate_lengths, reverse=True):
        intron_set = introns_by_length.get(L)
        if not intron_set:
            continue
        # Slide windows of length L over the sequence and test set membership
        if len(seq) >= L:
            for j in range(len(seq) - L + 1):
                window = seq[j : j + L]
                if window in intron_set:
                    return L
    return None


def compute_bins(lengths: List[int], n_bins: int) -> np.ndarray:
    arr = np.array(lengths, dtype=np.int32)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    return np.unique(np.quantile(arr, quantiles, interpolation="linear").astype(int))


def bootstrap_ci(
    calculator: MetricsCalculator,
    y_true_one_hot: np.ndarray,
    y_scores: np.ndarray,
    pos_indices: np.ndarray,
    neg_indices: np.ndarray,
    n_iter: int = 200,
    seed: int = 42,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return (F1_low,F1_high), (MCC_low,MCC_high) 95% CI by bootstrap."""
    rng = np.random.default_rng(seed)
    f1_values: List[float] = []
    mcc_values: List[float] = []

    for _ in range(n_iter):
        # Resample positives and negatives with replacement, keeping original counts
        sample_pos = rng.choice(pos_indices, size=len(pos_indices), replace=True)
        sample_neg = rng.choice(neg_indices, size=len(neg_indices), replace=True)
        idx = np.concatenate([sample_pos, sample_neg])

        metrics = calculator.calculate_metrics(y_scores[idx], y_true_one_hot[idx])
        # Indices: recall, precision, f1 (2), fpr, fdr, specificity, sensitivity, mcc (7), auroc, auprc
        f1_values.append(float(metrics[2]))
        mcc_values.append(float(metrics[7]))

    f1_ci = (float(np.percentile(f1_values, 2.5)), float(np.percentile(f1_values, 97.5)))
    mcc_ci = (float(np.percentile(mcc_values, 2.5)), float(np.percentile(mcc_values, 97.5)))
    return f1_ci, mcc_ci


def smooth_series(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """LOWESS smoothing if statsmodels is available; otherwise simple moving average."""
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess  # type: ignore

        smoothed = lowess(y, x, frac=0.3, return_sorted=False)
        return smoothed.astype(float)
    except Exception:
        window = max(3, min(11, len(y) // 5 * 2 + 1))
        pad = window // 2
        y_pad = np.pad(y, (pad, pad), mode="edge")
        kernel = np.ones(window, dtype=float) / window
        smoothed = np.convolve(y_pad, kernel, mode="valid")
        return smoothed.astype(float)


def run_analysis(
    config: Config,
    filetype: str,
    data_category: str,
    model_name: str,
    output_dir: Path,
    deciles: int = 10,
    bootstrap_iters: int = 200,
    kmer_size: int = 12,
) -> None:
    config.setup_tensorflow()
    config.ensure_directories()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build combined positives (short, long, mix_long_short), deduplicated
    pos_categories = ["short", "long", "mix_long_short"]
    all_pos_sequences: List[str] = []
    for cat in pos_categories:
        pos_path = config.get_data_file_path(filetype, cat)
        seqs = read_sequences_from_file(pos_path)
        all_pos_sequences.extend(seqs)
    # Deduplicate while preserving order
    all_pos_sequences = list(OrderedDict.fromkeys([s.upper() for s in all_pos_sequences]))

    # 2) Load negatives (full file)
    neg_path = config.get_negative_file_path(filetype)
    all_neg_sequences = [s.upper() for s in read_sequences_from_file(neg_path)]

    # 3) Create labels and split indices
    pos_labels = np.ones(len(all_pos_sequences), dtype=np.int32)
    neg_labels = np.zeros(len(all_neg_sequences), dtype=np.int32)
    all_sequences: List[str] = all_pos_sequences + all_neg_sequences
    y_all = np.concatenate([pos_labels, neg_labels], axis=0)

    # Logging dataset overview before split
    planned_neg_after_ratio = min(3 * len(all_pos_sequences), len(all_neg_sequences))
    logger.info(
        f"Deciles: {deciles} | Total positives (dedup): {len(all_pos_sequences)} | "
        f"Negatives available: {len(all_neg_sequences)} | Planned negatives after 1:3 (upper bound): {planned_neg_after_ratio}"
    )

    indices = np.arange(len(all_sequences))
    from sklearn.model_selection import train_test_split
    train_idx, test_idx, y_train, y_test = train_test_split(
        indices, y_all, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y_all
    )

    # 4) Encode sequences for train/test using DNASequenceProcessor
    processor = DNASequenceProcessor(config)
    seq_train = [all_sequences[i] for i in train_idx]
    seq_test = [all_sequences[i] for i in test_idx]

    X_train = processor.one_hot_encode(seq_train)
    X_test = processor.one_hot_encode(seq_test)

    from tensorflow.keras.utils import to_categorical
    y_train_cat = to_categorical(y_train, num_classes=config.N_CLASSES).astype(np.float32)
    y_test_cat = to_categorical(y_test, num_classes=config.N_CLASSES).astype(np.float32)

    # Train/val/test size logging (validation via validation_split=0.1)
    val_count = int(len(y_train) * 0.1)
    train_effective = len(y_train) - val_count
    logger.info(
        f"Split sizes | Train: {len(y_train)} (effective train: {train_effective}, val: {val_count}) | Test: {len(y_test)}"
    )

    # Pre-training: compute length deciles and print length ranges using test positives
    introns_path = config.SRC_DIR / "zzz_all_introns.txt"
    if not introns_path.exists():
        raise FileNotFoundError(f"Intron list not found: {introns_path}")

    intron_sequences = read_introns_file(introns_path)
    introns_by_length, kmer_to_lengths = build_introns_indices(intron_sequences, kmer_size=kmer_size)

    pre_positive_indices = np.where(y_test == 1)[0]
    pre_positive_sequences = [seq_test[idx] for idx in pre_positive_indices]

    pre_pos_idx_to_length: Dict[int, Optional[int]] = {}
    for local_i, seq in enumerate(pre_positive_sequences):
        length = find_intron_length_in_sequence(seq, introns_by_length, kmer_to_lengths, kmer_size=kmer_size)
        global_idx = int(pre_positive_indices[local_i])
        pre_pos_idx_to_length[global_idx] = length

    pre_valid_lengths = [l for l in pre_pos_idx_to_length.values() if l is not None]
    if len(pre_valid_lengths) < max(10, deciles):
        logger.warning("Not enough positives with detected intron lengths for decile analysis; reducing bins.")
        deciles = max(2, min(deciles, len(pre_valid_lengths)))

    pre_bin_edges = compute_bins(pre_valid_lengths, deciles)
    logger.info(f"Deciles (pre-train): {len(pre_bin_edges) - 1}")
    logger.info(f"Length bin edges (pre-train): {pre_bin_edges.tolist()}")
    for b in range(len(pre_bin_edges) - 1):
        low, high = int(pre_bin_edges[b]), int(pre_bin_edges[b + 1])
        logger.info(f"Pre-train decile {b + 1}/{len(pre_bin_edges) - 1} | Length range: {low}-{high}")

    # Model
    base_model: BaseModel = ModelFactory.create_model(model_name, config)
    model = base_model.get_model()

    # Train with a simple validation split from train set
    logger.info(
        f"Training {model_name} for sensitivity analysis on {filetype} - {data_category} (epochs={config.EPOCHS})"
    )
    model.fit(
        X_train,
        y_train_cat,
        epochs=config.EPOCHS,
        validation_split=0.1,
        batch_size=config.BATCH_SIZES[0] if config.BATCH_SIZES else 64,
        verbose=2,
    )

    # Predictions on test set
    y_scores = model.predict(X_test, verbose=0)

    # Map positive test samples to intron lengths
    # Reuse already built intron indices from pre-training step

    # Determine intron lengths for positives only (use original sequences preserved through split)
    positive_indices = np.where(y_test == 1)[0]
    positive_sequences = [seq_test[idx] for idx in positive_indices]

    detected_lengths: Dict[int, int] = {}
    pos_idx_to_length: Dict[int, Optional[int]] = {}
    for local_i, seq in enumerate(positive_sequences):
        length = find_intron_length_in_sequence(seq, introns_by_length, kmer_to_lengths, kmer_size=kmer_size)
        global_idx = int(positive_indices[local_i])
        pos_idx_to_length[global_idx] = length
        if length is not None:
            detected_lengths[length] = detected_lengths.get(length, 0) + 1

    # Build length deciles from detected positive lengths
    valid_lengths = [l for l in pos_idx_to_length.values() if l is not None]
    if len(valid_lengths) < max(10, deciles):
        logger.warning("Not enough positives with detected intron lengths for decile analysis; reducing bins.")
        deciles = max(2, min(deciles, len(valid_lengths)))

    bin_edges = compute_bins(valid_lengths, deciles)
    logger.info(f"Deciles computed: {len(bin_edges) - 1}")
    logger.info(f"Length bin edges: {bin_edges.tolist()}")

    # Compute per-bin metrics and bootstrap CIs
    calculator = MetricsCalculator()
    records: List[List[str]] = []

    all_neg_indices = np.where(y_test == 0)[0]
    rng = np.random.default_rng(config.RANDOM_SEED)
    for b in range(len(bin_edges) - 1):
        low, high = int(bin_edges[b]), int(bin_edges[b + 1])
        bin_pos_indices = np.array(
            [idx for idx, L in pos_idx_to_length.items() if (L is not None and low <= L <= high)], dtype=int
        )
        if bin_pos_indices.size == 0:
            continue

        # Maintain 1:3 positive:negative ratio for each bin
        num_pos = bin_pos_indices.size
        num_neg_target = min(3 * num_pos, all_neg_indices.size)
        if num_neg_target > 0:
            sampled_negs = rng.choice(all_neg_indices, size=num_neg_target, replace=False)
        else:
            sampled_negs = np.array([], dtype=int)

        eval_indices = np.concatenate([bin_pos_indices, sampled_negs])

        # Log current decile range and counts
        logger.info(
            f"Decile {b + 1}/{len(bin_edges) - 1} | Length range: {low}-{high} | "
            f"Positives: {num_pos} | Sampled negatives (1:3): {sampled_negs.size}"
        )
        metrics_tuple = calculator.calculate_metrics(y_scores[eval_indices], y_test_cat[eval_indices])
        # Positions: recall(0), precision(1), f1(2), fpr(3), fdr(4), specificity(5), sensitivity(6), mcc(7), auroc(8), auprc(9)

        f1_ci, mcc_ci = bootstrap_ci(
            calculator, y_test_cat, y_scores, bin_pos_indices, all_neg_indices, n_iter=bootstrap_iters, seed=config.RANDOM_SEED
        )

        row = [
            f"{low}-{high}",
            str(len(bin_pos_indices)),
            f"{float(metrics_tuple[2]):.4f}",
            f"{f1_ci[0]:.4f}",
            f"{f1_ci[1]:.4f}",
            f"{float(metrics_tuple[7]):.4f}",
            f"{mcc_ci[0]:.4f}",
            f"{mcc_ci[1]:.4f}",
        ]
        records.append(row)

    # Save decile metrics CSV
    decile_file = output_dir / f"decile_metrics_{model_name}_{data_category}_{filetype}.csv"
    import csv

    with open(decile_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Length_Bin", "Num_Positives", "F1", "F1_CI_low", "F1_CI_high", "MCC", "MCC_CI_low", "MCC_CI_high"])
        for row in records:
            writer.writerow(row)
    logger.info(f"Decile metrics saved to {decile_file}")

    # Prepare data for smoothing (use bin centers)
    bin_centers: List[float] = []
    f1_values: List[float] = []
    mcc_values: List[float] = []
    for row in records:
        low, high = row[0].split("-")
        center = (int(low) + int(high)) / 2.0
        bin_centers.append(center)
        f1_values.append(float(row[2]))
        mcc_values.append(float(row[5]))

    if bin_centers:
        x = np.array(bin_centers, dtype=float)
        f1_sm = smooth_series(x, np.array(f1_values, dtype=float))
        mcc_sm = smooth_series(x, np.array(mcc_values, dtype=float))

        trend_file = output_dir / f"smooth_trend_{model_name}_{data_category}_{filetype}.csv"
        with open(trend_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Length", "F1_smoothed", "MCC_smoothed"]) 
            for xi, f1i, mcci in zip(x.tolist(), f1_sm.tolist(), mcc_sm.tolist()):
                writer.writerow([f"{xi:.2f}", f"{f1i:.4f}", f"{mcci:.4f}"])
        logger.info(f"Smoothed trend saved to {trend_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Sensitivity analysis: performance vs intron length",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--filetype",
        choices=["donor", "acceptor"],
        default="acceptor",
        help="File type to analyze",
    )
    parser.add_argument(
        "--data-category",
        choices=["short", "long", "mix_long_short", "multiple", "single", "mix_single_multiple"],
        default="short",
        help="Data category to analyze",
    )
    parser.add_argument(
        "--model",
        choices=["IntSplicer", "SpliceRover", "SpliceFinder", "DeepSplicer"],
        default="IntSplicer",
        help="Model to train for analysis (ignored if --all-models)",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run sensitivity analysis for all available models",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    parser.add_argument("--deciles", type=int, default=10, help="Number of bins across length")
    parser.add_argument("--bootstrap-iters", type=int, default=200, help="Bootstrap iterations for 95% CI")
    parser.add_argument("--kmer-size", type=int, default=12, help="K-mer size used for intron indexing")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Config.PROJECT_ROOT / "sensitivity_results"),
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--all-filetypes",
        action="store_true",
        help="Run sensitivity analysis for both donor and acceptor",
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    config = Config()

    # Always run across ALL models and BOTH filetypes as requested
    model_list: List[str] = ModelFactory.list_available_models()
    filetypes: List[str] = ["donor", "acceptor"]

    logger.info(f"Running sensitivity analysis for models={model_list} on filetypes={filetypes} (outputs → {args.output_dir})")

    for ft in filetypes:
        for model_name in model_list:
            try:
                logger.info("\n" + "-" * 80)
                logger.info(f"Starting sensitivity analysis: model={model_name} | filetype={ft} | category={args.data_category}")
                run_analysis(
                    config=config,
                    filetype=ft,
                    data_category=args.data_category,
                    model_name=model_name,
                    output_dir=Path(args.output_dir),
                    deciles=args.deciles,
                    bootstrap_iters=args.bootstrap_iters,
                    kmer_size=args.kmer_size,
                )
            except Exception as e:
                logger.error(f"Sensitivity analysis failed for model={model_name}, filetype={ft}: {e}")


if __name__ == "__main__":
    main()


