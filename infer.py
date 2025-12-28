#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ProteinGym-style zero-shot inference with ESM-2 (hub weights).

This script reads one or more CSV files containing deep mutational scanning (DMS)
variants and evaluates how well ESM-2 zero-shot scores correlate with the
experimental DMS measurements.

Expected input columns (per row / variant):
  - `mutant`: mutation string in the common format like "A123G"
              (wildtype AA, 1-based position, mutant AA).
  - `mutated_sequence`: the full *mutant* protein sequence.
  - `DMS_score`: the experimental fitness/score for this variant.

For each variant, we:
  1) Recover the wildtype sequence by replacing the mutated residue back to the
     wildtype amino acid at the given position.
  2) Mask the mutated position in the WT sequence.
  3) Run ESM-2 to obtain log-probabilities at the masked position.
  4) Compute a delta log-probability:
        Δ = log P(mutant_aa | context) - log P(wildtype_aa | context)
     which is a standard "masked marginal" zero-shot score.

Outputs:
  - For each input CSV: a new CSV with an `esm2_delta_logp` column.
  - A `summary.csv` aggregating per-file Spearman statistics.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from scipy.stats import spearmanr

import esm

from importlib.metadata import distributions

# Minimal schema required by this script.
REQUIRED_COLS = {"mutant", "mutated_sequence", "DMS_score"}


# === Shared helpers (same style as saprot/esm) ===
def compute_spearman(pred_scores, true_scores) -> tuple[float | None, float | None]:
    """Compute Spearman correlation, returning (rho, pval) with NaN-safe handling."""
    rho, pval = spearmanr(pred_scores, true_scores, nan_policy="omit")
    rho_val = None if rho is None or (isinstance(rho, float) and pd.isna(rho)) else float(rho)
    pval_val = None if pval is None or (isinstance(pval, float) and pd.isna(pval)) else float(pval)
    return rho_val, pval_val


def _fmt_float(x: float | None, *, fmt: str) -> str:
    return "nan" if x is None else format(x, fmt)


def collect_installed_packages() -> list[str]:
    """Return all installed distributions as `name==version` (sorted)."""
    items: list[str] = []
    for dist in distributions():
        metadata = getattr(dist, "metadata", None)
        name = metadata.get("Name") if metadata is not None else None
        if not name:
            continue
        items.append(f"{name}=={dist.version}")
    return sorted(set(items), key=str.lower)


def print_runtime_environment() -> None:
    """Print Python + full installed package versions (once per run)."""
    print("========== Runtime ==========")
    print(f"Python:        {sys.version.replace(os.linesep, ' ')}")
    print(f"Executable:    {sys.executable}")
    print(f"Platform:      {sys.platform}")
    print("Packages:")
    for item in collect_installed_packages():
        print(f"  - {item}")
    print("=============================\n")


# === ESM-2 specific helpers ===
def parse_mutant(mut_str: str) -> tuple[str, int, str]:
    """Parse a mutation string like 'A123G' -> (wt_aa, pos1, mut_aa)."""
    wt_aa = mut_str[0]
    mut_aa = mut_str[-1]
    pos1 = int(mut_str[1:-1])
    return wt_aa, pos1, mut_aa


def recover_wt_sequence(mut_seq: str, wt_aa: str, pos1: int) -> str:
    """Reconstruct the wildtype sequence by restoring the WT residue at `pos1` (1-based)."""
    return mut_seq[: pos1 - 1] + wt_aa + mut_seq[pos1:]


def resolve_csv_paths(*, data_dir: Path, csv: str | None) -> list[Path]:
    """Resolve either a single CSV path or all CSVs under `data_dir`."""
    if csv is None:
        return sorted(p for p in data_dir.glob("*.csv") if p.is_file())

    candidate = Path(csv)
    if not candidate.is_absolute():
        candidate = (data_dir / candidate).resolve()

    if not candidate.exists():
        raise FileNotFoundError(f"CSV not found: {candidate}")

    return [candidate]


def load_dataset(*, csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path.name}: {sorted(missing)}")
    return df


def load_model(*, model_name: str) -> tuple[torch.nn.Module, object, str]:
    """Step 1: Load model + alphabet (downloaded from hub via esm)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.eval().to(device)
    return model, alphabet, device


def iter_batches(items: list[tuple[int, str, str, int, str]], batch_size: int) -> Iterable[list[tuple[int, str, str, int, str]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def preprocess_dataset(*, df: pd.DataFrame) -> tuple[list[tuple[int, str, str, int, str]], list[float]]:
    """
    Return:
      - prepared: list[(row_idx, wt_seq, wt_aa, pos1, mut_aa)]
      - true_scores: list[float]
    """
    prepared: list[tuple[int, str, str, int, str]] = []
    true_scores: list[float] = []

    for row_idx, row in enumerate(df.itertuples(index=False)):
        mut_str = str(getattr(row, "mutant"))
        mut_seq = str(getattr(row, "mutated_sequence")).strip().upper().replace(" ", "")
        score = float(getattr(row, "DMS_score"))

        wt_aa, pos1, mut_aa = parse_mutant(mut_str)
        pos0 = pos1 - 1
        if not (0 <= pos0 < len(mut_seq)):
            raise ValueError(f"{row_idx}: position {pos1} out of range for sequence length {len(mut_seq)}")
        if mut_seq[pos0] != mut_aa:
            raise ValueError(f"{row_idx}: mutated_sequence[{pos1}] is {mut_seq[pos0]!r}, expected mutant AA={mut_aa!r}")

        wt_seq = recover_wt_sequence(mut_seq, wt_aa, pos1)
        prepared.append((row_idx, wt_seq, wt_aa, pos1, mut_aa))
        true_scores.append(score)

    return prepared, true_scores


def score_delta_logp(
    *,
    model: torch.nn.Module,
    alphabet,
    device: str,
    batch_size: int,
    prepared: list[tuple[int, str, str, int, str]],
    progress_every: int,
) -> list[float]:
    """Step 2: masked-marginal Δ log-prob scoring with ESM-2."""
    batch_converter = alphabet.get_batch_converter()
    preds: list[float] = [float("nan")] * len(prepared)

    processed = 0
    with torch.no_grad():
        for batch in iter_batches(prepared, batch_size):
            data = [(f"row{row_idx}", wt_seq) for (row_idx, wt_seq, _, _, _) in batch]
            _, _, tokens = batch_converter(data)
            tokens = tokens.to(device)

            for j, (_, _, _, pos1, _) in enumerate(batch):
                tokens[j, 1 + (pos1 - 1)] = alphabet.mask_idx

            logits = model(tokens)["logits"]
            logp = torch.log_softmax(logits, dim=-1)

            for j, (row_idx, _, wt_aa, pos1, mut_aa) in enumerate(batch):
                pos_tok = 1 + (pos1 - 1)  # +BOS
                wt_i = alphabet.get_idx(wt_aa)
                mut_i = alphabet.get_idx(mut_aa)
                preds[row_idx] = float(logp[j, pos_tok, mut_i] - logp[j, pos_tok, wt_i])
                processed += 1

            if progress_every > 0 and (processed % progress_every == 0 or processed == len(prepared)):
                print(f"  predicted {processed}/{len(prepared)}")

    return preds


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ESM-2 zero-shot variant effect prediction via Δ = logP(mut|context) - logP(wt|context)."
    )
    parser.add_argument(
        "--input_csv",
        default=None,
        help="Only process this CSV (basename under data_dir, or an absolute path). If omitted, process all CSVs in data_dir.",
    )
    parser.add_argument("--data_dir", default="/opt/ml/processing/input/data")
    parser.add_argument("--output_dir", default="/opt/ml/processing/output")
    parser.add_argument("--output_suffix", default="_esm2_zeroshot.csv")
    parser.add_argument("--progress_every", type=int, default=100, help="Print progress every N variants (0 disables).")

    parser.add_argument("--model_name", default="esm2_t33_650M_UR50D", help="ESM-2 model name (downloaded from hub).")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for model forward passes.")
    return parser


def main() -> None:
    args = create_parser().parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_runtime_environment()

    csv_paths = resolve_csv_paths(data_dir=data_dir, csv=args.input_csv)
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under: {data_dir}")

    # Step 1: Load model
    model, alphabet, device = load_model(model_name=args.model_name)
    print(f"Loading ESM-2 model {args.model_name!r} from hub (device={device})")

    summaries: list[dict] = []

    for csv_path in csv_paths:
        # Step 2: Load dataset
        df = load_dataset(csv_path=csv_path)

        # Step 3: Dataset preprocessing
        prepared, true_scores = preprocess_dataset(df=df)

        # Step 4: Model inference
        pred_scores = score_delta_logp(
            model=model,
            alphabet=alphabet,
            device=device,
            batch_size=args.batch_size,
            prepared=prepared,
            progress_every=args.progress_every,
        )

        # Step 5: Spearman correlation (rank-based; robust to monotonic transforms)
        rho, pval = compute_spearman(pred_scores, true_scores)
        df["esm2_delta_logp"] = pred_scores

        out_name = f"{csv_path.stem}{args.output_suffix}"
        out_path = output_dir / out_name
        df.to_csv(out_path, index=False)

        print("\n========== ProteinGym zero-shot ==========")
        print("Model:        ESM-2")
        print(f"CSV:          {csv_path.name}")
        print(f"Variants:     {len(df)}")
        print(f"Spearman ρ:   {_fmt_float(rho, fmt='.4f')}")
        print(f"P-value:      {_fmt_float(pval, fmt='.2e')}")
        print(f"Saved to:     {out_path}")
        print("==========================================\n")

        summaries.append(
            {
                "csv": csv_path.name,
                "variants": int(len(df)),
                "spearman_rho": rho,
                "p_value": pval,
                "output_csv": out_path.name,
            }
        )

    # Write a compact summary across input CSVs (same shape as ESM script).
    summary_path = output_dir / "summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
