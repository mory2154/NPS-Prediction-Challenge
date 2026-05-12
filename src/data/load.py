"""
Load the IBM Telco Customer Churn dataset from data/raw/.

Supports three layouts found in the wild:

  1. **Single-file consolidated** (e.g. Kaggle yeanzc):
     One .xlsx or .csv with all ~33 columns including Satisfaction Score.

  2. **Multi-sheet workbook** (older IBM Cognos exports):
     One .xlsx with several sheets, all sharing 'Customer ID'.

  3. **Multi-file IBM Cognos v11.1.3** (Kaggle ylchang):
     Six .xlsx files split by topic — demographics, location, services,
     status, population, and a (sometimes partial) consolidated file.
     Five are keyed on 'Customer ID', population is keyed on 'Zip Code'.

The loader inspects the directory, detects the layout, and produces a single
unified DataFrame. It validates that 'Satisfaction Score' is present.

Run as CLI:
    python -m src.data.load
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.config import DATA_INTERIM, DATA_RAW

REQUIRED_COLUMNS = {"Satisfaction Score"}

# Canonical column names. Aliases get renamed at load time so downstream code
# can use a single naming convention.
COLUMN_ALIASES: dict[str, str] = {
    # Customer identifier
    "customerID": "Customer ID",
    "CustomerID": "Customer ID",
    # Tenure
    "tenure": "Tenure Months",
    "Tenure in Months": "Tenure Months",
    # Charges
    "MonthlyCharges": "Monthly Charges",
    "Monthly Charge": "Monthly Charges",
    "TotalCharges": "Total Charges",
    # Services & billing
    "PaymentMethod": "Payment Method",
    "InternetService": "Internet Service",
    "OnlineSecurity": "Online Security",
    "OnlineBackup": "Online Backup",
    "DeviceProtection": "Device Protection",
    "Device Protection Plan": "Device Protection",
    "TechSupport": "Tech Support",
    "Premium Tech Support": "Tech Support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming Movies",
    "PaperlessBilling": "Paperless Billing",
    # Demographics
    "SeniorCitizen": "Senior Citizen",
    "MultipleLines": "Multiple Lines",
    "PhoneService": "Phone Service",
}


# ============================================================
# Helpers
# ============================================================
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and apply canonical aliases."""
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    rename_map = {old: new for old, new in COLUMN_ALIASES.items() if old in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _list_data_files() -> tuple[list[Path], list[Path]]:
    """List xlsx and csv files in data/raw/, excluding hidden files."""
    if not DATA_RAW.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {DATA_RAW}")
    xlsx = sorted(
        p for p in DATA_RAW.glob("*.xlsx")
        if p.is_file() and not p.name.startswith(".")
    )
    csv = sorted(
        p for p in DATA_RAW.glob("*.csv")
        if p.is_file() and not p.name.startswith(".")
    )
    return xlsx, csv


def _load_one_xlsx(path: Path) -> pd.DataFrame:
    """Load a single xlsx, merging sheets if multi-sheet (all on Customer ID)."""
    sheets = pd.read_excel(path, sheet_name=None)

    if len(sheets) == 1:
        return next(iter(sheets.values()))

    # Multi-sheet — merge the sheets that share Customer ID
    keyed = [
        _normalize_columns(df) for df in sheets.values()
        if "Customer ID" in _normalize_columns(df).columns
    ]
    if not keyed:
        # Fallback: return the first sheet
        return next(iter(sheets.values()))

    base = keyed[0]
    for df in keyed[1:]:
        new_cols = ["Customer ID"] + [c for c in df.columns if c not in base.columns]
        if len(new_cols) > 1:
            base = base.merge(df[new_cols], on="Customer ID", how="left")
    return base


def _load_multifile(xlsx_files: list[Path]) -> pd.DataFrame:
    """Load and merge multiple xlsx files (IBM Cognos v11.1.3 layout)."""
    print(f"  Multi-file layout detected ({len(xlsx_files)} files)")

    # Load and normalise each
    loaded: list[tuple[str, pd.DataFrame]] = []
    for f in xlsx_files:
        df = _load_one_xlsx(f)
        df = _normalize_columns(df)
        loaded.append((f.stem, df))
        print(f"    · {f.name}: {len(df):>5} rows × {len(df.columns):>2} cols")

    # Partition by join key
    customer_keyed = [(n, d) for n, d in loaded if "Customer ID" in d.columns]
    zip_keyed = [
        (n, d) for n, d in loaded
        if "Zip Code" in d.columns and "Customer ID" not in d.columns
    ]
    orphan = [
        (n, d) for n, d in loaded
        if "Customer ID" not in d.columns and "Zip Code" not in d.columns
    ]

    if not customer_keyed:
        raise ValueError(
            "No file in data/raw/ contains a 'Customer ID' column to merge on. "
            f"Files seen: {[n for n, _ in loaded]}"
        )

    # Sort customer-keyed by descending column count → start from the richest
    customer_keyed.sort(key=lambda x: (-len(x[1].columns), x[0]))
    base_name, base = customer_keyed[0]
    print(f"\n  Base file: {base_name} ({len(base)} rows × {len(base.columns)} cols)")

    # Merge other customer-keyed files, only adding new columns
    for name, df in customer_keyed[1:]:
        new_cols = ["Customer ID"] + [c for c in df.columns if c not in base.columns]
        if len(new_cols) > 1:
            base = base.merge(df[new_cols], on="Customer ID", how="left")
            print(f"    + {name}: +{len(new_cols) - 1} new cols → "
                  f"{len(base)} × {len(base.columns)}")
        else:
            print(f"    - {name}: no new columns, skipped")

    # Merge zip-keyed files (e.g. population) via Zip Code
    for name, df in zip_keyed:
        if "Zip Code" in base.columns:
            new_cols = ["Zip Code"] + [c for c in df.columns if c not in base.columns]
            if len(new_cols) > 1:
                base = base.merge(df[new_cols], on="Zip Code", how="left")
                print(f"    + {name}: joined on Zip Code, +{len(new_cols) - 1} cols")
        else:
            print(f"    - {name}: skipped — base has no 'Zip Code'")

    if orphan:
        print(f"\n  ⚠ Files without recognized join key (skipped): "
              f"{[n for n, _ in orphan]}")

    return base


# ============================================================
# Validation
# ============================================================
def _validate(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if not missing:
        return
    raise ValueError(
        f"\n\nThe loaded dataset is missing required column(s): {missing}\n\n"
        f"This means you may have the BASIC Kaggle dataset 'blastchar/telco-customer-churn',\n"
        f"which lacks the 'Satisfaction Score' column needed to build the NPS target.\n\n"
        f"Recommended sources:\n"
        f"  · https://www.kaggle.com/datasets/ylchang/telco-customer-churn-1113\n"
        f"  · https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset\n\n"
        f"Found {len(df.columns)} columns starting with: {list(df.columns)[:8]}...\n"
    )


# ============================================================
# Public API
# ============================================================
def load_raw_telco(verbose: bool = True) -> pd.DataFrame:
    """Find, load, validate the IBM Telco dataset (any supported layout)."""
    xlsx, csv = _list_data_files()

    if not xlsx and not csv:
        raise FileNotFoundError(
            f"\nNo data file found in {DATA_RAW}.\n"
            f"Place the IBM Telco dataset there. Recommended:\n"
            f"  https://www.kaggle.com/datasets/ylchang/telco-customer-churn-1113\n"
        )

    if verbose:
        print(f"Scanning {DATA_RAW}...")

    # Layout decision tree:
    #   2+ xlsx files → multi-file IBM Cognos
    #   1 xlsx file → single (possibly multi-sheet)
    #   only csv → load first csv
    if len(xlsx) >= 2:
        df = _load_multifile(xlsx)
    elif len(xlsx) == 1:
        if verbose:
            print(f"  Single-file layout: {xlsx[0].name}")
        df = _load_one_xlsx(xlsx[0])
    else:
        if verbose:
            print(f"  CSV layout: {csv[0].name}")
        df = pd.read_csv(csv[0])

    df = _normalize_columns(df)
    _validate(df)

    # Coerce Total Charges if stored as string (basic Kaggle CSV quirk)
    if "Total Charges" in df.columns and df["Total Charges"].dtype == "object":
        df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

    if verbose:
        print(f"\n  ✓ Final dataset: {len(df):,} rows × {len(df.columns)} columns")

    return df


# ============================================================
# CLI
# ============================================================
def main() -> None:
    df = load_raw_telco()
    out = DATA_INTERIM / "telco_raw.parquet"
    df.to_parquet(out, index=False)
    print(f"  ✓ Saved to {out}")
    print(f"\nColumns ({len(df.columns)}):")
    for i, c in enumerate(df.columns, 1):
        dtype = str(df[c].dtype)
        n_unique = df[c].nunique(dropna=True)
        n_null = df[c].isna().sum()
        print(f"  {i:>2}. {c:<35} {dtype:<10} ({n_unique:>5} unique, {n_null:>4} null)")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as e:
        print(f"\n✗ {e}", file=sys.stderr)
        sys.exit(1)
