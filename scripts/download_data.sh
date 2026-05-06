#!/usr/bin/env bash
# ============================================================
# Download IBM Telco Customer Churn dataset (v11.1.3+) from Kaggle.
#
# Prerequisites:
#   1. pip install kaggle  (or `uv pip install kaggle`)
#   2. Set KAGGLE_USERNAME and KAGGLE_KEY in .env
#      OR place ~/.kaggle/kaggle.json (chmod 600)
#
# Alternative (manual): download from the IBM Cognos samples page
# and place the .xlsx file in data/raw/ as `Telco_customer_churn.xlsx`.
# ============================================================

set -euo pipefail

DATA_DIR="data/raw"
mkdir -p "$DATA_DIR"

# Load .env if it exists
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

if ! command -v kaggle >/dev/null 2>&1; then
    echo "Error: 'kaggle' CLI not found."
    echo "Install with: pip install kaggle"
    echo ""
    echo "Or download manually from:"
    echo "  https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
    echo "and place the file in $DATA_DIR/"
    exit 1
fi

echo "Downloading IBM Telco Customer Churn dataset..."
kaggle datasets download -d blastchar/telco-customer-churn -p "$DATA_DIR" --unzip

echo ""
echo "Note: the basic Kaggle version may NOT include 'Satisfaction Score'."
echo "If your downloaded file lacks that column, fetch v11.1.3+ from"
echo "IBM Cognos samples (search 'Telco customer churn') and place the"
echo "richer file in $DATA_DIR/Telco_customer_churn.xlsx"
echo ""
echo "Files in $DATA_DIR:"
ls -la "$DATA_DIR"
