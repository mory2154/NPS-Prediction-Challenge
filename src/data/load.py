"""
Load raw IBM Telco Customer Churn data and produce an interim, cleaned dataset.

To be implemented in Phase 1 / Phase 3.
Run with: `python -m src.data.load`
"""

from src.config import DATA_RAW, DATA_INTERIM


def main() -> None:
    print(f"[src.data.load] Loading raw data from {DATA_RAW}")
    print(f"[src.data.load] Output target: {DATA_INTERIM}")
    print("[src.data.load] TODO: implement raw → interim cleaning in Phase 3.")


if __name__ == "__main__":
    main()
