"""
Build the NPS classification target from `Satisfaction Score`.

Implements the baseline mapping defined in src.config and (later)
alternative mappings for sensitivity analysis. Drops all leaky features.

To be implemented in Phase 2.
Run with: `python -m src.data.target`
"""

from src.config import (
    DATA_INTERIM,
    DATA_PROCESSED,
    LEAKY_FEATURES,
    SATISFACTION_TO_NPS_BASELINE,
)


def main() -> None:
    print(f"[src.data.target] Reading from {DATA_INTERIM}")
    print(f"[src.data.target] Writing to {DATA_PROCESSED}")
    print(f"[src.data.target] Baseline mapping: {SATISFACTION_TO_NPS_BASELINE}")
    print(f"[src.data.target] Will drop: {LEAKY_FEATURES}")
    print("[src.data.target] TODO: implement in Phase 2.")


if __name__ == "__main__":
    main()
