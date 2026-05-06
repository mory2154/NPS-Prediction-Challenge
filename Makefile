# ============================================================
# NPS Prediction Challenge — Makefile
# Usage: `make help` to see all targets
# ============================================================

.PHONY: help install install-uv data eda baseline train evaluate app verbatims test lint format clean check-leaks

# Default target
help:
	@echo "NPS Prediction — available commands:"
	@echo ""
	@echo "  Setup"
	@echo "    install        Create venv and install dependencies (pip)"
	@echo "    install-uv     Same with uv (faster, requires uv installed)"
	@echo ""
	@echo "  Data"
	@echo "    data           Download raw data + build NPS target"
	@echo "    check-leaks    Audit features for correlation with label"
	@echo ""
	@echo "  Modeling"
	@echo "    baseline       Train baseline models (LR, ordinal, LGBM default)"
	@echo "    train          Train final tuned models"
	@echo "    evaluate       Evaluate all models on test set"
	@echo ""
	@echo "  Bonus"
	@echo "    verbatims      Generate synthetic customer verbatims (LLM)"
	@echo ""
	@echo "  App"
	@echo "    app            Launch Streamlit UI"
	@echo ""
	@echo "  Quality"
	@echo "    test           Run pytest suite"
	@echo "    lint           Lint with ruff"
	@echo "    format         Auto-format with ruff"
	@echo ""
	@echo "  Cleanup"
	@echo "    clean          Remove derived data, models, caches"

# ============================================================
# Setup
# ============================================================
install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip wheel
	. .venv/bin/activate && pip install -r requirements.txt
	@echo ""
	@echo "Done. Activate with: source .venv/bin/activate"

install-uv:
	uv venv
	uv pip install -r requirements.txt
	@echo ""
	@echo "Done. Activate with: source .venv/bin/activate"

# ============================================================
# Data pipeline
# ============================================================
data:
	python -m src.data.load
	python -m src.data.target

check-leaks:
	python -m src.data.audit_leaks

# ============================================================
# Modeling pipeline
# ============================================================
baseline:
	python -m src.models.baseline

train:
	python -m src.models.train

evaluate:
	python -m src.models.predict
	python -m src.evaluation.metrics

# ============================================================
# Bonus
# ============================================================
verbatims:
	python -m src.verbatims.generate

# ============================================================
# App
# ============================================================
app:
	streamlit run app/main.py

# ============================================================
# Quality
# ============================================================
test:
	pytest tests/ -v

lint:
	ruff check src/ tests/ app/

format:
	ruff format src/ tests/ app/

# ============================================================
# Cleanup
# ============================================================
clean:
	rm -rf data/interim/* data/processed/*
	rm -rf models/*.joblib models/*.pkl
	rm -rf reports/figures/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned."
