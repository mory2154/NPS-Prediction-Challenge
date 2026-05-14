# ============================================================
# NPS Prediction Challenge — Makefile (Phase 13 patch v2: build-alerts)
# ============================================================

.PHONY: help install install-uv data check-leaks build-dataset build-splits \
        build-features prepare-prompts load-verbatims inspect-verbatims \
        compute-embeddings \
        baseline baseline-quick tune tune-hybrid final-eval interpret fairness \
        batch-score recalibrate simulate-drift simulate-drift-calibrated \
        build-alerts \
        train evaluate app verbatims \
        test test-phase2 test-phase3 test-phase4 test-phase5 test-phase6 \
        test-phase7 test-phase8 test-phase9 test-phase10 test-phase11 \
        test-phase12 test-phase13 \
        lint format clean

help:
	@echo "NPS Prediction — available commands:"
	@echo ""
	@echo "  Modeling"
	@echo "    baseline       Phase 6"
	@echo "    tune           Phase 7"
	@echo "    tune-hybrid    Phase 8"
	@echo "    final-eval     Phase 9"
	@echo "    interpret      Phase 10"
	@echo "    fairness       Phase 11"
	@echo "    batch-score    Phase 12 (pre-compute predictions for the Streamlit app)"
	@echo ""
	@echo "  Monitoring (Phase 13)"
	@echo "    recalibrate                 Refit C2 with CalibratedClassifierCV(isotonic)"
	@echo "    simulate-drift              FIFO monthly drift simulation + build alerts"
	@echo "    simulate-drift-calibrated   Idem with the recalibrated C2 + build alerts"
	@echo "    build-alerts                Build monitoring_alerts.parquet (auto-called by simulate-drift)"
	@echo ""
	@echo "  App"
	@echo "    app            Launch the Streamlit UI"
	@echo ""
	@echo "  Tests"
	@echo "    test           Full pytest suite"
	@echo "    test-phaseN    Phase N only (N=2..13)"

install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip wheel
	. .venv/bin/activate && pip install -r requirements.txt

install-uv:
	uv venv && uv pip install -r requirements.txt

data:
	python -m src.data.load
check-leaks:
	python -m src.data.audit_leaks
build-dataset:
	python -m src.data.build_dataset
build-splits:
	python -m src.data.split
build-features:
	python -m src.features.build

prepare-prompts:
	python -m src.verbatims.prepare_prompts
load-verbatims:
	python -m src.verbatims.load_verbatims
inspect-verbatims:
	python -m src.verbatims.inspect
compute-embeddings:
	python -m src.features.embeddings

baseline:
	python -m src.models.baseline
baseline-quick:
	python -m src.models.baseline --quick
tune:
	python -m src.models.tuning
tune-hybrid:
	python -m src.models.tuning_hybrid
final-eval:
	python -m src.evaluation.final_eval
interpret:
	python -m src.interpretation.shap_utils
	python -m src.interpretation.linear_coef
	python -m src.interpretation.pca_loadings
fairness:
	python -m src.fairness.audit
batch-score:
	python -m src.inference.batch_score

# Phase 13 monitoring
recalibrate:
	python -m src.monitoring.recalibrate
simulate-drift:
	python -m src.monitoring.drift_simulator
	python -m src.monitoring.alerts
simulate-drift-calibrated:
	python -m src.monitoring.drift_simulator --use-calibrated
	python -m src.monitoring.alerts
build-alerts:
	python -m src.monitoring.alerts

train: tune
evaluate:
	python -m src.models.predict
	python -m src.evaluation.metrics

app:
	streamlit run app/main.py

test:
	pytest tests/ -v
test-phase2:
	pytest tests/test_phase2.py -v
test-phase3:
	pytest tests/test_phase3.py -v
test-phase4:
	pytest tests/test_phase4.py -v
test-phase5:
	pytest tests/test_phase5.py -v
test-phase6:
	pytest tests/test_phase6.py -v
test-phase7:
	pytest tests/test_phase7.py -v
test-phase8:
	pytest tests/test_phase8.py -v
test-phase9:
	pytest tests/test_phase9.py -v
test-phase10:
	pytest tests/test_phase10.py -v
test-phase11:
	pytest tests/test_phase11.py -v
test-phase12:
	pytest tests/test_phase12.py -v
test-phase13:
	pytest tests/test_phase13.py -v
lint:
	ruff check src/ tests/ app/
format:
	ruff format src/ tests/ app/

clean:
	rm -rf data/interim/* data/processed/*
	rm -rf models/*.joblib models/baselines/*.joblib models/tuned/*.joblib
	rm -rf models/hybrid/*.joblib models/results/*.parquet
	rm -rf reports/figures/* reports/*.md
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned."
