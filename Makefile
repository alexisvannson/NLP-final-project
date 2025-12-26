.PHONY: help install setup lint format test download-data preprocess train-baseline train-all evaluate demo docker-build docker-run clean

PYTHON := python3
VENV := venv
VENV_BIN := $(VENV)/bin
PIP := $(VENV_BIN)/pip
PYTHON_VENV := $(VENV_BIN)/python
PROJECT_ROOT := $(shell pwd)
export PYTHONPATH := $(PROJECT_ROOT)

help:
	@echo "Targets:"
	@echo "  install         : Install dependencies"
	@echo "  setup           : Project setup (venv + deps + data)"
	@echo "  lint, format    : Code checks/formatting"
	@echo "  test            : Run tests"
	@echo "  download-data   : Download datasets"
	@echo "  preprocess      : Preprocess datasets"
	@echo "  setup-models    : Download models (no finetune)"
	@echo "  finetune-all    : Fine-tune all models"
	@echo "  compare         : All model comparison (50 samples)"
	@echo "  compare-quick   : Fast extractive-only compare (10 samples)"
	@echo "  compare-full    : Full comparison (100 samples)"
	@echo "  analyze-length(.*) : Length-degradation analysis"
	@echo "  analyze-sections(.*) : Section structure analysis"
	@echo "  demo            : Launch Streamlit demo"
	@echo "  docker-*        : Docker commands"
	@echo "  clean           : Clean generated files"

install:
	@echo "Installing dependencies..."
	test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install "numpy<2"
	$(PIP) install -r requirements.txt
	$(PYTHON_VENV) -m spacy download en_core_web_sm
	$(PYTHON_VENV) -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"



lint:
	@echo "Linting code..."
	$(VENV_BIN)/black --check models/ src/ tests/ app.py || true
	$(VENV_BIN)/flake8 models/ src/ tests/ app.py --max-line-length=100 --extend-ignore=E203,W503 || true
	$(VENV_BIN)/mypy models/ src/ --ignore-missing-imports || true
	@echo "Lint done!"

format:
	@echo "Formatting code..."
	$(VENV_BIN)/isort models/ src/ tests/ app.py data/scripts/
	$(VENV_BIN)/black models/ src/ tests/ app.py data/scripts/
	@echo "Code formatted successfully!"

test:
	@echo "Running tests..."
	$(VENV_BIN)/pytest tests/ -v --cov=src --cov=models --cov-report=html --cov-report=term
	@echo "Tests complete!"

download-data:
	@echo "Downloading datasets..."
	$(PYTHON_VENV) data/scripts/download_datasets.py

preprocess:
	@echo "Preprocessing datasets..."
	$(PYTHON_VENV) data/scripts/preprocess.py

setup-models:
	@echo "Downloading pretrained models (no fine-tuning)..."
	@sed -i.bak 's/skip_fine_tuning: false/skip_fine_tuning: true/' configs/baseline.yaml configs/hierarchical.yaml configs/longformer.yaml
	@$(PYTHON_VENV) src/training.py --config configs/baseline.yaml
	@$(PYTHON_VENV) src/training.py --config configs/hierarchical.yaml
	@$(PYTHON_VENV) src/training.py --config configs/longformer.yaml
	@mv configs/baseline.yaml.bak configs/baseline.yaml 2>/dev/null || true
	@mv configs/hierarchical.yaml.bak configs/hierarchical.yaml 2>/dev/null || true
	@mv configs/longformer.yaml.bak configs/longformer.yaml 2>/dev/null || true
	@echo "✓ Pretrained models are ready!"
	@echo "To fine-tune, run: make finetune-all"

finetune-all:
	@echo "Fine-tuning all models..."
	@echo "Requires: data/processed/, ~4GB disk, GPU recommended, several hours."
	@read -p "Continue? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		sed -i.bak 's/skip_fine_tuning: true/skip_fine_tuning: false/' configs/baseline.yaml configs/hierarchical.yaml configs/longformer.yaml && \
		$(PYTHON_VENV) src/training.py --config configs/baseline.yaml && \
		$(PYTHON_VENV) src/training.py --config configs/hierarchical.yaml && \
		$(PYTHON_VENV) src/training.py --config configs/longformer.yaml && \
		mv configs/baseline.yaml.bak configs/baseline.yaml 2>/dev/null || true && \
		mv configs/hierarchical.yaml.bak configs/hierarchical.yaml 2>/dev/null || true && \
		mv configs/longformer.yaml.bak configs/longformer.yaml 2>/dev/null || true; \
		echo "✓ Models fine-tuned! Saved in models/checkpoints/"; \
	else \
		echo "Cancelled."; \
	fi

compare:
	@echo "Comparing all 6 models on 50 samples..."
	@if [ ! -d "data/processed/arxiv" ]; then \
		echo "ERROR: Run 'make preprocess'."; \
		echo "Quick start: make download-data; make preprocess; make compare"; \
		exit 1; \
	fi
	$(PYTHON_VENV) scripts/compare_models.py \
		--dataset arxiv \
		--num-samples 50 \
		--output-dir results/comparison \
		--device cpu
	@echo "✓ Comparison done."
	@echo "Results: results/comparison/"

compare-quick:
	@echo "Quick model comparison (10 samples, extractive only)..."
	$(PYTHON_VENV) scripts/compare_models.py \
		--dataset arxiv \
		--num-samples 10 \
		--output-dir results/comparison_quick \
		--models textrank lexrank \
		--metrics rouge \
		--device cpu

compare-full:
	@echo "Full model comparison (100 samples)..."
	@read -p "May take 30-60 min. Continue? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(PYTHON_VENV) scripts/compare_models.py \
			--dataset arxiv \
			--num-samples 100 \
			--output-dir results/comparison_full \
			--device cpu; \
	fi

analyze-length:
	@echo "Analyzing model performance by document length..."
	@if [ ! -d "data/processed/arxiv" ]; then \
		echo "ERROR: Run 'make preprocess'."; \
		exit 1; \
	fi
	$(PYTHON_VENV) scripts/analyze_length_degradation.py \
		--dataset arxiv \
		--num-samples 100 \
		--output-dir results/length_analysis \
		--device cpu
	@echo "✓ Length analysis done. See results/length_analysis/"

analyze-length-quick:
	@echo "Quick length analysis (50 samples, extractive only)..."
	$(PYTHON_VENV) scripts/analyze_length_degradation.py \
		--dataset arxiv \
		--num-samples 50 \
		--output-dir results/length_analysis_quick \
		--models textrank lexrank \
		--metrics rouge \
		--device cpu

analyze-length-full:
	@echo "Comprehensive length analysis (200 samples)..."
	@read -p "May take 60-90 min. Continue? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(PYTHON_VENV) scripts/analyze_length_degradation.py \
			--dataset arxiv \
			--num-samples 200 \
			--output-dir results/length_analysis_full \
			--device cpu; \
	fi

analyze-sections:
	@echo "Analyzing document section structure (50 samples)..."
	@if [ ! -d "data/processed/arxiv" ]; then \
		echo "ERROR: No processed data found. Please run 'make preprocess' first."; \
		exit 1; \
	fi
	$(PYTHON_VENV) scripts/analyze_sections.py \
		--dataset arxiv \
		--num-samples 50 \
		--output-dir results/section_analysis \
		--compare-models \
		--device cpu
	@echo "✓ Section analysis done. See results/section_analysis/"

analyze-sections-quick:
	@echo "Quick section analysis (30 samples)..."
	$(PYTHON_VENV) scripts/analyze_sections.py \
		--dataset arxiv \
		--num-samples 30 \
		--output-dir results/section_analysis_quick \
		--device cpu

evaluate:
	@echo "Running evaluation suite..."
	$(PYTHON_VENV) src/evaluation.py --all-models

demo:
	@echo "Launching Streamlit demo..."
	$(VENV_BIN)/streamlit run app.py

docker-build:
	@echo "Building Docker image..."
	docker build -t long-doc-summarization:latest .

docker-run:
	@echo "Running Docker container..."
	docker run -p 8501:8501 long-doc-summarization:latest

clean:
	@echo "Cleaning up..."
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} + || true
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache
	rm -rf build dist
	@echo "Cleanup complete!"

