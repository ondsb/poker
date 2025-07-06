# Poker ML Project Makefile
# Use: make <target>

.PHONY: help install test clean run-app preprocess-optimized preprocess-optimized-sample train-model train-optimized-quick sample-pipeline full-pipeline status

# Default target
help:
	@echo "Poker ML Project - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  install          Install dependencies"
	@echo "  clean           Clean up temporary files"
	@echo ""
	@echo "Testing:"
	@echo "  test            Run all tests"
	@echo ""
	@echo "Data Processing:"
	@echo "  preprocess-optimized Preprocess full 27GB dataset"
	@echo "  preprocess-optimized-sample Preprocess sample dataset for testing"
	@echo ""
	@echo "Model Operations:"
	@echo "  train-model     Train model on optimized data (50 chunks)"
	@echo "  train-optimized-quick Train model on subset (10 chunks) for testing"
	@echo ""
	@echo "Pipelines:"
	@echo "  sample-pipeline Complete sample pipeline (sample preprocess + quick train)"
	@echo "  full-pipeline   Complete full pipeline (full preprocess + full train)"
	@echo ""
	@echo "Application:"
	@echo "  run-app         Run Streamlit web app"
	@echo "  status          Show project status"
	@echo ""

# Setup
install:
	@echo "Installing dependencies..."
	uv pip install -r requirements.txt
	@echo "✅ Dependencies installed"

clean:
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	find . -type f -name "batch_*.jsonl" -delete
	@echo "✅ Cleanup complete"

# Testing
test:
	@echo "Running tests..."
	PYTHONPATH=. uv run pytest tests/ -v

# Data Processing
preprocess-optimized:
	@echo "🚀 Starting optimized preprocessing of full 27GB dataset with hole cards from showdown actions..."
	@mkdir -p data/processed/optimized
	PYTHONPATH=. uv run python src/data/preprocess_optimized.py \
		--input data/raw/poker_training_data.jsonl \
		--output data/processed/optimized \
		--chunk-size 100000

preprocess-optimized-sample:
	@echo "🧪 Starting optimized preprocessing of sample dataset for testing..."
	@mkdir -p data/processed/optimized
	PYTHONPATH=. uv run python src/data/preprocess_optimized.py \
		--input data/raw/poker_training_data.jsonl \
		--output data/processed/optimized \
		--chunk-size 100000 \
		--max-chunks 5

# Model Operations
train-model:
	@echo "Training model on optimized data..."
	@mkdir -p models
	PYTHONPATH=. uv run python src/models/train_model_optimized.py \
		--data-pattern "data/processed/optimized/chunk_*.parquet" \
		--output-dir models \
		--max-chunks 50

train-optimized-quick:
	@echo "⚡ Quick training on subset of data for testing..."
	@mkdir -p models
	PYTHONPATH=. uv run python src/models/train_model_optimized.py \
		--data-pattern "data/processed/optimized/chunk_*.parquet" \
		--output-dir models \
		--max-chunks 10

# Pipelines
sample-pipeline:
	@echo "🎯 Running SAMPLE pipeline (sample preprocess + quick train)..."
	@echo "Step 1: Preprocessing sample data..."
	$(MAKE) preprocess-optimized-sample
	@echo "Step 2: Quick training on subset..."
	$(MAKE) train-optimized-quick
	@echo "✅ Sample pipeline completed! Model ready for testing."

full-pipeline:
	@echo "🚀 Running FULL pipeline (full preprocess + full train)..."
	@echo "Step 1: Preprocessing full dataset..."
	$(MAKE) preprocess-optimized
	@echo "Step 2: Full training on large dataset..."
	$(MAKE) train-model
	@echo "✅ Full pipeline completed! Production model ready."

# Application
run-app:
	@echo "Starting Streamlit app..."
	PYTHONPATH=. uv run streamlit run src/web/app.py

status:
	@echo "Project status:"
	PYTHONPATH=. uv run python scripts/project_status.py 