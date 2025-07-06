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
	@echo "  cleanup-unused  Clean up unused directories and files"
	@echo "  cleanup-all     Complete cleanup (temp files + unused files)"
	@echo ""
	@echo "Testing:"
	@echo "  test            Run all tests"
	@echo ""
	@echo "Data Processing:"
	@echo "  preprocess Preprocess full 27GB dataset (conservative, 10K chunks, 2 cores)"
	@echo "  preprocess-sample Preprocess sample dataset for testing"
	@echo ""
	@echo "Model Operations:"
	@echo "  train-model     Train poker win probability model on conservative data"
	@echo "  train-model-sample Train model on sample data (2 chunks) for testing"
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

cleanup-unused:
	@echo "🧹 Cleaning up unused directories and files..."
	PYTHONPATH=. uv run python scripts/cleanup_unused.py
	@echo "✅ Unused files cleanup complete"

cleanup-all: clean cleanup-unused
	@echo "🧹 Complete cleanup finished"

# Testing
test:
	@echo "Running tests..."
	PYTHONPATH=. uv run pytest tests/ -v

preprocess:
	@echo "🚀 Starting conservative preprocessing of full 27GB dataset (5K chunks, 1 core)..."
	@mkdir -p data/processed/conservative
	PYTHONPATH=. uv run python src/data/preprocess_conservative.py \
		--input data/raw/poker_training_data.jsonl \
		--output data/processed/conservative \
		--chunk-size 100000 \
		--max-workers 8

preprocess-sample:
	@echo "🧪 Starting conservative preprocessing of sample dataset for testing..."
	@mkdir -p data/processed/conservative
	PYTHONPATH=. uv run python src/data/preprocess_conservative.py \
		--input data/raw/poker_training_data.jsonl \
		--output data/processed/conservative \
		--chunk-size 10000 \
		--max-workers 8 \
		--max-chunks 5

# Model Operations
train-model:
	@echo "🤖 Training poker win probability model on conservative data..."
	@mkdir -p models/conservative
	PYTHONPATH=. uv run python src/models/train_conservative.py \
		--data-dir data/processed/conservative \
		--output-dir models/conservative

train-model-sample:
	@echo "🧪 Training model on sample data (first 2 chunks)..."
	@mkdir -p models/conservative
	PYTHONPATH=. uv run python src/models/train_conservative.py \
		--data-dir data/processed/conservative \
		--output-dir models/conservative \
		--max-chunks 2

# Pipelines
sample-pipeline:
	@echo "🎯 Running SAMPLE pipeline (sample preprocess + quick train)..."
	@echo "Step 1: Preprocessing sample data..."
	$(MAKE) preprocess-sample
	@echo "Step 2: Quick training on subset..."
	$(MAKE) train-model-sample
	@echo "✅ Sample pipeline completed! Model ready for testing."

full-pipeline:
	@echo "🚀 Running FULL pipeline (full preprocess + full train)..."
	@echo "Step 1: Preprocessing full dataset..."
	$(MAKE) preprocess
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