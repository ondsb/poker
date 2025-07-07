# Poker ML Project Makefile
# Use: make <target>

.PHONY: help install test clean run-app preprocess-focused preprocess-focused-sample train-focused train-focused-sample focused-pipeline status

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
	@echo "  preprocess-focused         Preprocess full 27GB dataset (500K chunks, optimized for larger files)"
	@echo "  preprocess-focused-sample  Preprocess sample dataset for testing"
	@echo ""
	@echo "Model Operations:"
	@echo "  train-focused              Train focused model with meaningful features only"
	@echo "  train-focused-sample       Train focused model on sample data"
	@echo ""
	@echo "Pipelines:"
	@echo "  focused-pipeline           Complete focused pipeline (focused preprocess + focused train)"
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

preprocess-focused:
	@echo "🎯 Starting focused preprocessing of full 27GB dataset (500K chunks, optimized for larger files)..."
	@mkdir -p data/processed/focused
	PYTHONPATH=. uv run python src/data/preprocess_focused.py \
		--input data/raw/poker_training_data.jsonl \
		--output data/processed/focused \
		--chunk-size 500000

preprocess-focused-sample:
	@echo "🧪 Starting focused preprocessing of sample dataset for testing..."
	@mkdir -p data/processed/focused
	PYTHONPATH=. uv run python src/data/preprocess_focused.py \
		--input data/raw/poker_training_data.jsonl \
		--output data/processed/focused \
		--chunk-size 1000000 \
		--max-chunks 10

# Model Operations
train-focused:
	@echo "🎯 Training focused poker model with meaningful features..."
	@mkdir -p models/focused
	PYTHONPATH=. uv run python src/models/train_focused.py \
		--data-dir data/processed/focused \
		--output-dir models/focused

train-focused-sample:
	@echo "🧪 Training focused model on sample data (first 10 chunks)..."
	@mkdir -p models/focused
	PYTHONPATH=. uv run python src/models/train_focused.py \
		--data-dir data/processed/focused \
		--output-dir models/focused \
		--max-chunks 10

# Pipelines
focused-pipeline:
	@echo "🎯 Running FOCUSED pipeline (focused preprocess + focused train)..."
	@echo "Step 1: Focused preprocessing with meaningful features only..."
	$(MAKE) preprocess-focused
	@echo "Step 2: Focused training with simplified model..."
	$(MAKE) train-focused
	@echo "✅ Focused pipeline completed! Clean, interpretable model ready."

# Application
run-app:
	@echo "Starting Streamlit app..."
	PYTHONPATH=. uv run streamlit run src/web/app.py

status:
	@echo "Project status:"
	PYTHONPATH=. uv run python scripts/project_status.py 