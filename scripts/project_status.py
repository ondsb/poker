#!/usr/bin/env python3
"""
Project Status Report
Shows the current state of the improved poker prediction system.
"""

import os
from pathlib import Path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.utils import (
    get_config_summary,
    validate_paths,
    get_project_info,
    get_file_size_mb,
    count_jsonl_lines,
)


def print_header(title):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'-'*40}")
    print(f"  {title}")
    print(f"{'-'*40}")


def main():
    """Generate comprehensive project status report."""
    print_header("♠️ POKER ML PROJECT STATUS REPORT")

    # Project Configuration
    print_section("📋 PROJECT CONFIGURATION")
    config_summary = get_config_summary()
    print(f"Project Root: {config_summary['project_root']}")
    print(f"Data Directory: {config_summary['data_dir']}")
    print(f"Models Directory: {config_summary['models_dir']}")

    # Path Validation
    print_section("🔍 PATH VALIDATION")
    paths = validate_paths()
    for path_name, exists in paths.items():
        status = "✅" if exists else "❌"
        print(f"{status} {path_name}: {exists}")

    # Project Info
    print_section("📊 PROJECT STATUS")
    project_info = get_project_info()
    print(f"Status: {project_info['status'].upper()}")

    # Model Files
    print_section("🤖 MODEL FILES")
    for model_type, info in project_info["model_files"].items():
        if info.get("exists"):
            size_mb = info.get("size_mb", 0)
            print(f"✅ {model_type.title()} Model: {size_mb:.1f} MB")
        else:
            print(f"❌ {model_type.title()} Model: Not found")

    # Data Files
    print_section("📁 DATA FILES")
    for data_type, info in project_info["data_files"].items():
        if info.get("exists"):
            size_mb = info.get("size_mb", 0)
            lines = info.get("lines", 0)
            print(f"✅ {data_type.replace('_', ' ').title()}: {size_mb:.1f} MB, {lines:,} lines")
        else:
            print(f"❌ {data_type.replace('_', ' ').title()}: Not found")

    # File Structure
    print_section("📂 FILE STRUCTURE")
    core_files = [
        "app.py",
        "config.py",
        "utils.py",
        "load_data_eng.py",
        "train_model_improved.py",
        "validate_model.py",
        "test_multiplayer_prediction.py",
        "prepare_training_data.py",
        "data_processing.py",
        "extract_hole_cards.py",
        "check_dataset_quality.py",
        "feature_distribution_check.py",
        "test_pipeline.py",
        "test_improvements.py",
        "project_status.py",
        "README.md",
        "requirements.txt",
    ]

    total_size = 0
    for file in core_files:
        if os.path.exists(file):
            size_mb = get_file_size_mb(file)
            total_size += size_mb
            print(f"✅ {file}: {size_mb:.2f} MB")
        else:
            print(f"❌ {file}: Missing")

    print(f"\n📊 Total Core Files Size: {total_size:.2f} MB")

    # Improvements Summary
    print_section("🚀 IMPROVEMENTS SUMMARY")
    improvements = [
        "✅ Removed redundant files (predict.py, load_data_big.py)",
        "✅ Created centralized configuration (config.py)",
        "✅ Consolidated utilities (utils.py)",
        "✅ Added comprehensive error handling",
        "✅ Improved type hints throughout",
        "✅ Enhanced logging and debugging",
        "✅ Better data validation",
        "✅ Professional documentation (README.md)",
        "✅ Dependency management (requirements.txt)",
        "✅ Testing framework (test_improvements.py)",
        "✅ Modular architecture",
        "✅ Configuration-driven approach",
    ]

    for improvement in improvements:
        print(improvement)

    # Next Steps
    print_section("🎯 NEXT STEPS")
    next_steps = [
        "1. Run the web app: uv run streamlit run app.py",
        "2. Train model (if needed): uv run python train_model_improved.py",
        "3. Validate model: uv run python validate_model.py",
        "4. Test multi-player: uv run python test_multiplayer_prediction.py",
    ]

    for step in next_steps:
        print(step)

    # Recommendations
    print_section("💡 RECOMMENDATIONS")
    recommendations = []

    if not project_info["model_files"].get("improved", {}).get("exists"):
        recommendations.append("🔧 Train the improved model for better predictions")

    if not project_info["data_files"].get("with_hole_cards", {}).get("exists"):
        recommendations.append("📊 Prepare training data with hole cards")

    if not recommendations:
        recommendations.append("🎉 Project is ready to use!")

    for rec in recommendations:
        print(rec)

    print_header("🎉 PROJECT STATUS COMPLETE")


if __name__ == "__main__":
    main()
