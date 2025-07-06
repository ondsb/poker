#!/usr/bin/env python3
"""
Cleanup script to remove unused directories and files.
"""

import os
import shutil
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_directory_size(path):
    """Get directory size in MB."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / (1024 * 1024)  # Convert to MB

def cleanup_unused_directories():
    """Clean up unused directories and files."""
    logger.info("🧹 Starting cleanup of unused directories and files...")
    
    # Directories to remove (test/development directories)
    directories_to_remove = [
        "data/processed/unified_test",
        "data/processed/unified_test_fixed", 
        "data/processed/unified_27gb_test",
        "data/processed/massive_streaming",
        "data/processed/massive_parallel",
        "data/processed/player_centric",
        "data/intermediate",
        "data/validation_results",
        "catboost_info",
        ".pytest_cache",
        ".vscode",
        "__pycache__",
        "src/__pycache__",
        "src/core/__pycache__",
        "src/data/__pycache__",
        "src/models/__pycache__",
        "src/utils/__pycache__",
        "src/web/__pycache__",
        "tests/__pycache__",
        "scripts/__pycache__",
        "scripts/analysis/__pycache__",
        "scripts/visualization/__pycache__"
    ]
    
    # Files to remove
    files_to_remove = [
        "preprocessing.log",
        "test_preprocess_debug2.log",
        "test_preprocess_debug.log", 
        "test_preprocess.log",
        "test_app_enhanced.py",
        "test_app_features.py",
        "test_random_generation.py",
        "test_chunk_resume.py",
        "model_validation_results.png",
        "new_model_validation_results.png",
        "poker_xgb_model_improved.joblib"
    ]
    
    total_freed = 0
    
    # Remove directories
    for dir_path in directories_to_remove:
        if os.path.exists(dir_path):
            try:
                size_before = get_directory_size(dir_path)
                shutil.rmtree(dir_path)
                total_freed += size_before
                logger.info(f"🗑️ Removed directory: {dir_path} ({size_before:.1f} MB)")
            except Exception as e:
                logger.warning(f"⚠️ Could not remove {dir_path}: {e}")
    
    # Remove files
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                size_before = os.path.getsize(file_path) / (1024 * 1024)  # MB
                os.remove(file_path)
                total_freed += size_before
                logger.info(f"🗑️ Removed file: {file_path} ({size_before:.1f} MB)")
            except Exception as e:
                logger.warning(f"⚠️ Could not remove {file_path}: {e}")
    
    # Clean up any remaining __pycache__ directories
    for root, dirs, files in os.walk("."):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                cache_path = os.path.join(root, dir_name)
                try:
                    size_before = get_directory_size(cache_path)
                    shutil.rmtree(cache_path)
                    total_freed += size_before
                    logger.info(f"🗑️ Removed cache: {cache_path} ({size_before:.1f} MB)")
                except Exception as e:
                    logger.warning(f"⚠️ Could not remove {cache_path}: {e}")
    
    # Clean up .pyc files
    for root, dirs, files in os.walk("."):
        for file_name in files:
            if file_name.endswith(".pyc"):
                pyc_path = os.path.join(root, file_name)
                try:
                    size_before = os.path.getsize(pyc_path) / (1024 * 1024)  # MB
                    os.remove(pyc_path)
                    total_freed += size_before
                    logger.info(f"🗑️ Removed .pyc: {pyc_path} ({size_before:.1f} MB)")
                except Exception as e:
                    logger.warning(f"⚠️ Could not remove {pyc_path}: {e}")
    
    logger.info(f"✅ Cleanup complete! Freed {total_freed:.1f} MB of space")
    
    # Show current disk usage
    show_disk_usage()

def show_disk_usage():
    """Show current disk usage of key directories."""
    logger.info("📊 Current disk usage:")
    
    key_dirs = [
        "data/raw",
        "data/processed",
        "models",
        "src",
        "tests",
        "scripts"
    ]
    
    for dir_path in key_dirs:
        if os.path.exists(dir_path):
            size_mb = get_directory_size(dir_path)
            logger.info(f"   {dir_path}: {size_mb:.1f} MB")

if __name__ == "__main__":
    cleanup_unused_directories() 