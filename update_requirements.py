#!/usr/bin/env python3
"""
Requirements Updater for Document Forgery Detection System.

This script updates the requirements.txt file to ensure compatibility with Apple Silicon.
"""

import os
import sys
import platform
import re
from typing import Dict, List, Tuple

# Apple Silicon compatible versions
APPLE_SILICON_VERSIONS = {
    "numpy": ">=1.21.0",
    "scipy": ">=1.8.0",
    "tensorflow": ">=2.9.0",
    "torch": ">=1.12.0",
    "opencv-python": ">=4.6.0",
    "scikit-learn": ">=1.0.0",
    "pandas": ">=1.4.0"
}

# Python 3.13 compatible versions
PYTHON_313_VERSIONS = {
    "pydantic": ">=2.0.0",
    "fastapi": ">=0.100.0",
    "uvicorn": ">=0.23.0"
}

# Package replacements
PACKAGE_REPLACEMENTS = {
    "apache-tika": "tika-python>=2.6.0"
}

def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"

def is_python_313_or_higher() -> bool:
    """Check if running on Python 3.13 or higher."""
    python_version = tuple(map(int, platform.python_version().split(".")))
    return python_version >= (3, 13, 0)

def update_requirements(requirements_file: str) -> None:
    """Update requirements.txt file for compatibility."""
    if not os.path.exists(requirements_file):
        print(f"Error: Requirements file not found: {requirements_file}")
        return
    
    # Read requirements file
    with open(requirements_file, 'r') as f:
        lines = f.readlines()
    
    # Update requirements
    updated_lines = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            updated_lines.append(line)
            continue
        
        # Check for package replacements
        replaced = False
        for old_pkg, new_pkg in PACKAGE_REPLACEMENTS.items():
            if line.startswith(old_pkg):
                updated_lines.append(new_pkg)
                print(f"Replaced {line} -> {new_pkg}")
                replaced = True
                break
        
        if replaced:
            continue
        
        # Parse package name and version
        match = re.match(r'^([a-zA-Z0-9_.-]+)([<>=!~].+)?$', line)
        if not match:
            updated_lines.append(line)
            continue
        
        package_name = match.group(1).lower()
        version_spec = match.group(2) or ""
        
        # Check if package needs updating for Apple Silicon
        if is_apple_silicon() and package_name in APPLE_SILICON_VERSIONS:
            # Replace version specification
            updated_line = f"{package_name}{APPLE_SILICON_VERSIONS[package_name]}"
            updated_lines.append(updated_line)
            print(f"Updated for Apple Silicon: {line} -> {updated_line}")
        # Check if package needs updating for Python 3.13
        elif is_python_313_or_higher() and package_name in PYTHON_313_VERSIONS:
            # Replace version specification
            updated_line = f"{package_name}{PYTHON_313_VERSIONS[package_name]}"
            updated_lines.append(updated_line)
            print(f"Updated for Python 3.13: {line} -> {updated_line}")
        else:
            updated_lines.append(line)
    
    # Write updated requirements file
    with open(requirements_file, 'w') as f:
        f.write('\n'.join(updated_lines))
    
    print(f"Updated requirements file: {requirements_file}")

def main() -> None:
    """Main function."""
    # Get requirements file path
    if len(sys.argv) > 1:
        requirements_file = sys.argv[1]
    else:
        requirements_file = "document_forgery_detection/requirements.txt"
    
    print(f"Updating requirements file for compatibility: {requirements_file}")
    
    # Check system
    if is_apple_silicon():
        print("Detected Apple Silicon. Updating requirements for Apple Silicon compatibility.")
    
    if is_python_313_or_higher():
        print("Detected Python 3.13 or higher. Updating requirements for Python 3.13 compatibility.")
    
    update_requirements(requirements_file)

if __name__ == "__main__":
    main() 