#!/usr/bin/env python3
"""
Compatibility Checker for Document Forgery Detection System.

This script checks if all required libraries are compatible with the current system,
with special attention to Apple Silicon (M1/M2/M3/M4) compatibility.
"""

import sys
import platform
import subprocess
import pkg_resources
from typing import Dict, List, Tuple, Optional
import os

# ANSI colors for output
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
NC = '\033[0m'  # No Color

def print_colored(text: str, color: str) -> None:
    """Print colored text."""
    print(f"{color}{text}{NC}")

def check_system() -> Dict[str, str]:
    """Check system information."""
    system_info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }
    
    # Check if running on Apple Silicon
    if system_info["os"] == "Darwin" and system_info["architecture"] == "arm64":
        system_info["is_apple_silicon"] = "True"
    else:
        system_info["is_apple_silicon"] = "False"
    
    return system_info

def get_installed_packages() -> Dict[str, str]:
    """Get all installed packages and their versions."""
    return {pkg.key: pkg.version for pkg in pkg_resources.working_set}

def check_package_compatibility(packages: Dict[str, str], system_info: Dict[str, str]) -> List[Dict[str, str]]:
    """Check if packages are compatible with the current system."""
    results = []
    
    # Known problematic packages on Apple Silicon
    apple_silicon_issues = {
        "tensorflow": "< 2.9.0",  # TensorFlow < 2.9.0 has issues on Apple Silicon
        "torch": "< 1.12.0",      # PyTorch < 1.12.0 has issues on Apple Silicon
        "opencv-python": "< 4.6.0",  # OpenCV < 4.6.0 has issues on Apple Silicon
        "scipy": "< 1.8.0",       # SciPy < 1.8.0 has issues on Apple Silicon
        "numpy": "< 1.21.0",      # NumPy < 1.21.0 has issues on Apple Silicon
    }
    
    # Python 3.13 compatibility issues
    python_313_issues = {
        "pydantic": "< 2.0.0",    # pydantic < 2.0.0 has issues with Python 3.13
        "fastapi": "< 0.100.0",   # fastapi < 0.100.0 has issues with Python 3.13
        "uvicorn": "< 0.23.0",    # uvicorn < 0.23.0 has issues with Python 3.13
    }
    
    python_version = tuple(map(int, system_info["python_version"].split(".")))
    is_python_313_or_higher = python_version >= (3, 13, 0)
    
    for package, version in packages.items():
        result = {
            "package": package,
            "installed_version": version,
            "status": "OK",
            "message": ""
        }
        
        # Check for Apple Silicon compatibility issues
        if system_info["is_apple_silicon"] == "True" and package in apple_silicon_issues:
            problematic_version = apple_silicon_issues[package]
            if problematic_version.startswith("< "):
                min_version = problematic_version[2:]
                if pkg_resources.parse_version(version) < pkg_resources.parse_version(min_version):
                    result["status"] = "WARNING"
                    result["message"] = f"Version {version} may have issues on Apple Silicon. Consider upgrading to {min_version}+"
            elif problematic_version.startswith("> "):
                max_version = problematic_version[2:]
                if pkg_resources.parse_version(version) > pkg_resources.parse_version(max_version):
                    result["status"] = "WARNING"
                    result["message"] = f"Version {version} may have issues on Apple Silicon. Consider downgrading to {max_version}"
        
        # Check for Python 3.13 compatibility issues
        if is_python_313_or_higher and package in python_313_issues:
            problematic_version = python_313_issues[package]
            if problematic_version.startswith("< "):
                min_version = problematic_version[2:]
                if pkg_resources.parse_version(version) < pkg_resources.parse_version(min_version):
                    result["status"] = "WARNING"
                    result["message"] = f"Version {version} may have issues with Python 3.13. Consider upgrading to {min_version}+"
        
        # Special case for apache-tika vs tika-python
        if package == "apache-tika":
            result["status"] = "WARNING"
            result["message"] = "apache-tika may have compatibility issues. Consider using tika-python instead."
        
        results.append(result)
    
    return results

def check_required_packages(required_packages: List[str]) -> List[Dict[str, str]]:
    """Check if required packages are installed."""
    installed_packages = get_installed_packages()
    results = []
    
    # Package aliases (alternative packages that provide the same functionality)
    package_aliases = {
        "apache-tika": ["tika-python", "tika"],
    }
    
    for package in required_packages:
        # Check if package or any of its aliases are installed
        aliases = package_aliases.get(package, [])
        installed_alias = None
        
        if package in installed_packages:
            results.append({
                "package": package,
                "installed_version": installed_packages[package],
                "status": "OK",
                "message": "Installed"
            })
        else:
            # Check for aliases
            for alias in aliases:
                if alias in installed_packages:
                    installed_alias = alias
                    results.append({
                        "package": f"{package} (using {alias})",
                        "installed_version": installed_packages[alias],
                        "status": "OK",
                        "message": f"Using alternative package: {alias}"
                    })
                    break
            
            # If no alias found, report as not installed
            if not installed_alias:
                results.append({
                    "package": package,
                    "installed_version": "Not installed",
                    "status": "ERROR",
                    "message": "Not installed" + (f" (alternatives: {', '.join(aliases)})" if aliases else "")
                })
    
    return results

def check_spacy_models() -> List[Dict[str, str]]:
    """Check if required spaCy models are installed."""
    results = []
    
    # List of required spaCy models
    required_models = ["en_core_web_sm", "en_core_web_md"]
    
    for model in required_models:
        try:
            # Try to import the model
            __import__(model)
            results.append({
                "package": f"spacy-model:{model}",
                "installed_version": "Installed",
                "status": "OK",
                "message": "Installed"
            })
        except ImportError:
            results.append({
                "package": f"spacy-model:{model}",
                "installed_version": "Not installed",
                "status": "ERROR",
                "message": "Not installed. Run: python -m spacy download " + model
            })
    
    return results

def check_openssl_version() -> Dict[str, str]:
    """Check OpenSSL version."""
    result = {
        "package": "openssl",
        "installed_version": "Unknown",
        "status": "UNKNOWN",
        "message": "Could not determine OpenSSL version"
    }
    
    try:
        # Try to get OpenSSL version
        import ssl
        openssl_version = ssl.OPENSSL_VERSION
        result["installed_version"] = openssl_version
        result["status"] = "OK"
        result["message"] = ""
    except:
        pass
    
    return result

def main() -> None:
    """Main function."""
    print_colored("Checking system compatibility for Document Forgery Detection System", GREEN)
    print()
    
    # Check system information
    system_info = check_system()
    print_colored("System Information:", GREEN)
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    print()
    
    # Required packages for Document Forgery Detection System
    required_packages = [
        "PyPDF2",
        "python-docx",
        "tika-python",  # Changed from apache-tika to tika-python
        "pdfminer.six",
        "spacy",
        "scikit-learn",
        "nltk",
        "transformers",
        "Pillow",
        "exifread",
        "opencv-python",
        "cryptography",
        "pyOpenSSL",
        "fastapi",
        "uvicorn",
        "pydantic",
        "pandas",
        "numpy"
    ]
    
    # Check required packages
    print_colored("Checking required packages:", GREEN)
    package_results = check_required_packages(required_packages)
    
    # Check spaCy models
    spacy_results = check_spacy_models()
    
    # Check OpenSSL version
    openssl_result = check_openssl_version()
    
    # Combine results
    all_results = package_results + spacy_results + [openssl_result]
    
    # Check compatibility
    if system_info["is_apple_silicon"] == "True":
        print_colored("Checking Apple Silicon compatibility:", GREEN)
        installed_packages = get_installed_packages()
        compatibility_results = check_package_compatibility(installed_packages, system_info)
        
        # Merge compatibility results with package results
        for compat_result in compatibility_results:
            for result in all_results:
                if result["package"] == compat_result["package"] and compat_result["status"] != "OK":
                    result["status"] = compat_result["status"]
                    result["message"] = compat_result["message"]
    
    # Print results
    for result in all_results:
        status_color = GREEN if result["status"] == "OK" else (YELLOW if result["status"] == "WARNING" else RED)
        print(f"  {result['package']}: {result['installed_version']} - {status_color}{result['status']}{NC}")
        if result["message"]:
            print(f"    {result['message']}")
    
    print()
    
    # Check for errors
    errors = [result for result in all_results if result["status"] == "ERROR"]
    warnings = [result for result in all_results if result["status"] == "WARNING"]
    
    if errors:
        print_colored(f"Found {len(errors)} errors. Please fix them before continuing.", RED)
        print_colored("You can install missing packages with:", YELLOW)
        print_colored("  pip install -r document_forgery_detection/requirements.txt", YELLOW)
        print()
    
    if warnings:
        print_colored(f"Found {len(warnings)} warnings. These may affect performance but are not critical.", YELLOW)
        print()
    
    if not errors and not warnings:
        print_colored("All checks passed! Your system is compatible with Document Forgery Detection System.", GREEN)
    elif not errors:
        print_colored("No critical issues found. Your system should be compatible with Document Forgery Detection System.", GREEN)
        print_colored("Consider addressing the warnings for optimal performance.", YELLOW)

if __name__ == "__main__":
    main() 