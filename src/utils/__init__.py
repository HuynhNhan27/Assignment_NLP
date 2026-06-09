"""Utility functions and helpers."""

from typing import Any, Dict
import json
import yaml
import os


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        return {}
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}


def save_json(data: Any, output_path: str) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        output_path: Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        if isinstance(data, str):
            f.write(data)
        else:
            json.dump(data, f, indent=2)


def load_json(json_path: str) -> Dict[str, Any]:
    """
    Load data from JSON file.
    
    Args:
        json_path: JSON file path
        
    Returns:
        Loaded data dictionary
    """
    with open(json_path, 'r') as f:
        return json.load(f)
