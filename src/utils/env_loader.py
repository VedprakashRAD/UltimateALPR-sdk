#!/usr/bin/env python3
"""
Environment variable loader for the Vehicle Tracking System
"""

import os
from typing import Optional

def load_env_file(env_path: str = ".env") -> dict:
    """
    Load environment variables from a .env file.
    
    Args:
        env_path (str): Path to the .env file
        
    Returns:
        dict: Dictionary of environment variables
    """
    env_vars = {}
    
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    
    return env_vars

def get_env_var(key: str, default=None, var_type=str) -> Optional:
    """
    Get environment variable with type conversion and default value.
    
    Args:
        key (str): Environment variable key
        default: Default value if not found
        var_type: Type to convert to (str, int, float, bool)
        
    Returns:
        Value of environment variable or default
    """
    # First check actual environment variables
    value = os.environ.get(key)
    
    # If not found, try to load from .env file
    if value is None:
        env_vars = load_env_file()
        value = env_vars.get(key)
    
    # If still not found, return default
    if value is None:
        return default
    
    # Convert to appropriate type
    try:
        if var_type == bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        else:
            return var_type(value)
    except (ValueError, TypeError):
        return default

# Predefined environment variables for the system
AUTO_DELETE_IMAGES = get_env_var('AUTO_DELETE_IMAGES', False, bool)
KEEP_PROCESSED_IMAGES = get_env_var('KEEP_PROCESSED_IMAGES', False, bool)
IMAGE_STORAGE_PATH = get_env_var('IMAGE_STORAGE_PATH', './CCTV_photos')