import os
import sys
from pathlib import Path

# Add the project directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from read_config import Config

# Get absolute path given a relative path
def get_absolute_path(relative_path: str) -> str:
    """
    Args:
        relative_path (str): The relative path to convert.
        
    Returns:
        str: The absolute path.
    """
    # Get the current file's directory
    # This will give the directory where this script is located
    current_directory = os.path.dirname(os.path.abspath(__file__))
    # Join the current directory with the relative path
    absolute_path = os.path.join(current_directory, relative_path)

    return absolute_path

def get_config(config_file_path: str="config.yaml") -> Config:
    return Config(get_absolute_path(config_file_path))


if __name__ == "__main__":
    # test: get_absolute_path function
    relative_path = "data/bn_en_translation_train.txt"
    absolute_path = get_absolute_path(relative_path)
    print(f"Absolute path for '{relative_path}': {absolute_path}")
    # test: get_config function
    config = get_config()
    print(f"Config from: {config.dataset_dir}")
