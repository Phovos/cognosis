import sys
import importlib
from importlib.util import module_from_spec
import pathlib
mixins = []
__all__ = mixins if mixins else []

# Set the root directory to scan
root_dir = pathlib.Path('/path/to/root/directory')

# Create an array to store the files to load
files_to_load = []

# Recursively traverse the directory tree and collect files
for file in root_dir.rglob('*'):
    if file.is_file():
        # Store the file path and contents in the array
        files_to_load.append((file, file.read_text()))

# Create a module for each file
for file_path, file_contents in files_to_load:
    # Create a module name based on the file name
    module_name = file_path.stem

    # Use importlib to load the contents into the namespace as a module
    spec = module_from_spec(importlib.util.Loader(file_path))
    module = sys.modules[module_name] = spec.load_module()

"""This code uses `pathlib` to recursively traverse the directory tree, collects files in an array, and then creates
a module for each file using `importlib`. The contents of each file are loaded into the namespace as a module."""
