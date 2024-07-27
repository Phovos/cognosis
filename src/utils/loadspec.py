import sys
import importlib.util
import pathlib

# Configuration - Path to the root directory
root_dir = pathlib.Path(__file__).resolve().parent / 'docs'

# Dictionary to store dynamically created modules
loaded_modules = {}

# Function to create and load a module dynamically
def load_module_from_file(file_path: pathlib.Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Error loading module {module_name} from {file_path}: {e}")
        return None

# Recursively traverse the directory tree and process files
for file in root_dir.rglob('*'):
    if file.is_file():
        # Create a module name based on the file path
        module_name = f"{file.stem}_module"

        # Load module from file
        module = load_module_from_file(file, module_name)
        if module:
            loaded_modules[module_name] = module

# Adding loaded modules to the current module's namespace
for name, module in loaded_modules.items():
    globals()[name] = module

# Maintain __all__ to easily identify loaded modules
__all__ = list(loaded_modules.keys())

# Print loaded modules for confirmation (optional)
print("Loaded modules:", __all__)