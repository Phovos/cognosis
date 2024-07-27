import sys
import importlib.util
import pathlib

def main(specific_file=None):
    # Configuration - Path to the root directory containing the personal knowledge base
    root_dir = pathlib.Path(__file__).resolve().parents[2] / 'docs'
    supported_extensions = {'.txt', '.md'}  # Extend this set as needed

    # Dictionary to store dynamically created "modules"
    loaded_db = {}

    print(f"Root directory: {root_dir}")

    # Function to create and load a "module" dynamically with content from the file
    def load_content_from_file(file_path: pathlib.Path, module_name: str):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # Create a module-like object dynamically
                module = type(sys)(module_name)
                module.content = content
                # Optionally process the content here if necessary
                return module
        except Exception as e:
            print(f"Error reading file {module_name} from {file_path}: {e}")
            return None

    # Function to sanitize module names to avoid conflicts
    def sanitize_module_name(name: str) -> str:
        return name.replace('-', '_').replace(' ', '_').replace('.', '_')

    # If a specific file is given, process only that file
    if specific_file:
        specific_path = root_dir / specific_file
        print(f"Checking specific file: {specific_path}")
        if specific_path.is_file() and specific_path.suffix in supported_extensions:
            module_name = sanitize_module_name(specific_path.stem) + '_module'
            module = load_content_from_file(specific_path, module_name)
            if module:
                loaded_db[module_name] = module
        else:
            print(f"File {specific_file} with supported extension not found in {root_dir}")
    else:
        # Otherwise, process all files in the directory
        for file in root_dir.rglob('*'):
            print(f"Found file: {file}")
            if file.is_file() and file.suffix in supported_extensions:
                # Create a sanitized module name based on the file path
                module_name = sanitize_module_name(file.stem) + '_module'

                # Load "module" from file
                module = load_content_from_file(file, module_name)
                if module:
                    loaded_db[module_name] = module

    # Adding loaded "modules" to the current module's namespace
    for name, module in loaded_db.items():
        globals()[name] = module

    # Maintain __all__ to easily identify loaded "modules"
    global __all__
    __all__ = list(loaded_db.keys())

    # Print loaded "modules" for confirmation (optional)
    print("Loaded modules:", __all__)

if __name__ == "__main__":
    # Accept an optional command-line argument for a specific file
    specific_file = sys.argv[1] if len(sys.argv) > 1 else None
    main(specific_file)