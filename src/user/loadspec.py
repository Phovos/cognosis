import sys
import importlib.util
import pathlib
import sys
import importlib.util
from pathlib import Path
import mimetypes
import hashlib
from typing import Dict, Any, Optional
import json
from dataclasses import dataclass, asdict

@dataclass
class FileMetadata:
    """Comprehensive metadata for any file type"""
    path: str
    name: str
    stem: str
    suffix: str
    mime_type: str
    size: int
    hash: str
    is_text: bool
    encoding: Optional[str] = None

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

class FilesystemMemory:
    def __init__(self, root_dir: Optional[Path] = None):
        """
        Initialize filesystem memory mapping
        
        Args:
            root_dir: Directory to scan. Defaults to script's directory if not provided.
        """
        self.root_dir = root_dir or Path(__file__).resolve().parent
        self.loaded_modules: Dict[str, Any] = {}
        self.file_metadata: Dict[str, FileMetadata] = {}
        
    def _generate_file_hash(self, file_path: Path) -> str:
        """Generate SHA-256 hash of file contents"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _extract_file_metadata(self, file_path: Path) -> FileMetadata:
        """
        Extract comprehensive metadata for a given file
        
        Args:
            file_path: Path to the file
        
        Returns:
            FileMetadata object with file information
        """
        # Determine MIME type
        mime_type, encoding = mimetypes.guess_type(str(file_path))
        mime_type = mime_type or 'application/octet-stream'
        
        # Check if file is text-based
        is_text = mime_type.startswith(('text/', 'application/json', 'application/xml'))
        
        # Read file contents if it's a text file
        content = None
        if is_text:
            try:
                content = file_path.read_text(errors='replace')
            except Exception:
                is_text = False
        
        return FileMetadata(
            path=str(file_path),
            name=file_path.name,
            stem=file_path.stem,
            suffix=file_path.suffix,
            mime_type=mime_type,
            size=file_path.stat().st_size,
            hash=self._generate_file_hash(file_path),
            is_text=is_text,
            encoding=encoding
        )
    
    def load_module_from_file(self, file_path: Path) -> Optional[Any]:
        """
        Dynamically load a module from a file
        
        Args:
            file_path: Path to the file to load as a module
        
        Returns:
            Loaded module or None if loading fails
        """
        # Only attempt to load Python files
        if file_path.suffix != '.py':
            return None
        
        try:
            module_name = f"{file_path.stem}_module"
            spec = importlib.util.spec_from_file_location(module_name, str(file_path))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            print(f"Error loading module from {file_path}: {e}")
            return None
    
    def scan_filesystem(self, ignore_patterns: Optional[list] = None):
        """
        Scan the filesystem and build memory mapping
        
        Args:
            ignore_patterns: List of patterns to ignore during scanning
        """
        ignore_patterns = ignore_patterns or [
            '.git', '__pycache__', 'venv', '.env', 
            '*.pyc', '*.log', '*.tmp'
        ]
        
        for file_path in self.root_dir.rglob('*'):
            if file_path.is_file():
                # Skip ignored files/directories
                if any(file_path.match(pattern) for pattern in ignore_patterns):
                    continue
                
                # Extract metadata
                metadata = self._extract_file_metadata(file_path)
                self.file_metadata[str(file_path)] = metadata
                
                # Try to load as Python module
                module = self.load_module_from_file(file_path)
                if module:
                    module_name = f"{file_path.stem}_module"
                    self.loaded_modules[module_name] = module
    
    def export_metadata(self, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Export file metadata to a JSON file
        
        Args:
            output_path: Path to save metadata. If None, returns dictionary.
        
        Returns:
            Dictionary of file metadata
        """
        metadata_dict = {
            path: asdict(metadata) 
            for path, metadata in self.file_metadata.items()
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
        
        return metadata_dict

def main():
    # Create filesystem memory instance
    fs_memory = FilesystemMemory()
    
    # Scan filesystem
    fs_memory.scan_filesystem()
    
    # Export metadata
    metadata = fs_memory.export_metadata(
        Path(__file__).parent / 'filesystem_metadata.json'
    )
    
    # Print some basic stats
    print(f"Total files scanned: {len(fs_memory.file_metadata)}")
    print(f"Python modules loaded: {len(fs_memory.loaded_modules)}")

if __name__ == "__main__":
    main()