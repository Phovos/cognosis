# src/__init__.py

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
import json
import logging
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Generic, TypeVar
import threading
import queue
import time
import pathlib
import sys
import importlib.util

# Explicitly define what is exposed when the package is imported
__all__ = [
    "asyncio", "Future", "ThreadPoolExecutor", "contextmanager", "json", "logging", 
    "struct", "ABC", "abstractmethod", "dataclass", "field", "wraps", "Any", 
    "Callable", "Dict", "List", "Optional", "Generic", "TypeVar", "threading", 
    "queue", "time", "pathlib", "importlib"
]

# Import internal modules and submodules
try:
    from .app.theory import *
    from .app.llama import *
    from .app.kernel import *
    from .app.model import *
    from .app.atoms import *
except ModuleNotFoundError as e:
    print(f"Error importing internal modules: {e}")
    sys.exit(1)

__all__.extend([
    # Add the names of the imported symbols from the internal modules here
])

@dataclass(frozen=True)
class Field:
    name: str

@dataclass
class AppBus(EventBus):
    def __init__(self, name: str = "AppBus"):
        super().__init__()
        self.logger = Logger(name)

@dataclass
class AppModel(Theory):
    def __init__(self, name: str, description: str, fields: Dict[str, Field]):
        super().__init__(name, description, fields)
        self.logger = Logger(name)
        self.kernel = SymbolicKernel()