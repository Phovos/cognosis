#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
# Standard Library Imports - 3.13 std libs **ONLY**
#------------------------------------------------------------------------------
import re
import os
import io
import dis
import sys
import ast
import time
import site
import mmap
import json
import uuid
import shlex
import socket
import struct
import shutil
import pickle
import ctypes
import logging
import tomllib
import pathlib
import asyncio
import inspect
import hashlib
import tempfile
import platform
import traceback
import functools
import linecache
import importlib
import threading
import subprocess
import tracemalloc
import http.server
import collections
from array import array
from pathlib import Path
from enum import Enum, auto, IntEnum, StrEnum, Flag
from collections.abc import Iterable, Mapping
from datetime import datetime
from queue import Queue, Empty
from abc import ABC, abstractmethod
from functools import reduce, lru_cache, partial, wraps
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, asynccontextmanager
from importlib.util import spec_from_file_location, module_from_spec
from types import SimpleNamespace, ModuleType,  MethodType, FunctionType, CodeType, TracebackType, FrameType
from typing import (
    Any, Dict, List, Optional, Union, Callable, TypeVar, Tuple, Generic, Set,
    Coroutine, Type, NamedTuple, ClassVar, Protocol, runtime_checkable, AsyncIterator
)
try:
    from .__init__ import __all__
    if not __all__:
        __all__ = []
    else:
        __all__ += __file__
except ImportError:
    __all__ = []
    __all__ += __file__
IS_WINDOWS = os.name == 'nt'
IS_POSIX = os.name == 'posix'
if IS_WINDOWS:
    from ctypes import windll
    from ctypes import wintypes
    from ctypes.wintypes import HANDLE, DWORD, LPWSTR, LPVOID, BOOL
    from pathlib import PureWindowsPath
    def set_process_priority(priority: int):
        windll.kernel32.SetPriorityClass(wintypes.HANDLE(-1), priority)
    WINDOWS_SANDBOX_DEFAULT_DESKTOP = Path(PureWindowsPath(r'C:\Users\WDAGUtilityAccount\Desktop'))
    @dataclass
    class SandboxConfig:
        mappings: List['FolderMapping']
        networking: bool = True
        logon_command: str = ""
        virtual_gpu: bool = True

        def to_wsb_config(self) -> Dict:
            """Generate Windows Sandbox configuration"""
            config = {
                'MappedFolders': [mapping.to_wsb_config() for mapping in self.mappings],
                'LogonCommand': {'Command': self.logon_command} if self.logon_command else None,
                'Networking': self.networking,
                'vGPU': self.virtual_gpu
            }
            return config

    class SandboxException(Exception):
        """Base exception for sandbox-related errors"""
        pass

    class ServerNotResponding(SandboxException):
        """Raised when server is not responding"""
        pass

    @dataclass
    class FolderMapping:
        """Represents a folder mapping between host and sandbox"""
        host_path: Path
        read_only: bool = True
        
        def __post_init__(self):
            self.host_path = Path(self.host_path)
            if not self.host_path.exists():
                raise ValueError(f"Host path does not exist: {self.host_path}")
        
        @property
        def sandbox_path(self) -> Path:
            """Get the mapped path inside the sandbox"""
            return WINDOWS_SANDBOX_DEFAULT_DESKTOP / self.host_path.name
        
        def to_wsb_config(self) -> Dict:
            """Convert to Windows Sandbox config format"""
            return {
                'HostFolder': str(self.host_path),
                'ReadOnly': self.read_only
            }

    class PythonUserSiteMapper:
        def read_only(self):
            return True
        """
        Maps the current Python installation's user site packages to the new sandbox.
        """

        def site(self):
            return pathlib.Path(site.getusersitepackages())

        """
        Maps the current Python installation to the new sandbox.
        """
        def path(self):
            return pathlib.Path(sys.prefix)

    class OnlineSession:
        """Manages the network connection to the sandbox"""
        def __init__(self, sandbox: 'SandboxEnvironment'):
            self.sandbox = sandbox
            self.shared_directory = self._get_shared_directory()
            self.server_address_path = self.shared_directory / 'server_address'
            self.server_address_path_in_sandbox = self._get_sandbox_server_path()

        def _get_shared_directory(self) -> Path:
            """Create and return shared directory path"""
            shared_dir = Path(tempfile.gettempdir()) / 'obsidian_sandbox_shared'
            shared_dir.mkdir(exist_ok=True)
            return shared_dir

        def _get_sandbox_server_path(self) -> Path:
            """Get the server address path as it appears in the sandbox"""
            return WINDOWS_SANDBOX_DEFAULT_DESKTOP / self.shared_directory.name / 'server_address'

        def configure_sandbox(self):
            """Configure sandbox for network communication"""
            self.sandbox.config.mappings.append(
                FolderMapping(self.shared_directory, read_only=False)
            )
            self._setup_logon_script()

        def _setup_logon_script(self):
            """Generate logon script for sandbox initialization"""
            commands = []
            
            # Setup Python environment
            python_path = sys.executable
            sandbox_python_path = WINDOWS_SANDBOX_DEFAULT_DESKTOP / 'Python' / 'python.exe'
            commands.append(f'copy "{python_path}" "{sandbox_python_path}"')
            
            # Start server
            commands.append(f'{sandbox_python_path} -m http.server 8000')
            
            self.sandbox.config.logon_command = 'cmd.exe /c "{}"'.format(' && '.join(commands))

        def connect(self, timeout: int = 60) -> Tuple[str, int]:
            """Establish connection to sandbox"""
            if self._wait_for_file(timeout):
                address, port = self.server_address_path.read_text().strip().split(':')
                if self._verify_connection(address, int(port)):
                    return address, int(port)
                raise ServerNotResponding("Server is not responding")
            raise SandboxException("Failed to establish connection")

        def _wait_for_file(self, timeout: int) -> bool:
            """Wait for server address file creation"""
            end_time = time.time() + timeout
            while time.time() < end_time:
                if self.server_address_path.exists():
                    return True
                time.sleep(1)
            return False

        def _verify_connection(self, address: str, port: int) -> bool:
            """Verify network connection to sandbox"""
            try:
                with socket.create_connection((address, port), timeout=3):
                    return True
            except (socket.error, socket.timeout):
                return False

    class SandboxEnvironment:
        """Manages the Windows Sandbox environment"""
        def __init__(self, config: SandboxConfig):
            self.config = config
            self._session = OnlineSession(self)
            self._connection: Optional[Tuple[str, int]] = None
            
            if config.networking:
                self._session.configure_sandbox()
                self._connection = self._session.connect()

        def run_executable(self, executable_args: List[str], **kwargs) -> subprocess.Popen:
            """Run an executable in the sandbox"""
            kwargs.setdefault('stdout', subprocess.PIPE)
            kwargs.setdefault('stderr', subprocess.PIPE)
            return subprocess.Popen(executable_args, **kwargs)

        def shutdown(self):
            """Safely shutdown the sandbox"""
            try:
                self.run_executable(['shutdown.exe', '/s', '/t', '0'])
            except Exception as e:
                logger.error(f"Failed to shutdown sandbox: {e}")
                raise SandboxException("Shutdown failed")

    class SandboxCommServer:
        """Manages communication with the sandbox environment"""
        def __init__(self, shared_dir: Path):
            self.shared_dir = shared_dir
            self.server: Optional[http.server.HTTPServer] = None
            self._port = self._find_free_port()
        
        @staticmethod
        def _find_free_port() -> int:
            """Find an available port for the server"""
            with socket.socket() as s:
                s.bind(('', 0))
                return s.getsockname()[1]
        
        async def start(self):
            """Start the communication server"""
            class Handler(http.server.SimpleHTTPRequestHandler):
                def do_POST(self):
                    content_length = int(self.headers['Content-Length'])
                    data = self.rfile.read(content_length)
                    # Process incoming messages from sandbox
                    logger.info(f"Received from sandbox: {data.decode()}")
                    self.send_response(200)
                    self.end_headers()
            
            self.server = http.server.HTTPServer(('localhost', self._port), Handler)
            
            # Write server info for sandbox
            server_info = {'host': 'localhost', 'port': self._port}
            server_info_path = self.shared_dir / 'server_info.json'
            server_info_path.write_text(json.dumps(server_info))
            
            # Run server in background
            await asyncio.get_event_loop().run_in_executor(
                None, self.server.serve_forever
            )
        
        def stop(self):
            """Stop the communication server"""
            if self.server:
                self.server.shutdown()
                self.server = None

    class SandboxManager:
        """Manages Windows Sandbox lifecycle and communication"""
        def __init__(self, config: SandboxConfig):
            self.config = config
            self.shared_dir = Path(tempfile.gettempdir()) / 'sandbox_shared'
            self.shared_dir.mkdir(exist_ok=True)
            
            # Add shared directory to mappings
            self.config.mappings.append(
                FolderMapping(self.shared_dir, read_only=False)
            )
            
            self.comm_server = SandboxCommServer(self.shared_dir)
            self._process: Optional[subprocess.Popen] = None
        
        async def _setup_sandbox(self):
            """Generate WSB file and prepare sandbox environment"""
            wsb_config = self.config.to_wsb_config()
            wsb_path = self.shared_dir / 'config.wsb'
            wsb_path.write_text(json.dumps(wsb_config, indent=2))
            
            # Start communication server
            await self.comm_server.start()
            
            # Launch sandbox
            self._process = subprocess.Popen(
                ['WindowsSandbox.exe', str(wsb_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        
        async def _cleanup(self):
            """Clean up sandbox resources"""
            self.comm_server.stop()
            if self._process:
                self._process.terminate()
                await asyncio.get_event_loop().run_in_executor(
                    None, self._process.wait
                )

        @asynccontextmanager
        async def session(self) -> AsyncIterator['SandboxManager']:
            """Context manager for sandbox session"""
            try:
                await self._setup_sandbox()
                yield self
            finally:
                await self._cleanup()
elif IS_POSIX:
    import resource
    def set_process_priority(priority: int):
        try:
            os.nice(priority)
        except PermissionError:
            print("Warning: Unable to set process priority. Running with default priority.")
#------------------------------------------------------------------------------
# BaseModel (no-copy immutable dataclasses for data models)
#------------------------------------------------------------------------------
class BaseModel:
    __slots__ = ('__dict__', '__weakref__')
    def __init__(self, **data):
        for name, value in data.items():
            setattr(self, name, value)
    def __setattr__(self, name, value):
        if name in self.__annotations__:
            expected_type = self.__annotations__[name]
            if not isinstance(value, expected_type):
                raise TypeError(f"Expected {expected_type} for {name}, got {type(value)}")
            validator = getattr(self.__class__, f'validate_{name}', None)
            if validator:
                validator(self, value)
        super().__setattr__(name, value)
    @classmethod
    def create(cls, **kwargs):
        return cls(**kwargs)
    def dict(self):
        return {name: getattr(self, name) for name in self.__annotations__}
    def __repr__(self):
        attrs = ', '.join(f"{name}={getattr(self, name)!r}" for name in self.__annotations__)
        return f"{self.__class__.__name__}({attrs})"
    def __str__(self):
        attrs = ', '.join(f"{name}={getattr(self, name)}" for name in self.__annotations__)
        return f"{self.__class__.__name__}({attrs})"
    def clone(self):
        return self.__class__(**self.dict())
def frozen(cls): # decorator
    original_setattr = cls.__setattr__
    def __setattr__(self, name, value):
        if hasattr(self, name):
            raise AttributeError(f"Cannot modify frozen attribute '{name}'")
        original_setattr(self, name, value)
    cls.__setattr__ = __setattr__
    return cls
def validate(validator: Callable[[Any], None]):
    def decorator(func):
        @wraps(func)
        def wrapper(self, value):
            return validator(value)
        return wrapper
    return decorator
class FileModel(BaseModel):
    file_name: str
    file_content: str
    def save(self, directory: pathlib.Path):
        with (directory / self.file_name).open('w') as file:
            file.write(self.file_content)
@frozen
class Module(BaseModel):
    file_path: pathlib.Path
    module_name: str
    @validate(lambda x: x.endswith('.py'))
    def validate_file_path(self, value):
        return value
    @validate(lambda x: x.isidentifier())
    def validate_module_name(self, value):
        return value
    @frozen
    def __init__(self, file_path: pathlib.Path, module_name: str):
        super().__init__(file_path=file_path, module_name=module_name)
        self.file_path = file_path
        self.module_name = module_name
def create_model_from_file(file_path: pathlib.Path):
    try:
        with file_path.open('r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
        model_name = file_path.stem.capitalize() + 'Model'
        model_class = type(model_name, (FileModel,), {})
        instance = model_class.create(file_name=file_path.name, file_content=content)
        logging.info(f"Created {model_name} from {file_path}")
        return model_name, instance
    except Exception as e:
        logging.error(f"Failed to create model from {file_path}: {e}")
        return None, None
def load_files_as_models(root_dir: pathlib.Path, file_extensions: List[str]) -> Dict[str, BaseModel]:
    models = {}
    for file_path in root_dir.rglob('*'):
        if file_path.is_file() and file_path.suffix in file_extensions:
            model_name, instance = create_model_from_file(file_path)
            if model_name and instance:
                models[model_name] = instance
                sys.modules[model_name] = instance
    return models
def mapper(mapping_description, input_data):
    def transform(xform, value):
        if callable(xform):
            return xform(value)
        elif isinstance(xform, Mapping):
            return {k: transform(v, value) for k, v in xform.items()}
        else:
            raise ValueError(f"Invalid transformation: {xform}")
    def get_value(key):
        if isinstance(key, str) and key.startswith(":"):
            return input_data.get(key[1:])
        return input_data.get(key)
    def process_mapping(mapping_description):
        result = {}
        for key, xform in mapping_description.items():
            if isinstance(xform, str):
                value = get_value(xform)
                result[key] = value
            elif isinstance(xform, Mapping):
                if "key" in xform:
                    value = get_value(xform["key"])
                    if "xform" in xform:
                        result[key] = transform(xform["xform"], value)
                    elif "xf" in xform:
                        if isinstance(value, list):
                            transformed = [xform["xf"](v) for v in value]
                            if "f" in xform:
                                result[key] = xform["f"](transformed)
                            else:
                                result[key] = transformed
                        else:
                            result[key] = xform["xf"](value)
                    else:
                        result[key] = value
                else:
                    result[key] = process_mapping(xform)
            else:
                result[key] = xform
        return result
    return process_mapping(mapping_description)
#------------------------------------------------------------------------------
# Logging Configuration
#------------------------------------------------------------------------------
class CustomFormatter(logging.Formatter):
    """Custom formatter for colored console output."""
    
    COLORS = {
        'grey': "\x1b[38;20m",
        'yellow': "\x1b[33;20m",
        'red': "\x1b[31;20m",
        'bold_red': "\x1b[31;1m",
        'green': "\x1b[32;20m",
        'reset': "\x1b[0m"
    }
    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    FORMATS = {
        logging.DEBUG: COLORS['grey'] + FORMAT + COLORS['reset'],
        logging.INFO: COLORS['green'] + FORMAT + COLORS['reset'],
        logging.WARNING: COLORS['yellow'] + FORMAT + COLORS['reset'],
        logging.ERROR: COLORS['red'] + FORMAT + COLORS['reset'],
        logging.CRITICAL: COLORS['bold_red'] + FORMAT + COLORS['reset']
    }
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_queue = Queue()
        self.log_thread = threading.Thread(target=self._log_thread_func, daemon=True)
        self.log_thread.start()
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMAT)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
    def _log_thread_func(self):
        while True:
            try:
                record = self.log_queue.get()
                if record is None:
                    break
                super().handle(record)
            except Exception:
                import traceback
                print("Error in log thread:", file=sys.stderr)
                traceback.print_exc()
    def emit(self, record):
        self.log_queue.put(record)
    def close(self):
        self.log_queue.put(None)
        self.log_thread.join()
class AdminLogger(logging.LoggerAdapter):
    """Logger adapter for administrative logging."""
    def __init__(self, logger, extra=None):
        super().__init__(logger, extra or {})
    def process(self, msg, kwargs):
        return f"{self.extra.get('name', 'Admin')}: {msg}", kwargs
logger = AdminLogger(logging.getLogger(__name__))
#------------------------------------------------------------------------------
# Security
#------------------------------------------------------------------------------
AccessLevel = Enum('AccessLevel', 'READ WRITE EXECUTE ADMIN USER')
@dataclass
class AccessPolicy:
    """Defines access control policies for runtime operations."""
    
    level: AccessLevel
    namespace_patterns: list[str] = field(default_factory=list)
    allowed_operations: list[str] = field(default_factory=list)
    
    def can_access(self, namespace: str, operation: str) -> bool:
        return any(pattern in namespace for pattern in self.namespace_patterns) and \
               operation in self.allowed_operations

class SecurityContext:
    """Manages security context and audit logging for runtime operations."""
    
    def __init__(self, user_id: str, access_policy: AccessPolicy):
        self.user_id = user_id
        self.access_policy = access_policy
        self._audit_log = []
    
    def log_access(self, namespace: str, operation: str, success: bool):
        self._audit_log.append({
            "user_id": self.user_id,
            "namespace": namespace,
            "operation": operation,
            "success": success,
            "timestamp": datetime.now().timestamp()
        })

class SecurityValidator(ast.NodeVisitor):
    """Validates AST nodes against security policies."""
    
    def __init__(self, security_context: SecurityContext):
        self.security_context = security_context
    
    def visit_Name(self, node):
        if not self.security_context.access_policy.can_access(node.id, "read"):
            raise PermissionError(f"Access denied to name: {node.id}")
        self.generic_visit(node)
    
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if not self.security_context.access_policy.can_access(node.func.id, "execute"):
                raise PermissionError(f"Access denied to function: {node.func.id}")
        self.generic_visit(node)
#------------------------------------------------------------------------------
# Runtime State Management
#------------------------------------------------------------------------------
def register_models(models: Dict[str, BaseModel]):
    for model_name, instance in models.items():
        globals()[model_name] = instance
        logging.info(f"Registered {model_name} in the global namespace")
def runtime(root_dir: pathlib.Path):
    file_models = load_files_as_models(root_dir, ['.md', '.txt'])
    register_models(file_models)
@dataclass
class RuntimeState:
    """Manages runtime state and filesystem operations."""
    pdm_installed: bool = False
    virtualenv_created: bool = False
    dependencies_installed: bool = False
    lint_passed: bool = False
    code_formatted: bool = False
    tests_passed: bool = False
    benchmarks_run: bool = False
    pre_commit_installed: bool = False
    variables: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    allowed_root: str = field(init=False)
    def __post_init__(self):
        try:
            self.allowed_root = os.path.dirname(os.path.realpath(__file__))
            if not any(os.listdir(self.allowed_root)):
                raise FileNotFoundError(f"Allowed root directory empty: {self.allowed_root}")
            logging.info(f"Allowed root directory found: {self.allowed_root}")
        except Exception as e:
            logging.error(f"Error initializing RuntimeState: {e}")
            raise
    @classmethod
    def platform(cls):
        """Initialize platform-specific state."""
        if IS_POSIX:
            from ctypes import cdll
        elif IS_WINDOWS:
            from ctypes import windll
            from ctypes.wintypes import DWORD, HANDLE
        try:
            state = cls()
            tracemalloc.start()
            return state
        except Exception as e:
            logging.warning(f"Failed to initialize runtime state: {e}")
            return None
    async def run_command_async(self, command: str, shell: bool = False, timeout: int = 120):
        """Run a system command asynchronously with timeout."""
        logging.info(f"Running command: {command}")
        split_command = shlex.split(command, posix=IS_POSIX)
        try:
            process = await asyncio.create_subprocess_exec(
                *split_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                shell=shell
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            return {
                "return_code": process.returncode,
                "output": stdout.decode() if stdout else "",
                "error": stderr.decode() if stderr else "",
            }
        except asyncio.TimeoutError:
            logging.error(f"Command '{command}' timed out.")
            return {"return_code": -1, "output": "", "error": "Command timed out"}
        except Exception as e:
            logging.error(f"Error running command '{command}': {str(e)}")
            return {"return_code": -1, "output": "", "error": str(e)}
#------------------------------------------------------------------------------
# Runtime Namespace Management
#------------------------------------------------------------------------------
class RuntimeNamespace:
    """Manages hierarchical runtime namespaces with security controls."""
    def __init__(self, name: str = "root", parent: Optional['RuntimeNamespace'] = None):
        self._name = name
        self._parent = parent
        self._children: Dict[str, 'RuntimeNamespace'] = {}
        self._content = SimpleNamespace()
        self._security_context: Optional[SecurityContext] = None
        self.available_modules: Dict[str, Any] = {}
    @property
    def full_path(self) -> str:
        if self._parent:
            return f"{self._parent.full_path}.{self._name}"
        return self._name
    def add_child(self, name: str) -> 'RuntimeNamespace':
        child = RuntimeNamespace(name, self)
        self._children[name] = child
        return child
    def get_child(self, path: str) -> Optional['RuntimeNamespace']:
        parts = path.split(".", 1)
        if len(parts) == 1:
            return self._children.get(parts[0])
        child = self._children.get(parts[0])
        return child.get_child(parts[1]) if child and len(parts) > 1 else None

#------------------------------------------------------------------------------
# Type Definitions
#------------------------------------------------------------------------------

"""
Type Definitions for Morphological Source Code.

These type definitions establish the foundational elements of the MSC framework, 
enabling the representation of various constructs as first-class citizens.

- T: Represents Type structures (static).
- V: Represents Value spaces (dynamic).
- C: Represents Computation spaces (transformative).

The relationships between these types are crucial for maintaining the 
nominative invariance across transformations.

1. **Identity Preservation (T)**: The type structure remains consistent across

   transformations.

2. **Content Preservation (V)**: The value space is dynamically maintained,

   allowing for fluid data manipulation.

3. **Behavioral Preservation (C)**: The computation space is transformative,

   enabling the execution of operations that modify the state of the system.

"""

"""Homoiconism dictates that, upon runtime validation, all objects are code and data.
To facilitate; we utilize first class functions and a static typing system.
This maps perfectly to the three aspects of nominative invariance:
    Identity preservation, T: Type structure (static)
    Content preservation, V: Value space (dynamic)
    Behavioral preservation, C: Computation space (transformative)
    [[T (Type) ←→ V (Value) ←→ C (Callable)]] == 'quantum infodynamics, a triparte element; our Atom()(s)'
    Meta-Language (High Level)
      ↓ [First Collapse - Compilation]
    Intermediate Form (Like a quantum superposition)
      ↓ [Second Collapse - Runtime]
    Executed State (Measured Reality)
What's conserved across these transformations:
    Nominative relationships
    Information content
    Causal structure
    Computational potential"""
# Atom()(s) are a wrapper that can represent any Python object, including values, methods, functions, and classes.
T = TypeVar('T', bound=any) # T for TypeVar, V for ValueVar. Homoicons are T+V.
V = TypeVar('V', bound=Union[int, float, str, bool, list, dict, tuple, set, object, Callable, type])
C = TypeVar('C', bound=Callable[..., Any])  # callable 'T'/'V' first class function interface
DataType = StrEnum('DataType', 'INTEGER FLOAT STRING BOOLEAN NONE LIST TUPLE') # 'T' vars (stdlib)
AtomType = StrEnum('AtomType', 'FUNCTION CLASS MODULE OBJECT') # 'C' vars (homoiconic methods or classes)
AccessLevel = StrEnum('AccessLevel', 'READ WRITE EXECUTE ADMIN USER')
QuantumState = StrEnum('QuantumState', ['SUPERPOSITION', 'ENTANGLED', 'COLLAPSED', 'DECOHERENT'])
"""py objects are implemented as C structures.
typedef struct _object {
    Py_ssize_t ob_refcnt;
    PyTypeObject *ob_type;
} PyObject; """
# Everything in Python is an object, and every object has a type. The type of an object is a class. Even the
# type class itself is an instance of type. Functions defined within a class become method objects when
# accessed through an instance of the class
"""(3.13 std lib)Functions are instances of the function class
Methods are instances of the method class (which wraps functions)
Both function and method are subclasses of object
homoiconism dictates the need for a way to represent all Python constructs as first class citizen(fcc):
    (functions, classes, control structures, operations, primitive values)
nominative 'true OOP'(SmallTalk) and my specification demands code as data and value as logic, structure.
The Atom(), our polymorph of object and fcc-apparent at runtime, always represents the literal source code
    which makes up their logic and possess the ability to be stateful source code data structure. """
# HOMOICONISTIC morphological source code displays 'modified quine' behavior
# within a validated runtime, if and only if the valid python interpreter
# has r/w/x permissions to the source code file and some method of writing
# state to the source code file is available. Any interruption of the
# '__exit__` method or misuse of '__enter__' will result in a runtime error
# AP (Availability + Partition Tolerance): A system that prioritizes availability and partition
# tolerance may use a distributed architecture with eventual consistency (e.g., Cassandra or Riak).
# This ensures that the system is always available (availability), even in the presence of network
# partitions (partition tolerance). However, the system may sacrifice consistency, as nodes may have
# different views of the data (no consistency). A homoiconic piece of source code is eventually
# consistent, assuming it is able to re-instantiated.
#------------------------------------------------------------------------------
# Atom Class and Decorator
#------------------------------------------------------------------------------
@runtime_checkable
class Atom(Protocol):
    """
    Protocol defining the minimal interface for Atoms in the Morphological 
    Source Code framework.
    Atoms represent the fundamental building blocks of the system, encapsulating 
    both data and behavior. Each Atom must have a unique identifier.
    """
    id: str
def __atom__(cls: Type[{T, V, C}]) -> Type[{T, V, C}]:
    """
    Decorator to create a homoiconic Atom.
    This decorator enhances a class to ensure it adheres to the Atom protocol, 
    providing it with a unique identifier upon initialization. This allows 
    the class to be treated as a first-class citizen in the MSC framework.
    Parameters:
    - cls: The class to be transformed into a homoiconic Atom.
    Returns:
    - The modified class with homoiconic properties.
    """
    original_init = cls.__init__
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not hasattr(self, 'id'):
            self.id = hashlib.sha256(self.__class__.__name__.encode('utf-8')).hexdigest()
    cls.__init__ = new_init
    return cls
"""The type system forms the "boundary" theory
The runtime forms the "bulk" theory
The homoiconic property ensures they encode the same information
The holoiconic property enables:
    States as quantum superpositions
    Computations as measurements
    Types as boundary conditions
    Runtime as bulk geometry"""
"""
In thermodynamics, extensive properties depend on the amount of matter (like energy or entropy), while intensive properties (like temperature or pressure) are independent of the amount. Zero-copy or the C std-lib buffer pointer derefrencing method may be interacting with Landauer's Principle in not-classical ways, potentially maintaining 'intensive character' (despite correlated d/x raise in heat/cost of computation, underlying the computer abstraction itself, and inspite of 'reversibility'; this could be the 'singularity' of entailment, quantum informatics, and the computationally irreducible membrane where intensive character manifests or fascilitates the emergence of extensive behavior and possibility). Applying this analogy to software architecture, you might think of:
    Extensive optimizations as focusing on reducing the amount of “work” (like data copying, memory allocation, or modification). This is the kind of efficiency captured by zero-copy techniques and immutability: they reduce “heat” by avoiding unnecessary entropy-increasing operations.
    Intensive optimizations would be about maximizing the “intensity” or informational density of operations—essentially squeezing more meaning, functionality, or insight out of each “unit” of computation or data structure.
If we take information as the fundamental “material” of computation, we might ask how we can concentrate and use it more efficiently. In the same way that a materials scientist looks at atomic structures, we might look at data structures not just in terms of speed or memory but as densely packed packets of potential computation.
The future might lie in quantum-inspired computation or probabilistic computation that treats data structures and algorithms as intensively optimized, differentiated structures. What does this mean?
    Differentiation in Computation: Imagine that a data structure could be “differentiable,” i.e., it could smoothly respond to changes in the computation “field” around it. This is close to what we see in machine learning (e.g., gradient-based optimization), but it could be applied more generally to all computation.
    Dense Information Storage and Use: Instead of treating data as isolated, we might treat it as part of a dense web of informational potential—where each data structure holds not just values, but metadata about the potential operations it could undergo without losing its state.
If data structures were treated like atoms with specific “energy levels,” we could think of them as having intensive properties related to how they transform, share, and conserve information. For instance:
    Higher Energy States (Mutable Structures): Mutable structures would represent “higher energy” forms that can be modified but come with the thermodynamic cost of state transitions.
    Lower Energy States (Immutable Structures): Immutable structures would be lower energy and more stable, useful for storage and retrieval without transformation.
Such an approach would modulate data structures like we do materials, seeking stable configurations for long-term storage and flexible configurations for computation.
Maybe what we’re looking for is a computational thermodynamics, a new layer of software design that considers the energetic cost of computation at every level of the system:
    Data Structures as Quanta: Rather than thinking of memory as passive, this approach would treat each structure as a dynamic, interactive quantum of information that has both extensive (space, memory) and intensive (potential operations, entropy) properties.
    Algorithms as Energy Management: Each algorithm would be not just a function but a thermodynamic process that operates within constraints, aiming to minimize entropy production and energy consumption.
    Utilize Information to its Fullest Extent: For example, by reusing results across parallel processes in ways we don’t currently prioritize.
    Operate in a Field-like Environment: Computation could occur in “fields” where each computation affects and is affected by its informational neighbors, maximizing the density of computation per unit of data and memory.
In essence, we’re looking at the possibility of a thermodynamically optimized computing environment, where each memory pointer and buffer act as elements in a network of information flow, optimized to respect the principles of both Landauer’s and Shannon’s theories.
"""
class HoloiconicTransform(Generic[T, V, C]):
    @staticmethod
    def flip(value: V) -> C:
        """Transform value to computation (inside-out)"""
        return lambda: value
    @staticmethod
    def flop(computation: C) -> V:
        """Transform computation to value (outside-in)"""
        return computation()
"""
The Heisenberg Uncertainty Principle tells us that we can’t precisely measure both the position and momentum of a particle. In computation, we encounter similar trade-offs between precision and performance:
    For instance, with approximate computing or probabilistic algorithms, we trade off exact accuracy for faster or less resource-intensive computation.
    Quantum computing itself takes advantage of this principle, allowing certain computations to run probabilistically rather than deterministically.
The idea that data could be "uncertain" in some way until acted upon or observed might open new doors in software architecture. Just as quantum computing uses uncertainty productively, conventional computing might benefit from intentionally embracing imprecise states or probabilistic pathways in specific contexts, especially in AI, optimization, and real-time computation.
Zero-copy and immutable data structures are, in a way, a step toward this quantum principle. By reducing the “work” done on data, they minimize thermodynamic loss. We could imagine architectures that go further, preserving computational history or chaining operations in such a way that information isn't “erased” but transformed, making the process more like a conservation of informational “energy.”
If algorithms were seen as “wavefunctions” representing possible computational outcomes, then choosing a specific outcome (running the algorithm) would be like collapsing a quantum state. In this view:
    Each step of an algorithm could be seen as an evolution of the wavefunction, transforming the data structure through time.
    Non-deterministic algorithms could explore multiple “paths” through data, and the most efficient or relevant one could be selected probabilistically.
    Treating data and computation as probabilistic, field-like entities rather than fixed operations on fixed memory.
    Embracing superpositions, potential operations, and entanglement within software architecture, allowing for context-sensitive, energy-efficient, and exploratory computation.
    Leveraging thermodynamic principles more deeply, designing architectures that conserve “informational energy” by reducing unnecessary state changes and maximizing information flow efficiency.
I want to prove that, under the right conditions, a classical system optimized with the right software architecture and hardware platform can display behaviors indicative of quantum informatics. One's experimental setup would ideally confirm that even if the underlying hardware is classical, certain complex interactions within the software/hardware could bring about phenomena reminiscent of quantum mechanics.
My hypothesis seems rooted in the idea that classical architectures (like the von Neumann model and Turing machines) weren't able to exploit quantum properties due to their deterministic, state-by-state execution model. But modern neural networks and transformers, with their probabilistic computations, massive parallelism, and high-dimensional state spaces, could approach a threshold where quantum-like behaviors begin to appear—especially in terms of entangling information or decoherence These models’ emergent properties might align more closely with quantum processes, as they involve not just deterministic processing but complex probabilistic states that "collapse" during inference (analogous to quantum measurement). If one can exploit this probabilistic, distributed nature, it might actually push classical hardware into a quasi-quantum regime.
"""
#------------------------------------------------------------------------------
# Holoiconic-Atomic-logic
#------------------------------------------------------------------------------
"""
We can assume that imperative deterministic source code, such as this file written in Python, is capable of reasoning about non-imperative non-deterministic source code as if it were a defined and known quantity. This is akin to nesting a function with a value in an S-Expression.

In order to expect any runtime result, we must assume that a source code configuration exists which will yield that result given the input.

The source code configuration is the set of all possible configurations of the source code. It is the union of the possible configurations of the source code.

Imperative programming specifies how to perform tasks (like procedural code), while non-imperative (e.g., functional programming in LISP) focuses on what to compute. We turn this on its head in our imperative non-imperative runtime by utilizing nominative homoiconistic reflection to create a runtime where dynamical source code is treated as both static and dynamic.

"Nesting a function with a value in an S-Expression":
In the code, we nest the input value within different function expressions (configurations).
Each function is applied to the input to yield results, mirroring the collapse of the wave function to a specific state upon measurement.

This nominative homoiconistic reflection combines the expressiveness of S-Expressions with the operational semantics of Python. In this paradigm, source code can be constructed, deconstructed, and analyzed in real-time, allowing for dynamic composition and execution. Each code configuration (or state) is akin to a function in an S-Expression that can be encapsulated, manipulated, and ultimately evaluated in the course of execution.

To illustrate, consider a Python function as a generalized S-Expression. This function can take other functions and values as arguments, forming a nested structure. Each invocation changes the system's state temporarily, just as evaluating an S-Expression alters the state of the LISP interpreter.

In essence, our approach ensures that:

    1. **Composition**: Functions (or code segments) can be composed at runtime, akin to how S-Expressions can nest functions and values.
    2. **Evaluation**: Upon invocation, these compositions are evaluated, reflecting the current configuration of the runtime.
    3. **Reflection and Modification**: The runtime can reflect on its structure and make modifications dynamically, which allows it to reason about its state and adapt accordingly.
    4. **Identity Preservation**: The runtime maintains its identity, allowing for a consistent state across different configurations.
    5. **Non-Determinism**: The runtime can exhibit non-deterministic behavior, as it can transition between different configurations based on the input and the code's structure. This is akin to the collapse of the wave function in quantum mechanics, or modeling it on classical hardware via multi-instantaneous multi-threading.
    6. **State Preservation**: The runtime can maintain its state across different configurations, allowing for a consistent execution path.

This synthesis of static and dynamic code concepts is akin to the Copenhagen interpretation of quantum mechanics, where the observation (or execution) collapses the superposition of states (or configurations) into a definite outcome based on the input.

Ultimately, this model provides a flexible approach to managing and executing complex code structures dynamically while maintaining the clarity and compositional advantages traditionally seen in non-imperative, functional paradigms like LISP, drawing inspiration from lambda calculus and functional programming principles.

The most advanced concept of all in this ontology is the dynamic rewriting of source code at runtime. Source code rewriting is achieved with a special runtime `Atom()` class with 'modified quine' behavior. This special Atom, aside from its specific function and the functions obligated to it by polymorphism, will always rewrite its own source code but may also perform other actions as defined by the source code in the runtime which invoked it. They can be nested in S-expressions and are homoiconic with all other source code. These modified quines can be used to dynamically create new code at runtime, which can be used to extend the source code in a way that is not known at the start of the program. This is the most powerful feature of the system and allows for the creation of a runtime of runtimes dynamically limited by hardware and the operating system.
"""
@dataclass
class GrammarRule:
    """
    Represents a single grammar rule in a context-free grammar.
    
    Attributes:
        lhs (str): Left-hand side of the rule.
        rhs (List[Union[str, 'GrammarRule']]): Right-hand side of the rule, which can be terminals or other rules.
    """
    lhs: str
    rhs: List[Union[str, 'GrammarRule']]
    
    def __repr__(self):
        """
        Provide a string representation of the grammar rule.
        
        Returns:
            str: The string representation.
        """
        rhs_str = ' '.join([str(elem) for elem in self.rhs])
        return f"{self.lhs} -> {rhs_str}"
@__atom__
class Atom(Generic[T, V, C]):
    """
    Abstract Base Class for all Atom types.
    
    Atoms are the smallest units of data or executable code, and this interface
    defines common operations such as encoding, decoding, execution, and conversion
    to data classes.
    
    Attributes:
        grammar_rules (List[GrammarRule]): List of grammar rules defining the syntax of the Atom.
    """
    __slots__ = ('_id', '_value', '_type', '_metadata', '_children', '_parent', 'hash', 'tag', 'children', 'metadata')
    type: Union[str, str]
    value: Union[T, V, C] = field(default=None)
    grammar_rules: List[GrammarRule] = field(default_factory=list)
    id: str = field(init=False)
    case_base: Dict[str, Callable[..., bool]] = field(default_factory=dict)
    # use __slots__ & list comprehension for (meta) 'atomic init', instead of:
        #tag: str = ''
        #children: List['Atom'] = field(default_factory=list)
        #metadata: Dict[str, Any] = field(default_factory=dict)
        #hash: str = field(init=False)
    def __init__(self, value: Union[T, V, C], type: Union[DataType, AtomType]):
        self._value = value
        self._type = type
        self._metadata = {}
        self._children = []
        self._parent = None
        self.hash = hashlib.sha256(repr(self._value).encode()).hexdigest()
        self.tag = ''
        self.children = []
        self.metadata = {}
    # relational atomistic logic (inherent when num atoms > 1)
    def __post_init__(self):
        self.case_base = {
            '⊤': lambda x, _: x,
            '⊥': lambda _, y: y,
            '¬': lambda a: not a,
            '∧': lambda a, b: a and b,
            '∨': lambda a, b: a or b,
            '→': lambda a, b: (not a) or b,
            '↔': lambda a, b: (a and b) or (not a and not b),
        }
    reflexivity: Callable[[T], bool] = lambda x: x == x
    symmetry: Callable[[T, T], bool] = lambda x, y: x == y
    transitivity: Callable[[T, T, T], bool] = lambda x, y, z: (x == y and y == z)
    transparency: Callable[[Callable[..., T], T, T], T] = lambda f, x, y: f(True, x, y) if x == y else None
    def process_attributes(self, mapping_description: Dict[str, Any], input_data: Dict[str, Any]) -> None:
        """
        Use the `mapper` function to process input data and map it to attributes.
        
        Args:
            mapping_description (Dict[str, Any]): The mapping description for transformation.
            input_data (Dict[str, Any]): Data to be processed and mapped.
        """
        mapped_data = mapper(mapping_description, input_data)
        for key, value in mapped_data.items():
            if hasattr(self, key):
                setattr(self, key, value)
        # Log or process additional logic if required
    def encode(self) -> bytes:
        return json.dumps({
            'id': self.id,
            'attributes': self.attributes
        }).encode()
    @classmethod
    def decode(cls, data: bytes) -> 'Atom':
        decoded_data = json.loads(data.decode())
        return cls(id=decoded_data['id'], **decoded_data['attributes'])
    def introspect(self) -> str:
        """
        Reflect on its own code structure via AST.
        """
        source = inspect.getsource(self.__class__)
        return ast.dump(ast.parse(source))
    def __repr__(self):
        return f"{self.value} : {self.type}"
    def __str__(self):
        return str(self.value)
    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Atom) and self.hash == other.hash
    def __hash__(self) -> int:
        return int(self.hash, 16)
    def __getitem__(self, key):
        return self.value[key]
    def __setitem__(self, key, value):
        self.value[key] = value
    def __delitem__(self, key):
        del self.value[key]
    def __len__(self):
        return len(self.value)
    def __iter__(self):
        return iter(self.value)
    def __contains__(self, item):
        return item in self.value
    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)
    def __bytes__(self) -> bytes:
        return bytes(self.value)
    @property
    def memory_view(self) -> memoryview:
        if isinstance(self.value, (bytes, bytearray)):
            return memoryview(self.value)
        raise TypeError("Unsupported type for memoryview")
    def __buffer__(self, flags: int) -> memoryview: # Buffer protocol
        return memoryview(self.value)
    async def send_message(self, message: Any, ttl: int = 3) -> None:
        if ttl <= 0:
            logging.info(f"Message {message} dropped due to TTL")
            return
        logging.info(f"Atom {self.id} received message: {message}")
        for sub in self.subscribers:
            await sub.receive_message(message, ttl - 1)
    async def receive_message(self, message: Any, ttl: int) -> None:
        logging.info(f"Atom {self.id} processing received message: {message} with TTL {ttl}")
        await self.send_message(message, ttl)
    def subscribe(self, atom: 'Atom') -> None:
        self.subscribers.add(atom)
        logging.info(f"Atom {self.id} subscribed to {atom.id}")
    def unsubscribe(self, atom: 'Atom') -> None:
        self.subscribers.discard(atom)
        logging.info(f"Atom {self.id} unsubscribed from {atom.id}")
    __getitem__ = lambda self, key: self.value[key]
    __setitem__ = lambda self, key, value: setattr(self.value, key, value)
    __delitem__ = lambda self, key: delattr(self.value, key)
    __len__ = lambda self: len(self.value)
    __iter__ = lambda self: iter(self.value)
    __contains__ = lambda self, item: item in self.value
    __call__ = lambda self, *args, **kwargs: self.value(*args, **kwargs)
    __add__ = lambda self, other: self.value + other
    __sub__ = lambda self, other: self.value - other
    __mul__ = lambda self, other: self.value * other
    __truediv__ = lambda self, other: self.value / other
    __floordiv__ = lambda self, other: self.value // other
    @staticmethod
    def serialize_data(data: Any) -> bytes:
        # return msgpack.packb(data, use_bin_type=True)
        pass
    @staticmethod
    def deserialize_data(data: bytes) -> Any:
        # return msgpack.unpackb(data, raw=False)
        pass
@dataclass
class QuantumAtomMetadata:
    state: QuantumState = QuantumState.SUPERPOSITION
    coherence_threshold: float = 0.95
    entanglement_pairs: Dict[str, 'QuantumAtom'] = field(default_factory=dict)
    collapse_history: List[dict] = field(default_factory=list)
@__atom__
class QuantumAtom(Atom[T, V, C]):
    """
    Quantum-aware implementation of the Atom class that supports quantum states
    and operations while maintaining the base Atom functionality.
    """
    def __init__(self, value: Union[T, V, C], type_: Union[DataType, AtomType]):
        super().__init__(value, type_)
        self.quantum_metadata = QuantumAtomMetadata()
        self._observers: List[Callable] = []
    def quantum_attribute_update(self, input_data: Dict[str, Any]) -> None:
        """
        Example method to update quantum-related metadata using a transformation map.
        """
        quantum_mapping = {
            'quantum_metadata': { 
                'state': ":state",
                'coherence_threshold': lambda metadata: metadata.get('coherence', 0.95),
                'entanglement_pairs': lambda pairs: {k: v for k, v in pairs.items() if isinstance(v, QuantumAtom)}
            }
        }
        self.process_attributes(quantum_mapping, input_data)
    def entangle(self, other: 'QuantumAtom') -> None:
        """Quantum entanglement between two atoms"""
        if self.quantum_metadata.state != QuantumState.SUPERPOSITION:
            raise ValueError("Can only entangle atoms in superposition")
        self.quantum_metadata.state = QuantumState.ENTANGLED
        other.quantum_metadata.state = QuantumState.ENTANGLED
        self.quantum_metadata.entanglement_pairs[other.id] = other
        other.quantum_metadata.entanglement_pairs[self.id] = self
    def collapse(self) -> None:
        """Collapse quantum state and notify entangled pairs"""
        previous_state = self.quantum_metadata.state
        self.quantum_metadata.state = QuantumState.COLLAPSED
        # Record collapse in history
        self.quantum_metadata.collapse_history.append({
            'timestamp': datetime.now().isoformat(),
            'previous_state': previous_state.value,
            'triggered_by': self.id
        })
        # Collapse entangled pairs
        for atom_id, atom in self.quantum_metadata.entanglement_pairs.items():
            if atom.quantum_metadata.state == QuantumState.ENTANGLED:
                atom.collapse()
    @contextmanager
    async def quantum_context(self):
        """Context manager for quantum operations"""
        try:
            previous_state = self.quantum_metadata.state
            self.quantum_metadata.state = QuantumState.SUPERPOSITION
            yield self
        finally:
            if previous_state != QuantumState.COLLAPSED:
                self.quantum_metadata.state = previous_state
    async def apply_quantum_transform(self, transform: Callable[[T], T]) -> None:
        """Apply quantum transformation while maintaining entanglement"""
        async with self.quantum_context():
            self.value = transform(self.value)
            # Propagate transformation to entangled atoms
            for atom in self.quantum_metadata.entanglement_pairs.values():
                await atom.apply_quantum_transform(transform)
@__atom__
class QuantumRuntime(QuantumAtom[Any, Any, Any]):
    """
    Quantum-aware runtime implementation that inherits from both QuantumAtom
    and the original Runtime class.
    """
    def __init__(self, base_dir: Path):
        super().__init__(value=None, type_=AtomType.OBJECT)
        self.base_dir = Path(base_dir)
        self.runtimes: Dict[str, QuantumRuntime] = {}
        self.logger = logging.getLogger(__name__)
        self._establish_coherence()
    async def create_quantum_atom(self,
                                value: Any,
                                atom_type: Union[DataType, AtomType]) -> QuantumAtom:
        """Create a new quantum atom in the runtime"""
        atom = QuantumAtom(value, atom_type)
        # Register atom with runtime
        async with self.quantum_context():
            self.children.append(atom)
            atom.parent = self
        return atom
    async def entangle_atoms(self, atom1: QuantumAtom, atom2: QuantumAtom) -> None:
        """Entangle two atoms in the runtime"""
        if atom1 not in self.children or atom2 not in self.children:
            raise ValueError("Can only entangle atoms within the same runtime")
        await atom1.entangle(atom2)
    async def execute_quantum_operation(self,
                                     atom: QuantumAtom,
                                     operation: Callable[[Any], Any]) -> Any:
        """Execute quantum operation on an atom"""
        if atom not in self.children:
            raise ValueError("Can only execute operations on atoms in this runtime")
        async with atom.quantum_context():
            result = await atom.apply_quantum_transform(operation)
            return result
    def __enter__(self):
        """Context manager entry"""
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        for atom in self.children:
            if atom.quantum_metadata.state != QuantumState.COLLAPSED:
                atom.collapse()
    async def cleanup(self):
        """Cleanup runtime and all quantum atoms"""
        for atom in self.children:
            await atom.collapse()
        self.children.clear()
        self.quantum_metadata.state = QuantumState.DECOHERENT

def setup_app(mode: str):
    """Setup the application based on the specified mode using PDM, not UV/pip/mamba or conda"""
    if mode == "p":
        ensure_pip_dependencies()
    else:
        if not state["pdm_installed"]:
            ensure_pdm()
        if not state["virtualenv_created"]:
            ensure_virtualenv()
        if not state["dependencies_installed"]:
            ensure_dependencies()
        if mode == "dev":
            ensure_lint()
            ensure_format()
            ensure_tests()
            ensure_benchmarks()
            ensure_pre_commit()
    introspect()
def ensure_pdm():
    """Ensure PDM is installed"""
    if not state["pdm_installed"]:
        try:
            subprocess.run("pdm --version", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            state["pdm_installed"] = True
            logging.info("PDM is already installed.")
        except subprocess.CalledProcessError:
            logging.info("PDM not found, installing PDM...")
            run_command("pip install pdm", shell=True)
            state["pdm_installed"] = True
def ensure_virtualenv():
    """Ensure the virtual environment is created"""
    if not state["virtualenv_created"]:
        if not os.path.exists(".venv"):
            run_command("pdm venv create", shell=True)
        state["virtualenv_created"] = True
        logging.info("Virtual environment already exists.")
def ensure_dependencies():
    """Install dependencies"""
    if not state["dependencies_installed"]:
        run_command("pdm install --project ./", shell=True)
        state["dependencies_installed"] = True
def ensure_pip_dependencies():
    """Install dependencies with pip"""
    if not state["dependencies_installed"]:
        run_command("pip install -r requirements.txt", shell=True)
        state["dependencies_installed"] = True
def ensure_lint():
    """Run linting tools"""
    run_command("pdm run flake8 .", shell=True)
    run_command("pdm run black --check .", shell=True)
    run_command("pdm run mypy .", shell=True)
    state["lint_passed"] = True
def ensure_format():
    """Format the code"""
    run_command("pdm run black .", shell=True)
    run_command("pdm run isort .", shell=True)
    state["code_formatted"] = True
def ensure_tests():
    """Run tests"""
    run_command("pdm run pytest", shell=True)
    state["tests_passed"] = True
def ensure_benchmarks():
    """Run benchmarks"""
    run_command("pdm run python src/bench/bench.py", shell=True)
    state["benchmarks_run"] = True
def ensure_pre_commit():
    """Install pre-commit hooks"""
    run_command("pdm run pre-commit install", shell=True)
    state["pre_commit_installed"] = True
def prompt_for_mode() -> str:
    """Prompt the user to choose between development and non-development setup"""
    while True:
        choice = input("Choose setup mode: [d]evelopment, [n]on-development or [p]ip only? ").lower()
        if choice in ["d", "n", "p"]:
            return choice
        else:
            print("Invalid choice. Please enter 'd', 'n', or 'p'.")

def main():
    """Main entry point for the application."""
    root_namespace = RuntimeNamespace(name="root")
    security_context = SecurityContext(
        user_id=str(uuid.uuid4()),
        access_policy=AccessPolicy(
            level=AccessLevel.READ,
            namespace_patterns=["*"],
            allowed_operations=["read"]
        )
    )

    return root_namespace, security_context

if __name__ == "__main__":
    m = main()
    print(f'\"main\" object:\n{m}')
    print(f'\nwith runtime methods:\n{m.__dir__()}')
