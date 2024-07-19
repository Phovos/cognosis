# atom.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Union

class Atom(ABC):
    @abstractmethod
    def encode(self) -> bytes:
        pass

    @abstractmethod
    def decode(self, data: bytes) -> None:
        pass

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

class AtomicData(Atom):
    def __init__(self, data: Any):
        self.data = data

    def encode(self) -> bytes:
        return str(self.data).encode()

    def decode(self, data: bytes) -> None:
        self.data = data.decode()

    def execute(self, *args, **kwargs) -> Any:
        return self.data

    def __repr__(self) -> str:
        return f"AtomicData({self.data})"

class FormalTheory(Atom):
    def __init__(self):
        self.axioms = {}

    def add_axiom(self, name: str, axiom: callable):
        self.axioms[name] = axiom

    def encode(self) -> bytes:
        return str(self.axioms).encode()

    def decode(self, data: bytes) -> None:
        # Simplified decoding for demonstration
        self.axioms = eval(data.decode())

    def execute(self, axiom_name: str, *args, **kwargs) -> Any:
        if axiom_name in self.axioms:
            return self.axioms[axiom_name](*args, **kwargs)
        raise ValueError(f"Axiom {axiom_name} not found")

    def __repr__(self) -> str:
        return f"FormalTheory(axioms={list(self.axioms.keys())})"
