import json
from typing import Any, Dict, Type, Callable, List


class BaseModel:
    def dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def json(self) -> str:
        return json.dumps(self.dict())

    @classmethod
    def parse_obj(cls, data: Dict[str, Any]) -> "BaseModel":
        return cls(**data)

    @classmethod
    def parse_json(cls, json_str: str) -> "BaseModel":
        return cls.parse_obj(json.loads(json_str))

class Field:
    def __init__(self, type_: Type, default: Any = None, required: bool = True):
        self.type = type_
        self.default = default
        self.required = required

def create_model(model_name: str, **field_definitions: Field) -> Type[BaseModel]:
    fields = {}
    annotations = {}
    defaults = {}

    for field_name, field in field_definitions.items():
        annotations[field_name] = field.type
        if not field.required:
            defaults[field_name] = field.default

    def __init__(self, **data):
        for field_name, field in field_definitions.items():
            if field.required and field_name not in data:
                raise ValueError(f"Field {field_name} is required")
            value = data.get(field_name, field.default)
            if not isinstance(value, field.type):
                raise TypeError(
                    f"Expected {field.type} for {field_name}, got {type(value)}"
                )
            setattr(self, field_name, value)

    fields["__annotations__"] = annotations
    fields["__init__"] = __init__

    return type(model_name, (BaseModel,), fields)

User = create_model("User", ID=Field(int), name=Field(str))

class EventBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Any], None]]] = {}

    def subscribe(self, event_type: str, handler: Callable[[Any], None]):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Callable[[Any], None]):
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(handler)

    def publish(self, event_type: str, data: Any):
        for handler in self._subscribers.get(event_type, []):
            handler(data)

class Atom:
    def encode(self) -> bytes:
        pass

    def decode(self, data: bytes) -> None:
        pass

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        pass

class AtomicData(Atom):
    def __init__(self, data: Any):
        self.data = data

    def encode(self) -> bytes:
        return str(self.data).encode()

    def decode(self, data: bytes) -> None:
        self.data = data.decode()

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        return self.data

class FormalTheory(Atom):
    def __init__(self):
        self.axioms: Dict[str, Callable] = {}

    def add_axiom(self, name: str, axiom: Callable):
        self.axioms[name] = axiom

    def encode(self) -> bytes:
        return json.dumps({k: v.__name__ for k, v in self.axioms.items()}).encode()

    def decode(self, data: bytes) -> None:
        self.axioms = json.loads(data.decode())

    def execute(self, axiom_name: str, *args: Any, **kwargs: Any) -> Any:
        if axiom_name in self.axioms:
            return self.axioms[axiom_name](*args, **kwargs)
        raise ValueError(f"Axiom {axiom_name} not found")