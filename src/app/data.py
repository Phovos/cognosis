# data_models.py
from typing import Any, Dict, List, Union
from enum import Enum, auto

class DataType(Enum):
    INT = auto()
    FLOAT = auto()
    STRING = auto()
    BOOL = auto()
    LIST = auto()
    DICT = auto()

class Field:
    def __init__(self, data_type: DataType, required: bool = True):
        self.data_type = data_type
        self.required = required

class Model:
    def __init__(self, **kwargs):
        for field_name, field in self.__class__.__dict__.items():
            if isinstance(field, Field):
                if field.required and field_name not in kwargs:
                    raise ValueError(f"Required field {field_name} is missing")
                setattr(self, field_name, kwargs.get(field_name))

    def validate(self):
        for field_name, field in self.__class__.__dict__.items():
            if isinstance(field, Field):
                value = getattr(self, field_name, None)
                if field.required and value is None:
                    raise ValueError(f"Required field {field_name} is None")
                if value is not None:
                    self._validate_type(field_name, value, field.data_type)

    def _validate_type(self, field_name: str, value: Any, data_type: DataType):
        if data_type == DataType.INT and not isinstance(value, int):
            raise TypeError(f"Field {field_name} must be an integer")
        elif data_type == DataType.FLOAT and not isinstance(value, float):
            raise TypeError(f"Field {field_name} must be a float")
        elif data_type == DataType.STRING and not isinstance(value, str):
            raise TypeError(f"Field {field_name} must be a string")
        elif data_type == DataType.BOOL and not isinstance(value, bool):
            raise TypeError(f"Field {field_name} must be a boolean")
        elif data_type == DataType.LIST and not isinstance(value, list):
            raise TypeError(f"Field {field_name} must be a list")
        elif data_type == DataType.DICT and not isinstance(value, dict):
            raise TypeError(f"Field {field_name} must be a dictionary")

class Event(Model):
    id = Field(DataType.STRING)
    type = Field(DataType.STRING)
    detail_type = Field(DataType.STRING, required=False)
    message = Field(DataType.LIST)

class ActionRequest(Model):
    action = Field(DataType.STRING)
    params = Field(DataType.DICT)
    echo = Field(DataType.STRING, required=False)

class ActionResponse(Model):
    status = Field(DataType.STRING)
    retcode = Field(DataType.INT)
    data = Field(DataType.DICT)
    message = Field(DataType.STRING, required=False)
