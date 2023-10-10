import ast
from enum import Enum, EnumMeta
from typing import TypeVar, Type, Any

import aurum as au
from pydantic.main import BaseModel

C = TypeVar("C", bound=BaseModel)


def register_au_params(*config_types: Type[BaseModel]):
    parameters = {}
    for config_type in config_types:
        parameters.update(
            {
                key: (value if not isinstance(value, Enum) else value.value)
                for key, value in config_type.__field_defaults__.items()
            }
        )
    au.parameters(**parameters)


def _preprocess_param(param: Any, type_: Type):
    if isinstance(param, str) and type_ != str and not isinstance(type_, EnumMeta):
        return ast.literal_eval(param)
    else:
        return param


def load_au_params(config_type: Type[C]) -> C:
    return config_type(
        **{
            field_name: _preprocess_param(getattr(au, field_name), field.outer_type_)
            for field_name, field in config_type.__fields__.items()
        }
    )
