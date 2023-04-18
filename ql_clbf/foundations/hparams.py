import abc
import argparse
import copy

from dataclasses import dataclass, fields, MISSING
from typing import Tuple

@dataclass
class Hparams(abc.ABC):
    """A collection of hyperparameters.
    Add desired hyperparameters with their types as fields. Provide default values where desired.
    You must provide default values for _name and _description. Help text for field `f` is
    optionally provided in the field `_f`,
    """

    def __post_init__(self):
        if not hasattr(self, '_name'): raise ValueError('Must have field _name with string value.')
        if not hasattr(self, '_description'): raise ValueError('Must have field _name with string value.')

    @classmethod
    def add_args(cls, parser, defaults: 'Hparams' = None, prefix: str = None,
                 name: str = None, description: str = None, create_group: bool = True):
        if defaults and not isinstance(defaults, cls):
            raise ValueError(f'defaults must also be type {cls}.')

        if create_group:
            parser = parser.add_argument_group(name or cls._name, description or cls._description)

        for field in fields(cls):
            if field.name.startswith('_'): continue
            arg_name = f'--{field.name}' if prefix is None else f'--{prefix}_{field.name}'
            helptext = getattr(cls, f'_{field.name}') if hasattr(cls, f'_{field.name}') else ''

            if defaults: default = copy.deepcopy(getattr(defaults, field.name, None))
            elif field.default != MISSING: default = copy.deepcopy(field.default)
            else: default = None

            if field.type == bool:
                if (defaults and getattr(defaults, field.name) is not False) or field.default is not False:
                    raise ValueError(f'Boolean hyperparameters must default to False: {field.name}.')
                parser.add_argument(arg_name, action='store_true', help='(optional) ' + helptext)

            elif field.type in [str, float, int]:
                required = field.default is MISSING and (not defaults or not getattr(defaults, field.name))
                if required:  helptext = '(required: %(type)s) ' + helptext
                elif default: helptext = f'(default: {default}) ' + helptext
                else:         helptext = '(optional: %(type)s) ' + helptext
                parser.add_argument(arg_name, type=field.type, default=default, required=required, help=helptext)

            # If it is a nested hparams, use the field name as the prefix and add all arguments.
            elif isinstance(field.type, type) and issubclass(field.type, Hparams):
                subprefix = f'{prefix}_{field.name}' if prefix else field.name
                field.type.add_args(parser, defaults=default, prefix=subprefix, create_group=False)

            else: raise ValueError(f'Invalid field type {field.type} for hparams.')

    @classmethod
    def create_from_args(cls, args: argparse.Namespace, prefix: str = None) -> 'Hparams':
        d = {}
        for field in fields(cls):
            if field.name.startswith('_'): continue

            # Base types.
            if field.type in [bool, str, float, int]:
                arg_name = f'{field.name}' if prefix is None else f'{prefix}_{field.name}'
                if not hasattr(args, arg_name): raise ValueError(f'Missing argument: {arg_name}.')
                d[field.name] = getattr(args, arg_name)

            # Nested hparams.
            elif isinstance(field.type, type) and issubclass(field.type, Hparams):
                subprefix = f'{prefix}_{field.name}' if prefix else field.name
                d[field.name] = field.type.create_from_args(args, subprefix)

            else: raise ValueError(f'Invalid field type {field.type} for hparams.')

        return cls(**d)

    @property
    def display(self):
        nondefault_fields = [f for f in fields(self)
                             if not f.name.startswith('_') and ((f.default is MISSING) or getattr(self, f.name))]
        s = self._name + '\n'
        return s + '\n'.join(f'    * {f.name} => {getattr(self, f.name)}' for f in nondefault_fields)

    def __str__(self):
        fs = {}
        for f in fields(self):
            if f.name.startswith('_'): continue
            if f.default is MISSING or (getattr(self, f.name) != f.default):
                value = getattr(self, f.name)
                if isinstance(value, str): value = "'" + value + "'"
                if isinstance(value, Hparams): value = str(value)
                if isinstance(value, Tuple): value = 'Tuple(' + ','.join(str(h) for h in value) + ')'
                fs[f.name] = value
        elements = [f'{name}={fs[name]}' for name in sorted(fs.keys())]
        return 'Hparams(' + ', '.join(elements) + ')'