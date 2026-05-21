from __future__ import annotations

import re
import math
import numpy as np

from typing import TYPE_CHECKING

from nemantix.common.logger import get_package_logger
from nemantix.llm import AbstractLLMProxy, LLMResponse
from nemantix.core import node as nmx_nodes

if TYPE_CHECKING:
    from nemantix.knowledge_base import NemantixKnowledgeBase

from collections import OrderedDict
from typing import Optional, Dict, Any

logger = get_package_logger(__name__)


class OperationalEnv(Dict):
    """Operational memory environment"""

    def __init__(self):
        super().__init__()

    def get(self, key, default=None, /):
        """Variable retrieval"""
        if key in self:
            return self[key]

        return None

    def set(self, var_name: str, value):
        """Defines and assigns a variable"""
        if var_name in self:
            self[var_name] = value
            return

        # define variable
        self[var_name] = value

    def print(self, verbose=True):
        if verbose:
            logger.debug(f'{self.__class__.__name__}:')

        for k, v in self.items():
            logger.debug(f'  {k} = {v}')


class Frames(OperationalEnv):
    pass


class Metadata(OperationalEnv):
    """Stores all the intentables"""

    def get(self, key, default=None, /):
        if key in self:
            return self[key]

        return default


class Tools(OperationalEnv):
    """Stores all the imported tools"""
    pass


class Struct(OrderedDict):
    """Nemantix structure
        - A struct is an ordered sequence of entries;
        -e.g.: [[struct] = (1, 2, field: value, "str", ([a], var: [b]))]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.int_index = 0

        # mapping between fields and indices
        self.idx_to_key = {}
        self.key_to_idx = {}

    def __len__(self):
        keys = [k for k in self.keys() if isinstance(k, int)]
        return len(keys)

    def __iter__(self):
        return map(lambda x: x[1], self.items())

    def items(self):
        keys = []
        for key in self.keys():
            if isinstance(key, str):
                keys.append(key)
            else:
                if key not in self.idx_to_key:
                    keys.append(key)

        values = [self[key] for key in keys]
        return iter(zip(keys, values))

    def as_flat_list(self) -> list:
        """Returns a flat version of this struct as a list"""
        flat_list = []
        for _, v in self.items():
            if isinstance(v, Struct):
                flat_list.extend(v.as_flat_list())
            else:
                flat_list.append(v)

        return flat_list

    def can_be_seen_as_list(self) -> bool:
        return len(self.key_to_idx) == 0

    @classmethod
    def unbox_in(cls, value: list | dict | Any) -> Any:
        if isinstance(value, Struct):
            args, kwargs = value.to_args_and_kwargs()

            if len(args) == 0:
                return kwargs

            if len(kwargs) == 0:
                return args

            raise NotImplementedError

        if isinstance(value, list):
            return cls.unbox_in_list(values=value)

        if isinstance(value, dict):
            return cls.unbox_in_dict(values=value)

        return value

    @classmethod
    def unbox_in_list(cls, values: list) -> list:
        result = []
        for v in values:
            result.append(cls.unbox_in(value=v))

        return result

    @classmethod
    def unbox_in_dict(cls, values: dict) -> dict:
        result = {}
        for k, v in values.items():
            result[k] = cls.unbox_in(value=v)

        return result

    def set(self, value, key: Optional[str | int] = None):
        if key is None:
            key = self._next_index()

        self[key] = value

        if isinstance(key, str):
            # assign an int index to the named field
            if key not in self.key_to_idx:
                index = self._next_index()
            else:
                index = self.key_to_idx[key]

            self[index] = value

            self.idx_to_key[index] = key
            self.key_to_idx[key] = index

    def get(self, key: int | str, default=None, /):
        if isinstance(key, int) and key < 0:
            key = len(self) + key  # key is negative; so "add"
            if key < 0:
                return None

        return super().get(key, default)

    def update_field(self, key: int | str, value):
        if isinstance(key, int):
            int_key = key
            str_key = self.idx_to_key.get(int_key, None)
        else:
            assert isinstance(key, str)
            str_key = key
            int_key = self.key_to_idx.get(str_key, None)

        if int_key is not None:
            self[int_key] = value

        if str_key is not None:
            self[str_key] = value

    def append(self, value) -> 'Struct':
        """Appends the given value and returns itself"""
        self.set(value, key=None)
        return self

    def union(self, other: 'Struct') -> 'Struct':
        """Creates a new structure as the union of both"""
        assert isinstance(other, Struct)
        union = Struct()

        # TODO: deepcopy of inner structures?
        for k, v in self.items():
            key = k if isinstance(k, str) else None
            union.set(value=v, key=key)

        for k, v in other.items():
            key = k if isinstance(k, str) else None
            union.set(value=v, key=key)

        return union

    def is_index_a_field(self, index: int) -> bool:
        return index in self.idx_to_key

    def to_dict(self) -> dict:
        return {k: v for k, v in self.items()}

    def to_args_and_kwargs(self) -> tuple:
        args = []
        kwargs = {}

        for key in self.keys():
            if isinstance(key, int):
                if key in self.idx_to_key:
                    continue

                args.append(self[key])
            else:
                kwargs[key] = self[key]

        return args, kwargs

    def _next_index(self):
        curr_index = self.int_index
        self.int_index += 1
        return curr_index

    def __repr__(self):
        strings = []

        for k, v in self.items():
            string = str(v)

            if isinstance(v, Struct) and not issubclass(type(v), Struct):
                string = string[len('Struct'):]

            if isinstance(k, str):
                strings.append(f'{k}: {string}')
            else:
                if k not in self.idx_to_key:
                    # since str keys have also an int index;
                    # we omit the latter to avoid duplicates
                    strings.append(string)

        return f'Struct({", ".join(strings)})'

    def __eq__(self, other) -> bool:
        if isinstance(other, Struct):
            return id(self) == id(other)

        return False

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)


class ExternalVariables(Struct):
    """A special Struct containing externally configured variables.
        - it's read-only;
        - can be accessed through a special "ENV" variable from NXS/NXC.
    """

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            super().set(value=Secret.box(v, k), key=k)

    def set(self, value, key: Optional[str | int] = None):
        logger.warning(f'Trying to write read-only structure with "{key}"={value}"')

    def update(self, m, /):
        logger.warning(f'Trying to write read-only structure with "{m}"')

    def update_field(self, key: int | str, value):
        self.set(value, key)

    def as_flat_list(self) -> list:
        return []

    def to_args_and_kwargs(self) -> tuple:
        return [], {}

    def can_be_seen_as_list(self) -> bool:
        return False

    def __repr__(self):
        return f'ExternalVariables(num_vars={len(self)})'


class Frame:
    """Nemantix Frame: it formally defines the expected fields and meaning of a
       structured collection (i.e., a runtime.Struct)"""

    def __init__(self, name: str):
        assert len(name) > 0
        self.name = name
        self.slots: Dict[str, dict] = {}
        self.frames: Dict[str, Frame] = {}

    def add_slot(self, name: str, cardinality: Optional[str], types: list[dict]):
        self.slots[name.lower()] = dict(types=types, cardinality=cardinality)

    def add_frame(self, frame: 'Frame'):
        self.frames[frame.name.upper()] = frame

    def apply_prefix(self, struct: Struct) -> None | Struct:
        """Prefix frame application:
            - usage: {frame}(struct)
            - returns a new struct that matches the frame
        """
        validated_slots = []
        valid_struct = Struct()
        struct = self._extract_struct(struct)

        for k, v in struct.items():
            # skip extra fields
            if k not in self.slots:
                continue

            if isinstance(k, int):
                if not struct.is_index_a_field(index=k):
                    return None
            else:
                assert isinstance(k, str)
                k_lo = k.lower()

                if isinstance(v, Struct):
                    inner_frame = self.get_frame_from_slot(slot_name=k)
                    inner_struct = None

                    if inner_frame is None and k_lo not in self.slots:
                        return None

                    if isinstance(inner_frame, Frame):
                        inner_struct = inner_frame.apply_prefix(struct=v)

                    elif self._match_type(slot=self.slots[k_lo], value=v):
                        inner_struct = v

                    if inner_struct is None:
                        return None

                    validated_slots.append(k)
                    valid_struct.set(value=inner_struct, key=k)
                    continue
                else:
                    # if k not in self.slots:
                    #     return None

                    if not self._match_type(slot=self.slots[k_lo], value=v):
                        return None

                validated_slots.append(k)
                valid_struct.set(value=v, key=k)

        if len(validated_slots) == len(self.slots):
            return valid_struct

        return None

    def get_frame_from_slot(self, slot_name: str) -> None | Frame:
        slot = self.slots.get(slot_name.lower(), None)
        if slot is None:
            return None

        for kind in slot['types']:
            assert isinstance(kind, dict)
            if 'type' in kind and kind['type'] == nmx_nodes.SlotTypesEnum.FRAME:
                return self.frames.get(kind['name'].upper(), None)

        return None

    def apply_postfix(self, struct: Struct) -> Struct:
        """Postfix frame application:
            - i.e, (struct){frame}
            - it's more loose, checking if the structure can be (even) partially viewed in terms of the frame.
            Returns a new struct that complies with the frame, even if fields are missing.
        """
        valid_struct = Struct()
        struct = self._extract_struct(struct)

        for k, v in struct.items():
            if isinstance(k, int):
                continue

            assert isinstance(k, str)

            if isinstance(v, Struct):
                inner_frame = self.get_frame_from_slot(slot_name=k)
                if inner_frame is not None:
                    inner_struct = inner_frame.apply_postfix(struct=v)

                    if inner_struct is not None:
                        valid_struct.set(value=inner_struct, key=k)
            else:
                slot = self.slots.get(k.lower(), None)
                if slot is not None:
                    valid_struct.set(value=self._coerce(v, slot), key=k)

        for i, (k, slot) in enumerate(self.slots.items()):
            if k not in valid_struct:
                if not struct.is_index_a_field(index=i):
                    v = struct.get(i)
                    valid_struct.set(value=self._coerce(v, slot=slot), key=k)
                else:
                    valid_struct.set(value=self._get_default(slot=slot), key=k)

        return valid_struct

    @staticmethod
    def _extract_struct(struct: Struct) -> Struct:
        # TODO: workaround
        if len(struct) == 1 and '__' in struct:
            logger.info('Extracting struct in variable in "__" field.')
            struct = struct['__']

            if not isinstance(struct, Struct):
                logger.warning('Extracted value is not a Struct!')
                return Struct()

        return struct

    def _coerce(self, value, slot: dict):
        kinds = [kind['name'] for kind in slot['types']]

        if value is None:
            return self._get_default(slot)

        if isinstance(value, str):
            if nmx_nodes.SlotTypesEnum.TEXT in kinds:
                return value

        elif isinstance(value, bool):
            if nmx_nodes.SlotTypesEnum.BOOL in kinds:
                return value

        elif isinstance(value, int):
            if nmx_nodes.SlotTypesEnum.INT in kinds:
                return value

        elif isinstance(value, float):
            if nmx_nodes.SlotTypesEnum.FLOAT in kinds:
                return value

        elif isinstance(value, Struct):
            if nmx_nodes.SlotTypesEnum.STRUCT in kinds:
                return value

            if nmx_nodes.SlotTypesEnum.FRAME in kinds:
                return self._cast(value, slot)

        return self._cast(value, slot)

    def _cast(self, value, slot: dict):
        kinds = [kind['name'] for kind in slot['types']]
        for kind in kinds:
            if kind == nmx_nodes.SlotTypesEnum.INT:
                if isinstance(value, (bool, int, float)):
                    return int(value)

                return Builtin.to_num(value)

            if kind == nmx_nodes.SlotTypesEnum.BOOL:
                if isinstance(value, (int, float)):
                    return bool(value)

                return Builtin.to_bool(value)

            if kind == nmx_nodes.SlotTypesEnum.FLOAT:
                if isinstance(value, (int, bool)):
                    return float(value)

                return Builtin.to_num(value)

            if kind == nmx_nodes.SlotTypesEnum.TEXT:
                return Builtin.to_str(value)

            if kind == nmx_nodes.SlotTypesEnum.STRUCT:
                if isinstance(value, Struct):
                    return value

                return Struct()

            if kind == nmx_nodes.SlotTypesEnum.FRAME:
                if isinstance(value, Struct):
                    frame = self.frames.get(slot['name'], None)
                    assert frame is not None

                    for k, slot_ in frame.slots.items():
                        v = value.get(k)
                        value.set(value=self._cast(v, slot=slot_), key=k)

                    return value

        return None

    def _get_default(self, slot: dict):
        for kind in slot['types']:
            kind = kind['name']

            if kind == nmx_nodes.SlotTypesEnum.TEXT:
                return ''

            elif kind == nmx_nodes.SlotTypesEnum.INT:
                return 0

            elif kind == nmx_nodes.SlotTypesEnum.BOOL:
                return False

            elif kind == nmx_nodes.SlotTypesEnum.FLOAT:
                return 0.0

            elif kind == nmx_nodes.SlotTypesEnum.STRUCT:
                return Struct()

            elif kind == nmx_nodes.SlotTypesEnum.FRAME:
                frame = self.frames.get(slot['name'], None)
                assert frame is not None

                struct = Struct()
                for k, slot_ in frame.slots.items():
                    struct.set(value=self._get_default(slot=slot_), key=k)

                return struct
            else:
                raise NotImplementedError

        return None

    @staticmethod
    def _match_type(slot: dict, value) -> bool:
        for kind in slot['types']:
            kind = kind['name']

            if kind == nmx_nodes.SlotTypesEnum.TEXT:
                if isinstance(value, str):
                    return True

            elif kind == nmx_nodes.SlotTypesEnum.INT:
                if isinstance(value, bool):
                    return False

                if isinstance(value, float):
                    # checks if value can be interpreted as integer,
                    # e.g., 10.0 can be seen as 10
                    return float(int(value)) == value

                if isinstance(value, int):
                    return True

            elif kind == nmx_nodes.SlotTypesEnum.BOOL:
                if isinstance(value, bool):
                    return True

            elif kind == nmx_nodes.SlotTypesEnum.FLOAT:
                # TODO: should return true if value is int? (castable to float)
                if isinstance(value, float):
                    return True

            elif kind == nmx_nodes.SlotTypesEnum.STRUCT:
                # TODO: should also check the inner struct's types?
                if isinstance(value, Struct):
                    return True

            # TODO: handle FRAME and ENUM types
            elif kind == nmx_nodes.SlotTypesEnum.FRAME:
                raise NotImplementedError
            else:
                # ENUM
                return False

        return False

    def __repr__(self):
        strings = []
        for slot, value in self.slots.items():
            strings.append(slot)

            for kind in value['types']:
                assert isinstance(kind, dict)

                if kind.get('type', None) == nmx_nodes.SlotTypesEnum.FRAME:
                    frame_name = kind['name']
                    strings.pop()
                    frame_str = str(self.frames[frame_name])[len('frame'):]
                    strings.append(f'{slot}: {frame_name}{frame_str}')
                    break

        return f'Frame({", ".join(strings)})'


# TODO: make read-only (cannot modify and add new fields)
class DocRef(Struct):
    """Reference to knowledge base's document"""

    # TODO: add type
    def __init__(self, node_id: str, score: float, breadcrumbs: str, content: str):
        super().__init__()
        self.node_id = node_id
        self.content = content
        self.score = score
        self.breadcrumbs = breadcrumbs

        # also set fields to be used as a normal struct
        super().set(node_id, key='node_id')
        super().set(score, key='score')
        super().set(content, key='content')
        super().set(breadcrumbs, key='breadcrumbs')

    def set(self, value, key: Optional[str | int] = None):
        pass

    def append(self, value) -> 'DocRef':
        return self

    def union(self, other: 'Struct | DocRef') -> 'Struct':
        """Creates a new structure with a content field, having the contents
           of both documents.
        """
        union = Struct()
        content = f'{self.content} {other.get("content", "")}'
        union.set(value=content, key='content')
        return union

    def __repr__(self) -> str:
        return (f'DOC:<{self.node_id}>(content={self.content[:32]}, score={self.score:2}, '
                f'breadcrumbs={self.breadcrumbs[:16]})')

    def __eq__(self, other) -> bool:
        if isinstance(other, DocRef):
            if self.node_id == other.node_id:
                return True
            else:
                return self.content == other.content

        return False

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)


class Opaque:
    """Reference to a Python object used by a tool"""

    def __init__(self, obj: Any):
        self.obj = obj
        self.identifier = id(self.obj)

    def unbox(self):
        if isinstance(self.obj, Opaque):
            return self.obj.unbox()

        return self.obj

    @classmethod
    def unbox_in(cls, value: Any | list | dict | Struct) -> Any:
        """Unboxes all the Opaques found in the given variable or collection"""
        if isinstance(value, Opaque):
            return value.unbox()

        if isinstance(value, Struct):
            return cls.unbox_in_struct(value)

        if isinstance(value, dict):
            return cls.unbox_in_dict(value)

        if isinstance(value, list):
            return cls.unbox_in_list(value)

        return value

    @classmethod
    def unbox_in_list(cls, arguments: list | tuple) -> list:
        """Unbox (i.e., yields .obj) any opaque instance in provided arguments"""
        assert isinstance(arguments, (list, tuple))
        new_arguments = []

        for arg in arguments:
            new_arguments.append(cls.unbox_in(arg))

        return new_arguments

    @classmethod
    def unbox_in_dict(cls, kw_arguments: dict) -> dict[str, Any]:
        assert isinstance(kw_arguments, dict)
        new_kw_arguments = {}

        for k, v in kw_arguments.items():
            new_kw_arguments[k] = cls.unbox_in(v)

        return new_kw_arguments

    @classmethod
    def unbox_in_struct(cls, struct: Struct) -> Struct:
        assert isinstance(struct, Struct)
        new_struct = Struct()

        for k, v in struct.items():
            new_struct.set(cls.unbox_in(v), key=k)

        return new_struct

    def __repr__(self) -> str:
        return f'Opaque(id={self.identifier})'

    def __eq__(self, other) -> bool:
        if isinstance(other, Opaque):
            return self.identifier == other.identifier

        return False

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)


class Secret(Opaque):
    def __init__(self, obj, name: str):
        super().__init__(obj)
        self.name = name

    @staticmethod
    def box(value, name: str) -> 'Secret':
        if not isinstance(value, Secret):
            return Secret(obj=value, name=name)

        if isinstance(value, Opaque):
            return Secret(obj=value.obj, name=name)

        return value

    def __repr__(self):
        return f'Secret(name={self.name})'


class Builtin:
    """Collection of builtin functions"""
    INT_REGEX = re.compile(r'^[-+]?\d+$')
    FLOAT_REGEX = re.compile(r'^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$')

    @staticmethod
    def print(*args, **kwargs):
        args = ['<NONE>' if arg is None else arg for arg in args]

        if len(kwargs) > 0:
            print(*args, kwargs)
        else:
            print(*args)

    @staticmethod
    def coalesce(*args, **kwargs):
        """Returns the first not None element"""
        for arg in args:
            if Builtin.exists(arg):
                return arg

        for v in kwargs.values():
            if Builtin.exists(v):
                return v

        return None

    @staticmethod
    def exists(x=None, *_, **__) -> 'bool':
        """Checks for nullability"""
        return x is not None

    @staticmethod
    def type(x=None, *_, **__) -> 'str':
        """Returns the type name (as string) of the given variable"""
        if x is None:
            return 'none'

        if isinstance(x, (int, float)):
            return 'num'

        if isinstance(x, str):
            return 'str'

        if isinstance(x, bool):
            return 'bool'

        if isinstance(x, Struct):
            return 'struct'

        if isinstance(x, DocRef):
            return 'doc'

        return 'opaque'

    @staticmethod
    def size(*args, **__) -> int:
        if len(args) == 0:
            return 0

        if len(args) == 1:
            x = args[0]
        else:
            return len(args)

        if isinstance(x, (str, Struct)):
            return len(x)

        if isinstance(x, Opaque):
            return 0 if x.unbox() is None else 1

        if isinstance(x, DocRef):
            return 1

        return 0

    @staticmethod
    def substring(x=None, start=0, end: int | None = None, *_, **__) -> 'str':
        if x is None:
            return ""

        x = Builtin.to_str(x)

        if not isinstance(start, (int, float)):
            start = 0
        else:
            start = int(start)

        if end is None or not isinstance(end, (int, float)):
            end = len(x)
        else:
            end = int(end)

        return x[start:end]

    @staticmethod
    def to_num(x=None, *_, **__) -> int | float:
        """Explicit numeric conversion"""
        if isinstance(x, (int, float)):
            return x

        if isinstance(x, bool):
            return 1 if x else 0

        if isinstance(x, str):
            # empty string
            if len(x) == 0:
                return 0

            if x.lower() in ['true', 'false']:
                return int(x.lower() == 'true')

            if re.match(Builtin.INT_REGEX, x):
                return int(x)

            if re.match(Builtin.FLOAT_REGEX, x):
                return float(x)

        return 0

    @staticmethod
    def to_bool(x=None, *_, **__) -> 'bool':
        """Explicit boolean conversion"""
        if isinstance(x, bool):
            return x

        if isinstance(x, (int, float)):
            return False if int(x) == 0 else True

        if isinstance(x, str):
            if len(x) == 0:
                return False

            if x.lower() in ['true', 'false']:
                return x.lower() == 'true'

            if x.lower() == 'none':
                return False

            return bool(x)

        return False

    @staticmethod
    def to_str(x=None, *_, **__) -> 'str':
        """Explicit string conversion"""
        if isinstance(x, bool):
            return str(x).lower()

        if isinstance(x, (int, float)):
            return str(x)

        if isinstance(x, Struct):
            return str(x)

        if x is None:
            return 'none'

        if isinstance(x, str):
            return x

        return str(x)

    @staticmethod
    def num(x=None, *_, **__) -> int | float | None:
        """Implicit (soft) numeric conversion"""
        if x is None:
            return None

        if isinstance(x, (tuple, list, dict, set, Struct, DocRef, Opaque)):
            return None

        return Builtin.to_num(x)

    @staticmethod
    def bool(x=None, *_, **__) -> 'bool | None':
        if x is None:
            return None

        if isinstance(x, Struct):
            return len(x) > 0

        if isinstance(x, DocRef):
            return len(x.node_id) > 0

        if isinstance(x, Opaque):
            return x.identifier >= 0

        return Builtin.to_bool(x)

    @staticmethod
    def str(x=None, *_, **__) -> 'str | None':
        if x is None:
            return None

        return Builtin.to_str(x)

    @staticmethod
    def sin(x, *_, **__) -> float:
        return math.sin(Builtin.to_num(x))

    @staticmethod
    def cos(x, *_, **__) -> float:
        return math.cos(Builtin.to_num(x))

    @staticmethod
    def sqrt(x, *_, **__) -> float:
        return math.sqrt(Builtin.to_num(x))

    @staticmethod
    def ask_llm(llm: AbstractLLMProxy, prompt, *_, **kwargs) -> LLMResponse:
        return llm.invoke(prompt, **kwargs)

    @staticmethod
    def retrieve(knowledge_base: NemantixKnowledgeBase,
                 query,
                 top_k=5,
                 min_score=0.4,
                 *_,
                 doc_type: "list | str | None" = None,
                 content_type: "list | str | None" = None,
                 metadata: "list | str | None" = None,
                 **__
                 ) -> list | None:

        results = knowledge_base.retrieve(query=query, k=top_k, min_score=min_score,
                                          doc_type=doc_type, content_type=content_type,
                                          metadata_filters=metadata)
        docs = []

        for result in results:
            docs.append(DocRef(node_id=result['node_id'], score=result['score'],
                               content=result['content'],
                               breadcrumbs=result['breadcrumbs']))
        return docs

    @staticmethod
    def expand(knowledge_base: NemantixKnowledgeBase, node_id: 'str | list[str]') -> list[DocRef]:
        if not isinstance(node_id, list):
            node_id = [node_id]

        docs = []
        for id_ in set(node_id):
            results = knowledge_base.expand(node_id=id_)

            if isinstance(results, dict):
                results = [results]

            for result in results:
                docs.append(DocRef(node_id=result['node_id'], score=0.0,
                                   content=result['content'], breadcrumbs=''))
        return docs

    @staticmethod
    def extend(knowledge_base: NemantixKnowledgeBase, node_id: 'str | list[str]') -> list[DocRef]:
        if not isinstance(node_id, list):
            node_id = [node_id]

        docs = []
        for id_ in set(node_id):
            results = knowledge_base.extend(node_id=id_)

            for result in [results.get('previous_sibling', None),
                           results.get('next_sibling', None)]:
                if result is None:
                    continue

                assert isinstance(result, dict)
                docs.append(DocRef(node_id=result['node_id'], score=0.0,
                                   content=result['content'], breadcrumbs=''))
        return docs

    @staticmethod
    def generalize(knowledge_base: NemantixKnowledgeBase, node_id: 'str') -> DocRef:
        result = knowledge_base.generalize(node_id=node_id)
        return DocRef(node_id=result['node_id'], score=0.0,
                      content=result['content'], breadcrumbs='')


def compute_similarity(embedder, a, b, a_emb=None, b_emb=None) -> float:
    if isinstance(a_emb, np.ndarray):
        emb_a = a_emb.reshape([-1])
    else:
        emb_a = embedder.embed(a).reshape([-1])

    if isinstance(b_emb, np.ndarray):
        emb_b = b_emb.reshape([-1])
    else:
        emb_b = embedder.embed(b).reshape([-1])

    return float(emb_a @ emb_b.T)


def get_globals() -> dict[str, Any]:
    from nemantix.core.tools import Toolset, tool

    return {
        "Toolset": Toolset,
        "tool": tool,
        "Opaque": Opaque,
        "__builtins__": _allowed_builtins()}


# TODO: could wrap __import__ to improve security? or import commonly used libraries names (pd, np)?
def _allowed_builtins() -> dict:
    from typing import Optional, Set, List, Union, Dict, Any, Callable, Tuple

    return {
        "Exception": Exception, "TypeError": TypeError, "False": False, "min": min, "max": max,
        "print": print, "bool": bool, "dict": dict, "enumerate": enumerate, "filter": filter,
        "float": float, "frozenset": frozenset, "help": help, "int": int, "list": list,
        "map": map, "object": object, "range": range, "reversed": reversed, "set": set,
        "slice": slice, "str": str, "super": super, "tuple": tuple, "type": type, "zip": zip,
        "len": len, "round": round, "isinstance": isinstance, "__build_class__": __build_class__,
        "ValueError": ValueError, "None": None, "True": True, "Optional": Optional, "Set": Set,
        "List": List, "Union": Union, "Dict": Dict, "Any": Any, "any": any, "Callable": Callable,
        "Tuple": Tuple, "__name__": __name__, 'callable': callable, 'hasattr': hasattr, "all": all,
        'getattr': getattr, 'setattr': setattr, "__import__": __import__, 'input': input}
