"""A tiny string->class registry, the mechanism behind the config-driven swaps.

Each swappable axis (coarse sources, conditioning strategies) owns one Registry.
A class registers itself with a key; the factory looks it up by the config string.
Adding a new variant is a decorator + a config value, never a training-loop edit.
"""

from __future__ import annotations

from typing import Callable, Dict, Generic, Type, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, name: str) -> None:
        self.name = name
        self._entries: Dict[str, Type[T]] = {}

    def register(self, key: str) -> Callable[[Type[T]], Type[T]]:
        def deco(cls: Type[T]) -> Type[T]:
            if key in self._entries:
                raise KeyError(f"{self.name}: key {key!r} already registered")
            self._entries[key] = cls
            return cls

        return deco

    def get(self, key: str) -> Type[T]:
        if key not in self._entries:
            raise KeyError(
                f"{self.name}: unknown key {key!r}; "
                f"available: {sorted(self._entries)}"
            )
        return self._entries[key]

    def create(self, key: str, *args, **kwargs) -> T:
        return self.get(key)(*args, **kwargs)

    def keys(self):
        return sorted(self._entries)
