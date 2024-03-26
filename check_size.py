from typing import Dict
import re

class SizeChecker:
    def __init__(self):
        self.size_dict: Dict[str, int] = {}

        self.__clear = self.clear
        self.__register = self.register
        self.__register_like = self.register_like
        self.__check = self.check

    def __empty(self, *args, **kwargs):
        pass

    def enable(self):
        """
        Enable all size check.
        """
        self.clear = self.__clear
        self.register = self.__register
        self.register_like = self.__register_like
        self.check = self.__check
    
    def disable(self):
        """
        Disable all size check.
        """
        self.clear = self.__empty
        self.register = self.__empty
        self.register_like = self.__empty
        self.check = self.__empty

    def clear(self):
        self.size_dict = {}

    def register(self, name: str, size):
        """
        Examples:

        * Register size from an integer

            >>> checker = SizeChecker()
            >>> checker.register("M", 3)
            >>> checker.size_dict
            {'M': 3}

        * Register size from a tuple

            >>> checker.register("N, K, K, L, ", (3, 4, 4, 2))
            >>> checker.size_dict
            {'M': 3, 'N': 3, 'K': 4, 'L': 2}

        * Exceptions

            >>> # re-assign different values
            >>> checker.register("M", 5)
            ValueError: 'M' double-registered with a different value

            >>> # contradicting values
            >>> checker.register("P, R, R, ", (3, 2, 5))
            ValueError: 'R' double-registered with a different value
            >>> checker.size_dict # will be partially updated
            {'M': 3, 'N': 3, 'K': 4, 'L': 2, 'P': 3, 'R': 2}
        """
        if isinstance(size, int):
            size = [size, ]
        sizes = list(size)
        names = re.split("[ \t]*,[ \t]*", name.strip().rstrip(",").rstrip())

        assert len(names) == len(sizes), f"size lengths do not match, expected {len(names)}, got {len(size)}"
        for name, size in zip(names, sizes):
            try:
                int(name)
            except ValueError:
                if name in self.size_dict and self.size_dict[name] != size:
                    raise ValueError(f"'{name}' double-registered with a different value")
                self.size_dict[name] = int(size)
            else:
                raise ValueError(f"invalid name '{name}'")
                
    def register_like(self, name: str, tensor):
        """
        Example:

            >>> import numpy as np
            >>> a = np.random.randn(3, 2, 2)
            >>> checker = SizeChecker()
            >>> checker.register_like("N, M, M", a)
            >>> checker.size_dict
            {'N': 3, 'M': 2}

            >>> b = np.arange(5)
            >>> checker.register_like("K, ", b)
            >>> checker.size_dict
            {'N': 3, 'M': 2, 'K': 5}

        """
        assert hasattr(tensor, "shape"), "input tensor has no attribute 'shape'"
        self.register(name, tensor.shape)

    def check(self, asserted_size: str, tensor):
        """
        Example:

            >>> import torch
            >>> import torch.nn as nn
            >>> a = torch.ones(3, 2)
            >>> b = nn.Linear(2, 7)
            >>> checker = SizeChecker()
            >>> # (M, N) = (3, 2)
            >>> checker.register_like("M, N", a)
            >>> # (K, N) = (7, 2)
            >>> checker.register_like("K, N", b.weight)
            >>> checker.check("M, K", b(a)) # check passed
            >>> checker.check("4, K", b(a)) # check failed
            AssertionError: expected shape (4, 7) but got (3, 7)
            >>> checker.check("*, K", b(a)) # wildcard, check passed
        
        """
        assert hasattr(tensor, "shape"), "input tensor has no attribute 'shape'"
        asserted_shape_str = re.split("[ \t]*,[ \t]*", 
                                      asserted_size.strip().rstrip(",").rstrip())
        asserted_shape = []
        for size in asserted_shape_str:
            if size == "*":
                asserted_shape.append(size)
                continue
            try:
                asserted_shape.append(int(size))
            except ValueError:
                try:
                    asserted_shape.append(self.size_dict[size])
                except KeyError:
                    raise ValueError(f"Unknown size string '{size}'")
        asserted_shape = tuple(asserted_shape)
        real_shape = tuple(tensor.shape)
        assert equals(asserted_shape, real_shape), f"expected shape {asserted_shape} but got {real_shape}"


# Decorator, use @CheckSize(True) or @CheckSize(False) to decorate a class
# then one can use "self.checker"
class CheckSize:
    def __init__(self, enable: bool = True):
        self.enable = enable
    
    def __call__(self, cls):
        init = cls.__init__

        def __init__(__self__, *args, **kwargs):
            __self__.checker = SizeChecker()
            if self.enable:
                __self__.checker.enable()
            else:
                __self__.checker.disable()
            init(__self__, *args, **kwargs)
        
        cls.__init__ = __init__

        return cls

def equals(tuple1, tuple2) -> bool:
    """
    Allow wildcards.
    """
    if len(tuple1) != len(tuple2):
        return False
    
    for (e1, e2) in zip(tuple1, tuple2):
        if e1 == "*" or e2 == "*":
            continue
        if e1 != e2:
            return False
    
    return True

