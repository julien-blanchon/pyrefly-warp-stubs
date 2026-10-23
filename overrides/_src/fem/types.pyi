# Overrides merged into warp/_src/fem/types.py by tools/generate_stubs.py.
#
# Inside integrands, `u(s)` evaluates a field (`Field.call_operator = inner`)
# and `domain(s)` gives a position (`Domain.call_operator = position`), both
# resolved by warp.fem's code generator per field type. Upstream declares
# neither class callable. Stubbing this module also turns its `@wp.func`
# overloads by redefinition (`make_free_sample`, ...) into `typing.overload`s.

from typing import Any

class Field:
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

class Domain:
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
