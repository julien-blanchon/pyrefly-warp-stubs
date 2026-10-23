# Overrides merged into warp/_src/fem/operator.py by tools/generate_stubs.py.
#
# Inside integrands, operators (`fem.inner(u, s)`, `fem.grad(u, s)`) and other
# integrands are called like functions; warp.fem's code generator resolves
# those calls per field type. Upstream declares neither class callable, which
# made every such call a `not-callable` error. The result depends on the field
# (scalar, vector, matrix), so it is not known statically.

from typing import Any
from typing import overload as _overload

class Integrand:
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

class Operator:
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

# `@integrand` and `@integrand(kernel_options=...)`: upstream's two return paths
# made every decorated integrand a union of `Integrand` and the inner wrapper.
@_overload
def integrand(func: Callable[..., Any], kernel_options: dict[str, Any] | None = None) -> Integrand: ...
@_overload
def integrand(
    func: None = None, kernel_options: dict[str, Any] | None = None
) -> Callable[[Callable[..., Any]], Integrand]: ...
