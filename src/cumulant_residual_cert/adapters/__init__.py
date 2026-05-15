"""Chemistry-framework adapters.

Each adapter provides a single function that takes a framework-native state
representation and returns a structured estimate of $\\Delta$ and the
corresponding certified bias bar. Adapters are optional dependencies; import
the submodule you need:

>>> from cumulant_residual_cert.adapters import pyscf as cr_pyscf       # doctest: +SKIP
>>> from cumulant_residual_cert.adapters import openfermion as cr_of    # doctest: +SKIP
>>> from cumulant_residual_cert.adapters import qiskit_nature as cr_qn  # doctest: +SKIP

If the relevant chemistry stack is not installed, importing the submodule
raises ``ImportError`` with a clear pointer to the matching extras install
(``uv add 'cumulant-residual-cert[pyscf]'`` etc.).
"""
