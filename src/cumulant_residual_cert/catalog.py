"""User-facing data structures for fermionic-word catalogs.

Letters in the dictionary $\\mathcal{L} = \\{I, n, a, a^\\dagger\\}$:

    I       identity         charge 0
    n       number operator  charge 0
    a       annihilation     charge -1
    a_dag   creation         charge +1

A :class:`FermionicWord` is an ordered tuple of letters carrying a per-letter
charge pattern. The partition-lattice arithmetic depends only on the charge
pattern, not on the specific site indices, so the indices are abstract here.

A :class:`Catalog` is a finite collection of words with a shared maximum
length $r$ (the residual-bound order). The most important factory is
:meth:`Catalog.chemistry_r4`, which returns the 5-word chemistry catalog
used for chemistry-catalog residual bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Letter = Literal["I", "n", "a", "a_dag"]

LETTER_CHARGE: dict[str, int] = {
    "I": 0,
    "n": 0,
    "a": -1,
    "a_dag": +1,
}


def _validate_letter(letter: str) -> None:
    if letter not in LETTER_CHARGE:
        raise ValueError(
            f"unknown fermionic letter {letter!r}; expected one of {sorted(LETTER_CHARGE)}"
        )


@dataclass(frozen=True)
class FermionicWord:
    """An ordered word in :math:`\\{I, n, a, a^\\dagger\\}`.

    Parameters
    ----------
    letters : tuple of str
        The letters, in JW order. Must be from ``{"I", "n", "a", "a_dag"}``.
    name : str, optional
        Human-readable name. Defaults to a space-joined letter string.
    """

    letters: tuple[str, ...]
    name: str = ""

    def __post_init__(self) -> None:
        if not self.letters:
            raise ValueError("fermionic word must contain at least one letter")
        for L in self.letters:
            _validate_letter(L)
        if not self.name:
            # Frozen dataclass: must use object.__setattr__.
            object.__setattr__(self, "name", " ".join(self.letters))

    @property
    def length(self) -> int:
        return len(self.letters)

    @property
    def charges(self) -> tuple[int, ...]:
        return tuple(LETTER_CHARGE[L] for L in self.letters)

    @property
    def total_charge(self) -> int:
        return sum(self.charges)

    @property
    def is_charge_neutral(self) -> bool:
        return self.total_charge == 0


@dataclass(frozen=True)
class Catalog:
    """A finite collection of fermionic words sharing a max length :math:`r`."""

    words: tuple[FermionicWord, ...]
    r: int
    name: str = ""

    def __post_init__(self) -> None:
        if self.r < 3:
            raise ValueError("catalog order r must be >= 3 (residual bound is undefined below)")
        if not self.words:
            raise ValueError("catalog must contain at least one word")
        names_seen: set[str] = set()
        for w in self.words:
            if w.length > self.r:
                raise ValueError(
                    f"word {w.name!r} has length {w.length} > r = {self.r}"
                )
            if not w.is_charge_neutral:
                raise ValueError(
                    f"word {w.name!r} is not charge-neutral; this library certifies "
                    f"residuals only for U(1)-invariant states and neutral observables"
                )
            if w.name in names_seen:
                raise ValueError(
                    f"duplicate word name {w.name!r} in catalog; result dicts are "
                    f"keyed by name, so collisions would silently lose entries"
                )
            names_seen.add(w.name)

    @classmethod
    def chemistry_r4(cls) -> Catalog:
        """The 5-word chemistry catalog at $r = 4$.

        Words: $n_i n_j n_k$, $a^\\dagger_i a_j n_k$, $a^\\dagger_i a_j n_k n_\\ell$,
        $a^\\dagger_i a^\\dagger_j a_k a_\\ell$, $n_i n_j n_k n_\\ell$.
        """
        return cls(
            words=(
                FermionicWord(("n", "n", "n"), name="n_i n_j n_k"),
                FermionicWord(("a_dag", "a", "n"), name="a_dag_i a_j n_k"),
                FermionicWord(("a_dag", "a", "n", "n"), name="a_dag_i a_j n_k n_ell"),
                FermionicWord(("a_dag", "a_dag", "a", "a"), name="a_dag_i a_dag_j a_k a_ell"),
                FermionicWord(("n", "n", "n", "n"), name="n_i n_j n_k n_ell"),
            ),
            r=4,
            name="chemistry_r4",
        )

    def __iter__(self):
        return iter(self.words)

    def __len__(self) -> int:
        return len(self.words)

    def __contains__(self, item: object) -> bool:
        return item in self.words


def word(letters: str | tuple[str, ...], name: str = "") -> FermionicWord:
    """Convenience builder.

    Examples
    --------
    >>> word("n n n")
    FermionicWord(letters=('n', 'n', 'n'), name='n n n')
    >>> word(("a_dag", "a", "n"), name="hopping")
    FermionicWord(letters=('a_dag', 'a', 'n'), name='hopping')
    """
    if isinstance(letters, str):
        letters_tuple = tuple(letters.split())
    else:
        letters_tuple = tuple(letters)
    return FermionicWord(letters=letters_tuple, name=name)
