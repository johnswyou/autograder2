"""The answer check (design 7.5, 14.2): a student's stated final answer compared with
the key in code, before the grader sees it.

A number is converted to the key's unit with Pint and compared within tolerance. An
expression is read by this module's own whitelist parser into a tree, and two trees
are compared by evaluating them at fixed points. No text a student or a model wrote
reaches `eval`, SymPy or Pint's parser: a unit passes a whitelist of its own and goes
to Pint one unit name at a time. Length, depth and exponent limits apply before any
evaluation, so every check ends quickly. A reason is fixed text written here, and
whatever the check does not fully understand is "cannot tell", never a mismatch.
"""

import cmath
import functools
import math
import operator
import re
from collections.abc import Callable
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, NamedTuple, cast

import pint

from redpen.schemas.grade import Verdict
from redpen.schemas.key import KeyPart, Policies
from redpen.schemas.reading import FinalAnswer, Reading

MATCH, NO_MATCH, UNSURE = "matches", "does not match", "cannot tell"
MAX_LENGTH = 200  # characters in an expression
MAX_DEPTH = 30  # nested parentheses, signs and powers
MAX_EXPONENT = 64  # size of an exponent that holds no name
MAX_UNIT = 40  # characters in a unit
POINTS = 6  # where two expressions are evaluated
# Why a number in a unit other than the key's, which does not match as read, is
# "cannot tell": a unit Pint misread could make it match.
OTHER_UNIT = "The answer is in a unit other than the key's, and does not match as read."
_ROOT2, _ROOT3 = math.sqrt(2), math.sqrt(3)

_FUNCTIONS: dict[str, Callable[[complex], complex]] = {
    "sqrt": cmath.sqrt,
    "sin": cmath.sin,
    "cos": cmath.cos,
    "tan": cmath.tan,
    "asin": cmath.asin,
    "acos": cmath.acos,
    "atan": cmath.atan,
    "arcsin": cmath.asin,
    "arccos": cmath.acos,
    "arctan": cmath.atan,
    "ln": cmath.log,
    "exp": cmath.exp,
    "log": cmath.log10,  # or natural: both are tried
}
_OPERATORS: dict[str, Callable[[complex, complex], complex]] = {
    "+": complex.__add__,
    "-": complex.__sub__,
    "*": complex.__mul__,
    "/": complex.__truediv__,
}
# A number may carry an exponent (`2.5e3`, `1E-3`): one number, not a number and a
# name.
_TOKEN = re.compile(
    r"\s*(?:((?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)"
    r"|([A-Za-z][A-Za-z0-9_]*)|(\S))",
    re.ASCII,
)
_FACTOR = r"[A-Za-z]{1,20}(?:\^-?[1-4])?"
_PRODUCT = rf"{_FACTOR}(?:\*{_FACTOR})*"
# A product, then `/` and one factor or a product in parentheses, as often as wanted.
# `kg/m*s` is refused: which way it groups is not written.
_UNIT = re.compile(rf"{_PRODUCT}(?:/(?:{_FACTOR}|\({_PRODUCT}\)))*", re.ASCII)

# ("n", number), ("v", name), ("neg", tree), ("f", function, tree), (op, tree, tree)
type Node = (
    tuple[str, complex]
    | tuple[str, str]
    | tuple[str, Node]
    | tuple[str, str, Node]
    | tuple[str, Node, Node]
)


class Check(NamedTuple):
    verdict: Verdict
    reason: str


class _Unreadable(Exception):
    pass


class _Unclear(Exception):
    pass


def check_answer(part: KeyPart, reading: Reading, policies: Policies) -> Check | None:
    """The check of `reading`'s stated final answer against `part`'s key: a number
    when the key's answer has a value, else an expression when it has one, else
    None (nothing code can check). It never raises: whatever a check raises is
    "cannot tell"."""
    try:
        if part.answer.value is not None:
            return _number(part, reading.final_answer, policies)
        if part.answer.expression is not None:
            expression = reading.final_answer.expression
            rel_tol = (
                policies.rel_tol if part.answer.rel_tol is None else part.answer.rel_tol
            )
            return _expression(part.answer.expression, expression, rel_tol)
    except Exception:  # a unit Pint chokes on, a value out of Decimal's range
        return Check(UNSURE, "The check could not be completed.")
    return None


# --- numbers -----------------------------------------------------------------------


def _number(part: KeyPart, stated: FinalAnswer, policies: Policies) -> Check:
    answer = part.answer
    if stated.value is None:
        return Check(UNSURE, "No value is stated.")
    said = (stated.unit or "").strip()
    if not said and (answer.unit or "").strip() and policies.require_units:
        return Check(NO_MATCH, "No unit is stated, and the policies require one.")
    try:
        key_unit = _unit(answer.unit or "")
        unit = _unit(said) if said else key_unit
    except pint.PintError:
        key_unit = unit = None
    if key_unit is None or unit is None:
        return Check(UNSURE, "A unit is not one the check can read.")
    rel_tol = policies.rel_tol if answer.rel_tol is None else answer.rel_tol
    keys = [x for x in [answer.value, *answer.accept_also] if x is not None]
    # Decimal arithmetic with an infinity does not raise: it would match anything.
    if not all(math.isfinite(x) for x in [stated.value, rel_tol, *keys]):
        return Check(UNSURE, "A value or the tolerance is not a finite number.")
    tol, figures = Decimal(str(rel_tol)), not policies.penalise_sig_figs
    written = Decimal(str(stated.value))
    targets = [Decimal(str(x)) for x in keys]
    try:
        # The key's own unit, as read: the same, or one it converts to by exactly 1.
        own = unit == key_unit or _convert(1.0, unit, key_unit) == 1
    except pint.DimensionalityError:
        own = False
    except pint.PintError:
        return Check(UNSURE, "A unit is not one the check can read.")
    try:
        # The value in the key's unit for the tolerance, and each key value in the
        # student's unit for the rounding rule.
        value = _convert(stated.value, unit, key_unit)
        mine = [_convert(x, key_unit, unit) for x in keys]
        found = [
            _how(value, t, written, m, tol, figures)
            for t, m in zip(targets, mine, strict=True)
        ]
    except pint.DimensionalityError:
        found = []
    except pint.PintError:
        return Check(UNSURE, "A unit is not one the check can read.")
    for n, how in enumerate(found):
        whose = "a value the key also accepts" if n else "the key's value"
        if how == "within":
            return Check(MATCH, f"The value is within tolerance of {whose}.")
        if how == "rounded":
            return Check(MATCH, f"The value is {whose}, rounded.")
    if "one figure" in found:
        return Check(
            UNSURE,
            "The value is a key value rounded to one figure, and the reading cannot"
            " show whether more figures were written.",
        )
    # In another unit, a conversion can make an answer match and nothing else: Pint
    # may have read the unit as something the student did not mean (`Mm` for mm).
    if not own:
        return Check(UNSURE, OTHER_UNIT)
    return Check(
        NO_MATCH, "The value is outside the tolerance of every value the key accepts."
    )


def _how(
    value: Decimal,
    want: Decimal,
    written: Decimal,
    mine: Decimal,
    tol: Decimal,
    figures: bool,
) -> str:
    """How a stated value agrees with a key value: "within" tolerance (`value` and
    `want` in the key's unit; a key of zero takes the tolerance as absolute),
    "rounded" (the value as `written` is the key value `mine`, in the student's unit,
    rounded to two figures or more), "one figure" (the same, but `written` has fewer
    than two significant digits), or "" (neither). The reading keeps no trailing
    zeros, so any rounding of two figures or more counts."""
    if abs(value - want) <= tol * (abs(want) or 1):
        return "within"
    if figures and any(written == _figures(mine, k) for k in range(2, 18)):
        digits = len(written.normalize().as_tuple().digits)
        return "rounded" if digits >= 2 else "one figure"
    return ""


@functools.cache
def _registry() -> pint.UnitRegistry:
    return pint.UnitRegistry()


def _unit(text: str) -> pint.Unit | None:
    """The unit `text` writes, or None when it is not unit names joined by `*`, a
    space or `/`, with powers from -4 to 4. The text never reaches Pint's parser:
    each name is looked up on its own. Raises `pint.PintError` for a name Pint does
    not know."""
    if len(text) > MAX_UNIT:
        return None
    text = re.sub(r"\s*([*/^()])\s*", r"\1", text.strip())
    text = re.sub(r"\s+", "*", text)
    if not text:
        return _registry().dimensionless
    if not _UNIT.fullmatch(text):
        return None
    first, *under = text.split("/")
    factors = [(f, 1) for f in first.split("*")]
    factors += [(f, -1) for group in under for f in group.strip("()").split("*")]
    units: list[pint.Unit] = []
    for factor, sign in factors:
        name, _, power = factor.partition("^")
        units.append(_registry().Unit(name) ** (sign * int(power or 1)))
    return functools.reduce(operator.mul, units)


def converts(unit: str, to: str) -> bool:
    """Whether the whitelist reads both units and Pint converts the first to the
    second: they are of one dimension. It never raises."""
    try:
        found, target = _unit(unit), _unit(to)
        if found is None or target is None:
            return False
        _convert(1.0, found, target)
    except Exception:  # a name Pint does not know, another dimension
        return False
    return True


def _convert(value: float, unit: pint.Unit, to: pint.Unit) -> Decimal:
    if unit == to:
        return Decimal(str(value))
    quantity = cast(Any, _registry().Quantity(value, unit))  # Pint's types are partial
    return Decimal(str(quantity.to(to).magnitude))


def _figures(x: Decimal, k: int) -> Decimal:
    """`x` rounded to `k` significant figures, half away from zero."""
    return x.quantize(Decimal(1).scaleb(x.adjusted() - k + 1), ROUND_HALF_UP)


# --- expressions -------------------------------------------------------------------


def _expression(key_text: str, said_text: str | None, rel_tol: float) -> Check:
    if not (said_text or "").strip():
        return Check(UNSURE, "No expression is stated.")
    try:
        key = _parse(key_text)
    except (_Unreadable, _Unclear, ArithmeticError, ValueError):
        return Check(UNSURE, "The key's expression is not one the check can read.")
    try:
        said = _parse(said_text or "")
    except _Unclear:
        return Check(UNSURE, "The order of operations in the expression is unclear.")
    except (_Unreadable, ArithmeticError, ValueError):
        return Check(UNSURE, "The expression is not one the check can read.")
    # Compared only over the same names: another name may be another symbol for the
    # same quantity, a unit written after it, or a constant put in as a number.
    known, used = _names(key, "v"), _names(said, "v")
    if used != known:
        extra = used - known
        if extra and all(_alike(name, known) for name in extra):
            return Check(
                UNSURE,
                "A name the key's lacks may be one of its own written another way,"
                " such as in another letter case.",
            )
        return Check(UNSURE, "The expression and the key's do not use the same names.")
    logs = [cmath.log10]
    if "log" in _names(key, "f") | _names(said, "f"):
        logs.append(cmath.log)
    # A tolerance that is not a finite number would match anything.
    tol = rel_tol if math.isfinite(rel_tol) else 0.0
    found = {_compare(key, said, log, tol) for log in logs}
    if len(found) > 1:
        return Check(UNSURE, "Whether log is base 10 or natural decides it.")
    return found.pop()


def _parse(text: str) -> Node:
    """The tree of `text`, read by the whitelist grammar:

        expr  := term (("+" | "-") term)*
        term  := unary (("*" | "/") unary | power)*    side by side means times
        unary := ("+" | "-") unary | power
        power := atom ("^" unary)?
        atom  := number | name | function "(" expr ")" | "(" expr ")"

    after dropping a leading "name =". Raises `_Unclear` for a product written side
    by side after a `/` (`a/2b`) or after a power (`x^2y`), and `_Unreadable` for
    anything else outside the grammar or its limits."""
    if len(text) > MAX_LENGTH:
        raise _Unreadable
    text = re.sub(r"^\s*[A-Za-z][A-Za-z0-9_]*\s*=", "", text, count=1, flags=re.ASCII)
    parser = _Parser(text)
    tree = parser.expr()
    if parser.at < len(parser.tokens):
        raise _Unreadable
    return tree


class _Parser:
    def __init__(self, text: str) -> None:
        self.tokens = [
            ("number", n) if n else ("name", w) if w else (c, c)
            for n, w, c in _TOKEN.findall(text)
        ]
        self.at = 0
        self.depth = 0
        self.raised = False  # the last factor read ends in a power

    def kind(self) -> str | None:
        return self.tokens[self.at][0] if self.at < len(self.tokens) else None

    def take(self) -> tuple[str | None, str]:
        token = self.tokens[self.at] if self.at < len(self.tokens) else (None, "")
        self.at += 1
        return token

    def expect(self, kind: str) -> None:
        if self.take()[0] != kind:
            raise _Unreadable

    def deeper[T](self, read: Callable[[], T]) -> T:
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise _Unreadable
        found = read()
        self.depth -= 1
        return found

    def expr(self) -> Node:
        # Each term of a chain such as x+x+... adds a level to the tree without
        # going deeper here: only MAX_LENGTH bounds that depth, which _value
        # recurses through.
        tree = self.deeper(self.term)
        while (kind := self.kind()) in ("+", "-"):
            self.take()
            tree = (str(kind), tree, self.deeper(self.term))
        return tree

    def term(self) -> Node:
        tree, divided = self.unary(), False
        while (kind := self.kind()) is not None:
            if kind in ("*", "/"):
                self.take()
                divided |= kind == "/"
                tree = (kind, tree, self.unary())
            elif kind in ("name", "("):
                if divided or self.raised:
                    raise _Unclear
                tree = ("*", tree, self.power())
            else:
                break
        return tree

    def unary(self) -> Node:
        if (kind := self.kind()) in ("+", "-"):
            self.take()
            tree = self.deeper(self.unary)
            return ("neg", tree) if kind == "-" else tree
        return self.power()

    def power(self) -> Node:
        base = self.atom()
        if self.kind() != "^":
            self.raised = False
            return base
        self.take()
        exponent = self.deeper(self.unary)
        # Exponents inside it were held to the limit already, so this ends quickly.
        if not _names(exponent, "v") and abs(_value(exponent, {}, cmath.log10)) > (
            MAX_EXPONENT
        ):
            raise _Unreadable
        self.raised = True
        return ("^", base, exponent)

    def atom(self) -> Node:
        kind, text = self.take()
        if kind == "number":
            return ("n", complex(float(text)))
        if kind == "(":
            tree = self.expr()
            self.expect(")")
            return tree
        if kind != "name" or "__" in text:
            raise _Unreadable
        if text in _FUNCTIONS:
            self.expect("(")
            tree = ("f", text, self.expr())
            self.expect(")")
            return tree
        if self.kind() == "(":  # a call of a function not on the list
            raise _Unreadable
        return ("n", complex(math.pi)) if text == "pi" else ("v", text)


def _names(tree: Node, kind: str) -> set[str]:
    """The variable names (`kind` "v") or function names ("f") in `tree`."""
    found = {str(tree[1])} if tree[0] == kind else set[str]()
    for child in tree[1:]:
        if isinstance(child, tuple):
            found |= _names(child, kind)
    return found


def _alike(name: str, known: set[str]) -> bool:
    """Whether `name`, which the key's expression lacks, may still be the key's:
    another case, no underscore, `e` or `pi`, or the key's names side by side."""
    parts = {_fold(k) for k in known} | {"e", "pi"}
    word = _fold(name)
    ends = {0}
    for i in range(1, len(word) + 1):
        if any(i - len(p) in ends and word.endswith(p, 0, i) for p in parts):
            ends.add(i)
    return len(word) in ends


def _fold(name: str) -> str:
    return name.replace("_", "").casefold()


def _value(
    tree: Node, at: dict[str, complex], log: Callable[[complex], complex]
) -> complex:
    match tree:
        case ("n", complex() as number):
            return number
        case ("v", str() as name):
            return at[name]
        case ("neg", tuple() as inner):
            return -_value(inner, at, log)
        case ("f", str() as function, tuple() as inner):
            use = log if function == "log" else _FUNCTIONS[function]
            return use(_value(inner, at, log))
        case (str() as op, tuple() as left, tuple() as right):
            a = _value(left, at, log)
            b = _value(right, at, log)
            if op != "^":
                return _OPERATORS[op](a, b)
            # A whole exponent as an int: (-2)^2 is then exactly 4, not 4 - 1e-16i.
            whole = b.imag == 0 and b.real.is_integer() and abs(b.real) <= MAX_EXPONENT
            return a ** int(b.real) if whole else a**b
        case _:
            raise _Unreadable


def evaluate(text: str) -> float | None:
    """The value of `text`, a computation in numbers only, read by the whitelist
    parser with `log` as base 10. None when the parser does not read it, when it
    holds a name, or when its value is not a finite real number."""
    try:
        tree = _parse(text)
        if _names(tree, "v"):
            return None
        value = _value(tree, {}, cmath.log10)
    except (_Unreadable, _Unclear, ArithmeticError, ValueError):
        return None
    if not cmath.isfinite(value) or abs(value.imag) > 1e-9 * max(1, abs(value.real)):
        return None
    return value.real


def _compare(
    key: Node, said: Node, log: Callable[[complex], complex], rel_tol: float
) -> Check:
    """Both trees evaluated at `POINTS` fixed points of their names, between 0.5 and
    3. A point where either fails, or is not real, is left out. Within `rel_tol` of
    the key's value at every point is a match, as a number is (`_how`)."""
    names = sorted(_names(key, "v") | _names(said, "v"))
    agree: list[bool] = []
    near: list[bool] = []
    for i in range(POINTS):
        # Irrational steps, so no two names share a value at any point.
        at = {
            name: complex(0.5 + 2.5 * (((i + 1) * _ROOT2 + (j + 1) * _ROOT3) % 1))
            for j, name in enumerate(names)
        }
        try:
            a, b = _value(key, at, log), _value(said, at, log)
        except (ArithmeticError, ValueError):
            continue
        real = (abs(x.imag) <= 1e-9 * max(1, abs(x.real)) for x in (a, b))
        if cmath.isfinite(a) and cmath.isfinite(b) and all(real):
            agree.append(math.isclose(a.real, b.real, rel_tol=1e-9, abs_tol=1e-12))
            near.append(abs(b.real - a.real) <= rel_tol * (abs(a.real) or 1))
    if len(agree) < 3:
        return Check(UNSURE, "The expressions could not be evaluated at enough points.")
    if all(agree):
        return Check(MATCH, "The expression equals the key's at every point tried.")
    if all(near):
        return Check(
            MATCH,
            "The expression is within tolerance of the key's at every point tried.",
        )
    if not any(agree):
        return Check(NO_MATCH, "The expression differs from the key's at every point.")
    return Check(UNSURE, "The expression equals the key's at some points, not all.")
