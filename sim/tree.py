"""Rooted, time-calibrated trees with known node ages.

Ages are measured backwards from the present (age 0 = present day), in
arbitrary time units (years, or millennia — the simulator does not care, as
long as rates are expressed in the same unit). Branch length of a node is
``parent.age - node.age``.

Supports:
  * Newick parsing / writing (branch lengths in time units, internal labels).
  * Yule (pure-birth) tree simulation conditioned on the number of tips and
    rescaled to a fixed root age.
  * Deterministic node ordering (post-order) so results are reproducible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np


@dataclass(eq=False)
class Node:
    name: Optional[str] = None
    age: float = 0.0
    parent: Optional["Node"] = None
    children: list["Node"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def branch_length(self) -> float:
        """Time between this node and its parent (0 for the root)."""
        if self.parent is None:
            return 0.0
        return self.parent.age - self.age

    def add_child(self, child: "Node") -> None:
        child.parent = self
        self.children.append(child)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Node(name={self.name!r}, age={self.age:.4g}, n_children={len(self.children)})"


class Tree:
    """A rooted tree whose nodes carry absolute ages."""

    def __init__(self, root: Node):
        if root.parent is not None:
            raise ValueError("root must not have a parent")
        self.root = root
        self._name_internal_nodes()
        self.validate()

    # ------------------------------------------------------------------ traversal
    def postorder(self) -> Iterator[Node]:
        stack: list[tuple[Node, bool]] = [(self.root, False)]
        while stack:
            node, visited = stack.pop()
            if visited:
                yield node
            else:
                stack.append((node, True))
                for child in reversed(node.children):
                    stack.append((child, False))

    def preorder(self) -> Iterator[Node]:
        stack = [self.root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children))

    def leaves(self) -> list[Node]:
        return [n for n in self.preorder() if n.is_leaf]

    def internal_nodes(self) -> list[Node]:
        return [n for n in self.preorder() if not n.is_leaf]

    def branches(self) -> list[Node]:
        """All non-root nodes; each identifies the branch above it."""
        return [n for n in self.preorder() if not n.is_root]

    @property
    def leaf_names(self) -> list[str]:
        return [n.name for n in self.leaves()]  # type: ignore[misc]

    @property
    def root_age(self) -> float:
        return self.root.age

    def tree_length(self) -> float:
        return sum(n.branch_length for n in self.branches())

    def node_by_name(self, name: str) -> Node:
        for n in self.preorder():
            if n.name == name:
                return n
        raise KeyError(name)

    def mrca(self, names: list[str]) -> Node:
        """Most recent common ancestor of the named nodes."""
        targets = [self.node_by_name(nm) for nm in names]
        paths = []
        for t in targets:
            path = []
            n: Optional[Node] = t
            while n is not None:
                path.append(n)
                n = n.parent
            paths.append(path)
        common = set(map(id, paths[0]))
        for p in paths[1:]:
            common &= set(map(id, p))
        for n in paths[0]:
            if id(n) in common:
                return n
        raise RuntimeError("nodes do not share a root")  # pragma: no cover

    # ------------------------------------------------------------------ checks
    def _name_internal_nodes(self) -> None:
        used = {n.name for n in self.preorder() if n.name}
        i = 0
        for n in self.postorder():
            if not n.name:
                while f"n{i}" in used:
                    i += 1
                n.name = f"n{i}"
                used.add(n.name)

    def validate(self, tol: float = 1e-9) -> None:
        names = [n.name for n in self.preorder()]
        if len(names) != len(set(names)):
            raise ValueError("node names must be unique")
        for n in self.preorder():
            if n.age < -tol:
                raise ValueError(f"negative age at {n.name}")
            for c in n.children:
                if c.parent is not n:
                    raise ValueError("inconsistent parent pointer")
                if c.age > n.age + tol:
                    raise ValueError(f"child {c.name} older than parent {n.name}")

    # ------------------------------------------------------------------ I/O
    def to_newick(self, precision: int = 6, internal_labels: bool = True) -> str:
        def fmt(x: float) -> str:
            return f"{x:.{precision}f}".rstrip("0").rstrip(".") or "0"

        def rec(n: Node) -> str:
            s = ""
            if n.children:
                s = "(" + ",".join(rec(c) for c in n.children) + ")"
                if internal_labels and n.name:
                    s += _quote(n.name)
            else:
                s = _quote(n.name or "")
            if not n.is_root:
                s += ":" + fmt(n.branch_length)
            return s

        return rec(self.root) + ";"

    @classmethod
    def from_newick(cls, newick: str, tip_ages: Optional[dict[str, float]] = None) -> "Tree":
        """Parse a Newick string with branch lengths.

        Node ages are reconstructed from branch lengths. By default the tree is
        assumed ultrametric-ish and the youngest tip is placed at age 0; pass
        ``tip_ages`` to anchor tips explicitly (all tips must then be given,
        and the implied ages must be consistent).
        """
        root, depths = _parse_newick(newick)
        # depths: distance from root to node
        max_depth = max(depths[id(n)] for n in _iter(root) if not n.children)
        for n in _iter(root):
            n.age = max_depth - depths[id(n)]
        if tip_ages is not None:
            leaves = [n for n in _iter(root) if not n.children]
            missing = {n.name for n in leaves} - set(tip_ages)
            if missing:
                raise ValueError(f"tip_ages missing for {sorted(missing)}")
            # shift so that the first tip matches; then check consistency
            first = leaves[0]
            shift = tip_ages[first.name] - first.age
            for n in _iter(root):
                n.age += shift
            for n in leaves:
                if abs(n.age - tip_ages[n.name]) > 1e-6 * max(1.0, root.age):
                    raise ValueError(f"branch lengths inconsistent with tip age of {n.name}")
        # clean float noise on near-zero tip ages
        for n in _iter(root):
            if abs(n.age) < 1e-12:
                n.age = 0.0
        return cls(root)

    def copy(self) -> "Tree":
        return Tree.from_newick(self.to_newick(precision=12))


# ---------------------------------------------------------------------- simulation


def simulate_yule(n_tips: int, root_age: float, rng: np.random.Generator,
                  tip_prefix: str = "L") -> Tree:
    """Simulate a Yule tree with ``n_tips`` contemporaneous tips, rescaled so
    the root has age ``root_age``.

    The topology and relative node heights come from the standard pure-birth
    process (birth rate 1, run until ``n_tips`` lineages exist); the whole tree
    is then rescaled so the root age is exactly ``root_age``.
    """
    if n_tips < 2:
        raise ValueError("need at least two tips")
    if root_age <= 0:
        raise ValueError("root_age must be positive")

    # Forward-time process: time t measured from the root (t=0) forwards.
    root = Node(age=0.0)  # temporarily store forward split time in .age
    active: list[Node] = [root]
    t = 0.0
    split_times: dict[int, float] = {id(root): 0.0}
    while len(active) < n_tips:
        k = len(active)
        t += rng.exponential(1.0 / k)
        i = int(rng.integers(k))
        parent = active.pop(i)
        split_times[id(parent)] = t
        for _ in range(2):
            c = Node()
            parent.add_child(c)
            active.append(c)
    # Final waiting time to the present: the tips extend to the time of the
    # next (unobserved) event, which is the standard "stop at n tips" choice
    # with the extra exponential so that tip branches are not zero-length.
    t_end = t + rng.exponential(1.0 / len(active))
    scale = root_age / t_end
    for node in _iter(root):
        if node.children:
            node.age = (t_end - split_times[id(node)]) * scale
        else:
            node.age = 0.0
    # Name tips in a stable order
    leaves = [n for n in _iter(root) if not n.children]
    for j, leaf in enumerate(leaves):
        leaf.name = f"{tip_prefix}{j + 1}"
    root.age = root_age  # avoid rounding drift
    return Tree(root)


def simulate_yule_given_root(n_tips: int, root_age: float, birth_rate: float,
                             rng: np.random.Generator, tip_prefix: str = "L") -> Tree:
    """Draw a tree from the Yule prior *conditioned on the root age*.

    This is exactly the density BEAST2's ``YuleModel`` uses with
    ``conditionalOnRoot="true"`` (rho = 1, a = 0): given root age T, the
    n - 2 non-root internal node ages are iid Exp(birth_rate) truncated to
    (0, T), and the ranked labelled topology is uniform (built here by
    merging uniformly random pairs of lineages, youngest node first).
    """
    if n_tips < 2:
        raise ValueError("need at least two tips")
    if root_age <= 0 or birth_rate <= 0:
        raise ValueError("root_age and birth_rate must be positive")
    # Inverse-CDF sampling from Exp(r) truncated to (0, T)
    u = rng.random(n_tips - 2)
    ages = -np.log1p(-u * (1.0 - np.exp(-birth_rate * root_age))) / birth_rate
    ages = np.sort(ages)
    lineages = [Node(name=f"{tip_prefix}{i + 1}", age=0.0) for i in range(n_tips)]
    for age in list(ages) + [root_age]:
        i, j = sorted(rng.choice(len(lineages), size=2, replace=False))
        parent = Node(age=float(age))
        b = lineages.pop(j)
        a = lineages.pop(i)
        parent.add_child(a)
        parent.add_child(b)
        lineages.append(parent)
    (root,) = lineages
    return Tree(root)


# ---------------------------------------------------------------------- helpers


def _iter(root: Node) -> Iterator[Node]:
    stack = [root]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.children))


_SAFE_LABEL = re.compile(r"^[A-Za-z0-9_.\-]+$")


def _quote(label: str) -> str:
    if label == "" or _SAFE_LABEL.match(label):
        return label
    return "'" + label.replace("'", "''") + "'"


def _parse_newick(s: str) -> tuple[Node, dict[int, float]]:
    s = s.strip()
    if not s.endswith(";"):
        raise ValueError("Newick string must end with ';'")
    s = s[:-1]
    pos = 0
    n = len(s)

    def skip_ws() -> None:
        nonlocal pos
        while pos < n and s[pos].isspace():
            pos += 1

    def read_label() -> Optional[str]:
        nonlocal pos
        skip_ws()
        if pos < n and s[pos] == "'":
            pos += 1
            out = []
            while pos < n:
                if s[pos] == "'":
                    if pos + 1 < n and s[pos + 1] == "'":
                        out.append("'")
                        pos += 2
                        continue
                    pos += 1
                    break
                out.append(s[pos])
                pos += 1
            return "".join(out)
        start = pos
        while pos < n and s[pos] not in ",():;[" and not s[pos].isspace():
            pos += 1
        lab = s[start:pos]
        return lab or None

    def read_length() -> float:
        nonlocal pos
        skip_ws()
        if pos < n and s[pos] == "[":  # skip comments like [&rate=...]
            depth = 0
            while pos < n:
                if s[pos] == "[":
                    depth += 1
                elif s[pos] == "]":
                    depth -= 1
                    if depth == 0:
                        pos += 1
                        break
                pos += 1
            skip_ws()
        if pos < n and s[pos] == ":":
            pos += 1
            skip_ws()
            start = pos
            while pos < n and (s[pos].isdigit() or s[pos] in ".-+eE"):
                pos += 1
            return float(s[start:pos])
        return 0.0

    def parse_subtree() -> tuple[Node, float]:
        nonlocal pos
        skip_ws()
        node = Node()
        if pos < n and s[pos] == "(":
            pos += 1
            while True:
                child, bl = parse_subtree()
                child_bl[id(child)] = bl
                node.add_child(child)
                skip_ws()
                if s[pos] == ",":
                    pos += 1
                    continue
                if s[pos] == ")":
                    pos += 1
                    break
                raise ValueError(f"unexpected character {s[pos]!r} at {pos}")
        node.name = read_label()
        bl = read_length()
        return node, bl

    child_bl: dict[int, float] = {}
    root, _ = parse_subtree()
    skip_ws()
    if pos != n:
        raise ValueError(f"trailing characters in Newick at {pos}")
    for nd in _iter(root):
        if not nd.children and not nd.name:
            raise ValueError("all tips must be labelled")
    depths: dict[int, float] = {id(root): 0.0}
    for nd in _iter(root):
        for c in nd.children:
            depths[id(c)] = depths[id(nd)] + child_bl[id(c)]
    return root, depths
