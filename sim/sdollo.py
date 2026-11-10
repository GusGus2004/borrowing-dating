"""Forward simulation of cognate-class evolution under the stochastic Dollo model.

Model (per concept c, per branch b):

* Cognate classes for concept c are born as a Poisson process at rate
  ``lambda * r_c * rho_b`` (per unit time).
* Each existing class is lost independently at rate ``mu * r_c * rho_b``.
* A class, once lost on a lineage, never reappears there (Dollo); a class is
  born exactly once in the whole tree.

Here ``r_c`` is the concept's rate multiplier (gamma-distributed across
concepts, mean 1; discretised into ``k`` equal-probability categories by the
Yang 1994 mean method, or drawn continuously if ``gamma_categories == 0``)
and ``rho_b`` is the branch's clock rate multiplier (uncorrelated lognormal
with mean 1; ``clock_sd == 0`` gives a strict clock).

Because ``r_c`` and ``rho_b`` scale birth and death together, the stationary
number of classes per concept is Poisson(lambda / mu) regardless of rate.
The root is drawn from this stationary distribution.

Simulation is exact (no time discretisation): on a branch of effective
length ``tau = t * r_c * rho_b`` a parental class survives with probability
``exp(-mu * tau)``; new births arrive at uniform times ``s`` in ``[0, tau]``
and survive to the end of the branch with probability ``exp(-mu * (tau - s))``.
Classes born and lost on the same branch are never observable and are not
stored.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
from scipy import special, stats

from .tree import Tree

MISSING = -1  # matrix code for "language has no form for this concept"


@dataclass(frozen=True)
class SDolloParams:
    n_concepts: int = 200
    birth_rate: float = 2e-4      # lambda: new classes per concept per time unit
    loss_rate: float = 2e-4       # mu: loss rate per class per time unit
    gamma_shape: Optional[float] = None  # alpha; None = no across-concept variation
    gamma_categories: int = 4     # discrete-gamma categories; 0 = continuous gamma
    clock_sd: float = 0.0         # sd of log branch rate; 0 = strict clock

    def __post_init__(self) -> None:
        if self.n_concepts < 1:
            raise ValueError("n_concepts must be >= 1")
        if self.birth_rate <= 0 or self.loss_rate <= 0:
            raise ValueError("birth_rate and loss_rate must be positive")
        if self.gamma_shape is not None and self.gamma_shape <= 0:
            raise ValueError("gamma_shape must be positive")
        if self.gamma_categories < 0:
            raise ValueError("gamma_categories must be >= 0")
        if self.clock_sd < 0:
            raise ValueError("clock_sd must be >= 0")

    @property
    def stationary_mean(self) -> float:
        """Expected number of classes per concept per language."""
        return self.birth_rate / self.loss_rate


@dataclass
class SDolloResult:
    tree: Tree
    params: SDolloParams
    seed: Optional[int]
    concept_rates: np.ndarray            # (n_concepts,)
    branch_rates: dict[str, float]       # node name -> clock multiplier of branch above it
    class_concept: np.ndarray            # (n_classes,) concept index of each class
    class_birth_node: list[str]          # node at the bottom of the birth branch ("root" origin = root name)
    class_birth_age: np.ndarray          # (n_classes,) absolute age of birth (root classes: root age)
    leaf_names: list[str]
    presence: np.ndarray                 # (n_leaves, n_classes) bool
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ views
    @property
    def n_classes(self) -> int:
        return self.presence.shape[1]

    def observed(self) -> "SDolloResult":
        """Drop classes absent from every leaf (they are unobservable)."""
        keep = self.presence.any(axis=0)
        return SDolloResult(
            tree=self.tree,
            params=self.params,
            seed=self.seed,
            concept_rates=self.concept_rates,
            branch_rates=self.branch_rates,
            class_concept=self.class_concept[keep],
            class_birth_node=[b for b, k in zip(self.class_birth_node, keep) if k],
            class_birth_age=self.class_birth_age[keep],
            leaf_names=self.leaf_names,
            presence=self.presence[:, keep],
            extra=dict(self.extra, n_unobserved_dropped=int((~keep).sum())),
        )

    def concept_counts(self) -> np.ndarray:
        """(n_leaves, n_concepts) number of classes each language has per concept."""
        out = np.zeros((len(self.leaf_names), self.params.n_concepts), dtype=int)
        for j, c in enumerate(self.class_concept):
            out[:, c] += self.presence[:, j]
        return out

    def binary_matrix(self) -> np.ndarray:
        """Presence/absence matrix (int8) with concept-level missing data.

        Entry is 1 if the language has the class, 0 if it has some other class
        for the same concept, and ``MISSING`` (-1) if the language has no class
        at all for that concept (a lexical gap — absence is then uninformative).
        """
        m = self.presence.astype(np.int8)
        empty = self.concept_counts() == 0  # (n_leaves, n_concepts)
        m[empty[:, self.class_concept]] = MISSING
        return m

    def truth(self) -> dict:
        """Ground-truth record (JSON-serialisable)."""
        return {
            "seed": self.seed,
            "params": asdict(self.params),
            "newick": self.tree.to_newick(precision=10),
            "node_ages": {n.name: n.age for n in self.tree.preorder()},
            "branch_rates": self.branch_rates,
            "concept_rates": self.concept_rates.tolist(),
            "root_age": self.tree.root_age,
            **{k: v for k, v in self.extra.items()},
        }


# ---------------------------------------------------------------------- rates


def discrete_gamma_rates(alpha: float, k: int) -> np.ndarray:
    """Mean rate of each of ``k`` equal-probability gamma categories (Yang 1994).

    The gamma has shape ``alpha`` and mean 1. Returned rates average to 1.
    """
    if k == 1:
        return np.ones(1)
    cuts = stats.gamma.ppf(np.arange(1, k) / k, a=alpha, scale=1.0 / alpha)
    # P(X < x) for Gamma(alpha+1, scale 1/alpha) is gammainc(alpha+1, x*alpha)
    cdf_hi = np.append(special.gammainc(alpha + 1, cuts * alpha), 1.0)
    cdf_lo = np.insert(special.gammainc(alpha + 1, cuts * alpha), 0, 0.0)
    rates = k * (cdf_hi - cdf_lo)
    return rates / rates.mean()  # remove tiny numerical drift


def draw_concept_rates(params: SDolloParams, rng: np.random.Generator) -> np.ndarray:
    n = params.n_concepts
    if params.gamma_shape is None:
        return np.ones(n)
    a = params.gamma_shape
    if params.gamma_categories == 0:
        return rng.gamma(shape=a, scale=1.0 / a, size=n)
    cats = discrete_gamma_rates(a, params.gamma_categories)
    return cats[rng.integers(params.gamma_categories, size=n)]


def draw_branch_rates(tree: Tree, clock_sd: float, rng: np.random.Generator) -> dict[str, float]:
    """Uncorrelated lognormal relaxed clock, mean 1 on the natural scale."""
    out: dict[str, float] = {}
    for node in tree.branches():
        if clock_sd == 0:
            out[node.name] = 1.0  # type: ignore[index]
        else:
            out[node.name] = float(  # type: ignore[index]
                rng.lognormal(mean=-0.5 * clock_sd**2, sigma=clock_sd))
    return out


# ---------------------------------------------------------------------- simulator


def simulate(tree: Tree, params: SDolloParams, seed: Optional[int] = None,
             rng: Optional[np.random.Generator] = None) -> SDolloResult:
    """Simulate cognate classes down ``tree`` under SDollo.

    Either ``seed`` (recorded in the result) or a ready-made ``rng`` may be
    given. Passing ``seed`` is preferred so the run is reproducible from the
    ground-truth record alone.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    lam, mu = params.birth_rate, params.loss_rate
    C = params.n_concepts

    concept_rates = draw_concept_rates(params, rng)
    branch_rates = draw_branch_rates(tree, params.clock_sd, rng)

    # Class registry (grown in chunks of numpy arrays)
    concept_chunks: list[np.ndarray] = []
    node_chunks: list[np.ndarray] = []  # index into node_names
    age_chunks: list[np.ndarray] = []
    node_names = [n.name for n in tree.preorder()]
    node_index = {nm: i for i, nm in enumerate(node_names)}
    n_classes = 0

    def register(concepts: np.ndarray, node_name: str, ages: np.ndarray) -> np.ndarray:
        nonlocal n_classes
        ids = np.arange(n_classes, n_classes + len(concepts))
        concept_chunks.append(concepts.astype(np.int64))
        node_chunks.append(np.full(len(concepts), node_index[node_name], dtype=np.int64))
        age_chunks.append(ages.astype(float))
        n_classes += len(concepts)
        return ids

    # Root: stationary Poisson(lambda/mu) classes per concept
    root = tree.root
    n_root = rng.poisson(lam / mu, size=C)
    root_concepts = np.repeat(np.arange(C), n_root)
    root_ids = register(root_concepts, root.name, np.full(len(root_concepts), root.age))  # type: ignore[arg-type]
    # node name -> (class ids present at the node, their concept indices)
    state: dict[str, tuple[np.ndarray, np.ndarray]] = {
        root.name: (root_ids, root_concepts)  # type: ignore[dict-item]
    }

    # Pre-order so parents are processed before children
    for node in tree.preorder():
        if node.is_root:
            continue
        parent_ids, parent_concepts = state[node.parent.name]  # type: ignore[union-attr]
        t = node.branch_length
        tau = t * branch_rates[node.name] * concept_rates  # (C,) effective lengths

        # Survival of inherited classes
        p_keep = np.exp(-mu * tau[parent_concepts])
        kept = rng.random(len(parent_ids)) < p_keep

        # Births on this branch
        n_birth = rng.poisson(lam * tau)
        b_concepts = np.repeat(np.arange(C), n_birth)
        frac = rng.random(len(b_concepts))           # birth position as fraction of branch
        s = frac * tau[b_concepts]
        alive = rng.random(len(b_concepts)) < np.exp(-mu * (tau[b_concepts] - s))
        b_concepts, frac = b_concepts[alive], frac[alive]
        birth_ages = node.parent.age - frac * t  # type: ignore[union-attr]
        new_ids = register(b_concepts, node.name, birth_ages)  # type: ignore[arg-type]

        state[node.name] = (  # type: ignore[index]
            np.concatenate([parent_ids[kept], new_ids]),
            np.concatenate([parent_concepts[kept], b_concepts]),
        )
        # free memory of parents whose children are all done
        par = node.parent
        if par is not None and node is par.children[-1]:
            del state[par.name]  # type: ignore[arg-type]

    class_concept = np.concatenate(concept_chunks)
    class_node = np.concatenate(node_chunks)
    class_age = np.concatenate(age_chunks)

    leaves = tree.leaves()
    presence = np.zeros((len(leaves), n_classes), dtype=bool)
    for i, leaf in enumerate(leaves):
        presence[i, state[leaf.name][0]] = True  # type: ignore[index]

    return SDolloResult(
        tree=tree,
        params=params,
        seed=seed,
        concept_rates=concept_rates,
        branch_rates=branch_rates,
        class_concept=class_concept,
        class_birth_node=[node_names[i] for i in class_node],  # type: ignore[misc]
        class_birth_age=class_age,
        leaf_names=[l.name for l in leaves],  # type: ignore[misc]
        presence=presence,
    )
