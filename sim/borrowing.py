"""Stochastic Dollo with lexical borrowing between contemporaneous lineages.

The vertical part is the SDollo model of :mod:`sim.sdollo` (births at rate
``lambda * r_c * rho_b`` per concept, losses at ``mu * r_c * rho_b`` per class,
root at stationarity). On top of it:

* Each lineage, for each concept, undergoes **borrowing events** at rate
  ``beta * w_c`` per unit (calendar) time, with ``w_c = r_c`` if
  ``scale_by_concept_rate`` else 1. Borrowing is contact, so it runs on
  calendar time and is not scaled by the branch clock rate.
* At an event the donor is another lineage alive at that moment that has at
  least one class for the concept, chosen with weight ``1`` (``contact="uniform"``)
  or ``exp(-d / distance_scale)`` (``contact="distance"``), where ``d`` is the
  patristic distance between the two lineages at that time,
  ``2 * (age(MRCA) - t)``. If no lineage has a class for the concept the
  event is void.
* One of the donor's classes for the concept is copied to the recipient. With
  ``mode="replace"`` (default; the usual linguistic case) the recipient's own
  classes for that concept are lost; with ``mode="add"`` it keeps them.

A borrowed class present on two lineages violates the Dollo assumption
of the inference model (one birth, then only losses down a single tree);
that mismatch is what milestone 2 measures.

Simulation is exact: a Gillespie process over all lineages within each
epoch between consecutive node ages. Every borrowing event is logged, and
each class carried by a lineage is tagged inherited or borrowed, so the
leaves' (language, concept) entries have ground-truth provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .phonology import LineagePhonology, PhonologyParams, random_protoform
from .sdollo import SDolloParams, SDolloResult, draw_branch_rates, draw_concept_rates
from .tree import Node, Tree

CONTACT_MODELS = ("uniform", "distance")
MODES = ("replace", "add")


@dataclass(frozen=True)
class BorrowingParams:
    rate: float = 0.0                  # beta: events per concept per lineage per time unit
    contact: str = "uniform"
    distance_scale: float = 1000.0     # for contact="distance": e-folding patristic distance
    mode: str = "replace"
    scale_by_concept_rate: bool = False

    def __post_init__(self) -> None:
        if self.rate < 0:
            raise ValueError("borrowing rate must be >= 0")
        if self.contact not in CONTACT_MODELS:
            raise ValueError(f"contact must be one of {CONTACT_MODELS}")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.distance_scale <= 0:
            raise ValueError("distance_scale must be positive")


def _ancestors(node: Node) -> list[Node]:
    out = []
    n: Optional[Node] = node
    while n is not None:
        out.append(n)
        n = n.parent
    return out


def _mrca_age(a: Node, b: Node) -> float:
    anc = {id(x) for x in _ancestors(a)}
    for x in _ancestors(b):
        if id(x) in anc:
            return x.age
    raise RuntimeError("no common ancestor")  # pragma: no cover


def simulate_with_borrowing(tree: Tree, params: SDolloParams, borrowing: BorrowingParams,
                            seed: Optional[int] = None,
                            rng: Optional[np.random.Generator] = None,
                            phonology: Optional[PhonologyParams] = None) -> SDolloResult:
    """Simulate SDollo + borrowing down ``tree``.

    The result is an :class:`SDolloResult`; ``extra`` holds

    * ``borrowing``: the parameters,
    * ``borrow_events``: list of dicts (age, recipient branch, donor branch,
      concept, class id, replaced class ids),
    * ``borrow_attempts``: (recipient, donor) for every event where a donor was
      chosen, including redundant ones (recipient already had the class),
    * ``n_void_borrow_events``: events with no donor or a redundant copy,
    * ``borrowed``: (n_leaves, n_classes) bool array, True where a leaf
      carries the class because of a borrowing event on its own ancestry
      (removed by :func:`borrowed_entries` before JSON export),
    * ``sound_changes`` (with ``phonology``): per branch, the regular changes
      (age, rule) that happened on it.

    With ``phonology`` every class carries a phoneme string per lineage and the
    result's ``forms`` holds the leaf forms. Phonology uses its own random
    stream (spawned from ``rng``'s seed sequence), so the cognate data are
    identical with and without it.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    phon_rng = (np.random.default_rng(rng.bit_generator.seed_seq.spawn(1)[0])  # type: ignore[attr-defined]
                if phonology is not None else None)
    forms: dict[str, dict[int, str]] = {}
    sched: dict[str, LineagePhonology] = {}
    sound_changes: dict[str, list] = {}
    lam, mu = params.birth_rate, params.loss_rate
    C = params.n_concepts
    concept_rates = draw_concept_rates(params, rng)
    branch_rates = draw_branch_rates(tree, params.clock_sd, rng)
    w_borrow = concept_rates if borrowing.scale_by_concept_rate else np.ones(C)
    beta = borrowing.rate

    cls_concept: list[int] = []
    cls_node: list[str] = []
    cls_age: list[float] = []

    def new_class(c: int, node_name: str, age: float) -> int:
        cls_concept.append(c)
        cls_node.append(node_name)
        cls_age.append(age)
        return len(cls_concept) - 1

    # lineage state: per concept a list of class ids; plus set of borrowed ids
    root = tree.root
    root_state = [[new_class(c, root.name, root.age) for _ in range(k)]  # type: ignore[arg-type]
                  for c, k in enumerate(rng.poisson(lam / mu, size=C))]
    root_forms = ({cid: random_protoform(phonology, phon_rng) for lst in root_state for cid in lst}
                  if phonology is not None else {})
    State = tuple[list[list[int]], set[int]]
    state: dict[str, State] = {}
    events: list[dict] = []
    attempts: list[tuple[str, str]] = []  # (recipient, chosen donor), incl. redundant copies
    n_void = 0

    def spawn_children(node: Node, st: State, fm: dict[int, str]) -> list[Node]:
        for ch in node.children:
            state[ch.name] = ([list(x) for x in st[0]], set(st[1]))  # type: ignore[index]
            if phonology is not None:
                forms[ch.name] = dict(fm)  # type: ignore[index]
                sched[ch.name] = LineagePhonology(node.age, ch.age, phonology, C, phon_rng)  # type: ignore[index, arg-type]
        return list(node.children)

    def advance_phonology(to_age: float) -> None:
        if phonology is not None:
            for nm in names:
                sched[nm].advance(to_age, forms[nm], phon_rng)  # type: ignore[arg-type]

    active: list[Node] = spawn_children(root, (root_state, set()), root_forms)
    # epoch boundaries: every node age below the root, descending
    t = root.age
    leaves_done: dict[str, State] = {}

    while active:
        t_end = max(n.age for n in active)  # next node reached
        L = len(active)
        rho = np.array([branch_rates[n.name] for n in active])  # type: ignore[index]
        names = [n.name for n in active]
        if beta > 0 and borrowing.contact == "distance":
            mrca = np.array([[_mrca_age(a, b) if i != j else np.nan
                              for j, b in enumerate(active)] for i, a in enumerate(active)])
        counts = np.array([[len(state[nm][0][c]) for c in range(C)] for nm in names],
                          dtype=float)  # (L, C)
        birth_rate_l = lam * rho * concept_rates.sum()  # (L,)
        borrow_rate_total = beta * w_borrow.sum() * (L if L > 1 else 0)

        while True:
            loss_lc = mu * rho[:, None] * concept_rates[None, :] * counts
            total = birth_rate_l.sum() + loss_lc.sum() + borrow_rate_total
            dt = rng.exponential(1.0 / total) if total > 0 else np.inf
            if t - dt <= t_end:
                t = t_end
                advance_phonology(t_end)
                break
            t -= dt
            advance_phonology(t)
            u = rng.random() * total
            if u < birth_rate_l.sum():
                l = int(np.searchsorted(np.cumsum(birth_rate_l), u, side="right"))
                c = int(rng.choice(C, p=concept_rates / concept_rates.sum()))
                cid = new_class(c, names[l], t)  # type: ignore[arg-type]
                state[names[l]][0][c].append(cid)
                counts[l, c] += 1
                if phonology is not None:
                    forms[names[l]][cid] = random_protoform(phonology, phon_rng)  # type: ignore[index, arg-type]
                continue
            u -= birth_rate_l.sum()
            if u < loss_lc.sum():
                flat = int(np.searchsorted(np.cumsum(loss_lc.ravel()), u, side="right"))
                l, c = divmod(flat, C)
                lst = state[names[l]][0][c]
                cid = lst.pop(int(rng.integers(len(lst))))
                state[names[l]][1].discard(cid)
                counts[l, c] -= 1
                if phonology is not None:
                    del forms[names[l]][cid]  # type: ignore[index]
                continue
            # borrowing: recipient uniform over lineages, concept by w_borrow
            l = int(rng.integers(L))
            c = int(rng.choice(C, p=w_borrow / w_borrow.sum()))
            others = np.array([m for m in range(L) if m != l and counts[m, c] > 0])
            if len(others) == 0:
                n_void += 1
                continue
            if borrowing.contact == "uniform":
                wts = np.ones(len(others))
            else:
                d = 2.0 * (mrca[l, others] - t)
                wts = np.exp(-(d - d.min()) / borrowing.distance_scale)
            m = int(others[rng.choice(len(others), p=wts / wts.sum())])
            attempts.append((names[l], names[m]))  # type: ignore[arg-type]
            donor_list = state[names[m]][0][c]
            cid = donor_list[int(rng.integers(len(donor_list)))]
            rec_list, rec_borrowed = state[names[l]]
            if cid in rec_list[c]:  # recipient already has this cognate: no change
                n_void += 1
                continue
            replaced = list(rec_list[c]) if borrowing.mode == "replace" else []
            if borrowing.mode == "replace":
                for x in replaced:
                    rec_borrowed.discard(x)
                    if phonology is not None:
                        del forms[names[l]][x]  # type: ignore[index]
                rec_list[c] = []
            rec_list[c].append(cid)
            rec_borrowed.add(cid)
            counts[l, c] = len(rec_list[c])
            if phonology is not None:
                forms[names[l]][cid] = forms[names[m]][cid]  # type: ignore[index]
            events.append({"age": float(t), "recipient": names[l], "donor": names[m],
                           "concept": c, "class": cid, "replaced": replaced})

        # node(s) reached at t_end
        nxt = []
        for n in active:
            if n.age >= t_end - 1e-12 * max(1.0, abs(t_end)):
                if phonology is not None:
                    sound_changes[n.name] = sched.pop(n.name).rules  # type: ignore[index, arg-type]
                if n.is_leaf:
                    leaves_done[n.name] = state.pop(n.name)  # type: ignore[index, arg-type]
                else:
                    nxt.extend(spawn_children(n, state.pop(n.name),  # type: ignore[arg-type]
                                              forms.pop(n.name, {})))  # type: ignore[arg-type]
            else:
                nxt.append(n)
        active = nxt

    leaves = tree.leaves()
    n_cls = len(cls_concept)
    presence = np.zeros((len(leaves), n_cls), dtype=bool)
    borrowed = np.zeros((len(leaves), n_cls), dtype=bool)
    for i, leaf in enumerate(leaves):
        lists, bset = leaves_done[leaf.name]  # type: ignore[index]
        for lst in lists:
            presence[i, lst] = True
        if bset:
            borrowed[i, list(bset)] = True
    leaf_forms = None
    if phonology is not None:
        leaf_forms = np.full((len(leaves), n_cls), None, dtype=object)
        for i, leaf in enumerate(leaves):
            for cid, f in forms[leaf.name].items():  # type: ignore[index]
                leaf_forms[i, cid] = f

    return SDolloResult(
        tree=tree, params=params, seed=seed, concept_rates=concept_rates,
        branch_rates=branch_rates, class_concept=np.array(cls_concept, dtype=np.int64),
        class_birth_node=cls_node, class_birth_age=np.array(cls_age, dtype=float),
        leaf_names=[l.name for l in leaves], presence=presence,  # type: ignore[misc]
        extra={"borrowing": borrowing.__dict__.copy(), "borrow_events": events,
               "borrow_attempts": attempts, "n_void_borrow_events": n_void, "borrowed": borrowed,
               **({"sound_changes": sound_changes} if phonology is not None else {})},
        forms=leaf_forms,
    )


def borrowed_fraction(res: SDolloResult) -> float:
    """Fraction of leaf (language, class) entries that were acquired by borrowing."""
    b = res.extra["borrowed"]
    return float(b.sum() / max(res.presence.sum(), 1))


def borrowed_entries(res: SDolloResult) -> tuple[SDolloResult, list[dict]]:
    """Restrict to observed classes and list the borrowed leaf entries.

    Returns the observed result (with ``borrowed`` removed from ``extra`` so it
    serialises) and records ``{"language", "concept", "class_index"}`` where
    ``class_index`` indexes the observed classes (i.e. cognate set order in CLDF).
    """
    keep = res.presence.any(axis=0)
    b = res.extra["borrowed"][:, keep]
    obs = res.observed()
    obs.extra = {k: v for k, v in obs.extra.items() if k not in ("borrowed", "borrow_events", "borrow_attempts")}
    obs.extra["n_borrow_events"] = len(res.extra["borrow_events"])
    obs.extra["borrowed_fraction"] = borrowed_fraction(res)
    rows = [{"language": obs.leaf_names[i], "concept": int(obs.class_concept[j]),
             "class_index": int(j)} for i, j in zip(*np.nonzero(b))]
    return obs, rows
