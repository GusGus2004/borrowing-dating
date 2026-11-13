"""Draw TreeAnnotator MCC trees (median heights) for the Mayan runs.

Usage: plot_mcc.py OUT.png TREE1[:title] [TREE2[:title] ...]

Tips are coloured by the conventional subgroups (Kaufman's classification),
internal nodes carry the 95% HPD of their age (bar) and posterior clade
support (label when < 0.95). Ages are years before AD 2000.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GROUPS = {
    "Huastecan": ["HUASTEC", "CHICOMUCELTEC"],
    "Yucatecan": ["ITZAJ", "LACANDON", "MAYA_YUCATECA", "MOPAN"],
    "Cholan–Tzeltalan": ["CHOL_TUMBALA", "CHONTAL_TABASCO", "CHORTI", "TZELTAL_BACHAJON", "TZOTZIL_SAN_ANDRES"],
    "Q'anjob'alan–Chujean": ["ACATECO_SAN_MIGUEL_ACATAN", "QANJOBAL_SANTA_EULALIA", "JACALTECO_WESTERN", "MOCHO",
                             "CHUJ_SAN_MATEO_IXTATAN", "TOJOLABAL"],
    "Mamean": ["MAM_CONSEPCION_CHIQUIRICHAPA", "TECO_TECTITAN", "AGUACATEC", "IXIL_CHAJUL"],
    "K'iche'an": ["CENTRAL_QUICHE", "EASTERN_QUICHE_RABINAL", "NORTHERN_CAKCHIQUEL_TECPAN", "TZUTUJIL_SANTIAGO_ATITLAN",
                  "SACAPULTECO_SACAPULAS_CENTRO", "SIPAKAPENSE", "USPANTEKO", "POCOMAM_EASTERN", "POQOMCHI_WESTERN",
                  "EASTERN_KEKCHI_CAHABON"],
}
COLOURS = dict(zip(GROUPS, ["#8c564b", "#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]))
GROUP_OF = {l: g for g, ls in GROUPS.items() for l in ls}
PRETTY = {"ACATECO_SAN_MIGUEL_ACATAN": "Akatek", "AGUACATEC": "Awakatek", "CENTRAL_QUICHE": "K'iche' (Central)",
          "CHICOMUCELTEC": "Chicomuceltec", "CHOL_TUMBALA": "Ch'ol", "CHONTAL_TABASCO": "Chontal",
          "CHORTI": "Ch'orti'", "CHUJ_SAN_MATEO_IXTATAN": "Chuj", "EASTERN_KEKCHI_CAHABON": "Q'eqchi'",
          "EASTERN_QUICHE_RABINAL": "Achi (Rabinal)", "HUASTEC": "Huastec", "ITZAJ": "Itza'", "IXIL_CHAJUL": "Ixil",
          "JACALTECO_WESTERN": "Jakaltek", "LACANDON": "Lacandon", "MAM_CONSEPCION_CHIQUIRICHAPA": "Mam",
          "MAYA_YUCATECA": "Yucatec", "MOCHO": "Mocho'", "MOPAN": "Mopan", "NORTHERN_CAKCHIQUEL_TECPAN": "Kaqchikel",
          "POCOMAM_EASTERN": "Poqomam", "POQOMCHI_WESTERN": "Poqomchi'", "QANJOBAL_SANTA_EULALIA": "Q'anjob'al",
          "SACAPULTECO_SACAPULAS_CENTRO": "Sakapultek", "SIPAKAPENSE": "Sipakapense", "TECO_TECTITAN": "Tektitek",
          "TOJOLABAL": "Tojolabal", "TZELTAL_BACHAJON": "Tzeltal", "TZOTZIL_SAN_ANDRES": "Tzotzil",
          "TZUTUJIL_SANTIAGO_ATITLAN": "Tz'utujil", "USPANTEKO": "Uspantek"}


class N:
    def __init__(self):
        self.children, self.name, self.length, self.ann = [], None, 0.0, {}


def read_mcc(path: Path) -> N:
    txt = path.read_text()
    tr = dict(re.findall(r"^\s*(\d+)\s+([^,;\s]+)[,]?\s*$", txt.split("Translate")[1].split(";")[0], re.M))
    s = txt[txt.index("tree TREE"):]
    s = s[s.index("=") + 1:].strip()
    pos = 0

    def ann():
        nonlocal pos
        d = {}
        if s.startswith("[&", pos):
            end = s.index("]", pos)
            body = s[pos + 2:end]
            for m in re.finditer(r"([\w%.]+)=(\{[^}]*\}|[^,]*)", body):
                d[m.group(1)] = m.group(2)
            pos = end + 1
        return d

    def node():
        nonlocal pos
        n = N()
        if s[pos] == "(":
            pos += 1
            n.children.append(node())
            while s[pos] == ",":
                pos += 1
                n.children.append(node())
            pos += 1  # ')'
        m = re.match(r"[^\[:,();]*", s[pos:])
        lab = m.group(0); pos += len(lab)
        if lab:
            n.name = tr.get(lab, lab)
        n.ann = ann()
        if s[pos] == ":":
            pos += 1
            m = re.match(r"[0-9.eE+-]+", s[pos:]); n.length = float(m.group(0)); pos += len(m.group(0))
        return n

    root = node()
    _ages(root)
    return root


def _ages(n: N) -> float:
    if not n.children:
        n.age = 0.0
    else:
        n.age = max(_ages(c) + c.length for c in n.children)
        for c in n.children:  # fix tiny rounding so children hang exactly
            pass
    return n.age


def leaves(n):
    return [n] if not n.children else [l for c in n.children for l in leaves(c)]


def draw(ax, root: N, title: str):
    order = leaves(root)
    y = {id(l): i for i, l in enumerate(order)}

    def place(n):
        if n.children:
            ys = [place(c) for c in n.children]
            y[id(n)] = sum(ys) / len(ys)
        return y[id(n)]

    place(root)

    def rec(n):
        for c in n.children:
            ax.plot([n.age, n.age], [y[id(c)], y[id(n)]], color="#333", lw=1)
            ax.plot([c.age, n.age], [y[id(c)], y[id(c)]], color="#333", lw=1)
            rec(c)
        if n.children:
            hpd = n.ann.get("height_95%_HPD")
            if hpd:
                lo, hi = (float(v) for v in hpd.strip("{}").split(","))
                ax.plot([lo, hi], [y[id(n)], y[id(n)]], color="#4a7bb7", lw=5, alpha=0.3, solid_capstyle="butt")
            p = float(n.ann.get("posterior", 1))
            if p < 0.95:
                ax.text(n.age + 40, y[id(n)] + 0.25, f"{p:.2f}", fontsize=6.5, color="#555", ha="left")
        else:
            g = GROUP_OF.get(n.name)
            ax.text(-60, y[id(n)], PRETTY.get(n.name, n.name), va="center", fontsize=8, color=COLOURS.get(g, "k"))

    rec(root)
    ax.invert_xaxis()
    ax.set_xlim(max(5000, root.age * 1.15), -1500)
    ax.set_ylim(-1, len(order))
    ax.set_yticks([])
    for sp in ("left", "right", "top"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("years before AD 2000 (blue bars: 95% HPD of node age)")
    ax.set_title(title, fontsize=10, pad=28)
    ax.set_xticks([t for t in range(0, int(ax.get_xlim()[0]) + 1, 1000)])
    for t, lab in ((1708, "Tikal St. 29\n(AD 292)"), (4200, "Kaufman\nc. 2200 BC")):
        ax.axvline(t, color="#bbb", ls=":", lw=0.8, zorder=0)
        ax.text(t, len(order) - 0.6, lab, fontsize=6.5, color="#888", ha="center", va="bottom")


def main():
    out = Path(sys.argv[1])
    specs = [a.split(":", 1) if ":" in a else (a, Path(a).stem) for a in sys.argv[2:]]
    fig, axes = plt.subplots(1, len(specs), figsize=(7.2 * len(specs), 8.5))
    axes = [axes] if len(specs) == 1 else axes
    for ax, (p, title) in zip(axes, specs):
        draw(ax, read_mcc(Path(p)), title)
    handles = [plt.Line2D([], [], color=c, lw=6, label=g) for g, c in COLOURS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.suptitle("Mayan MCC trees (ASJP 40-item lists, draft calibrations, no borrowing correction)", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out, dpi=150)
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
