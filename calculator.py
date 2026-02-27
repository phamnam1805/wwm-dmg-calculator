#!/usr/bin/env python3
"""
Where Winds Meet (WWM) - Damage Calculator
==========================================
Usage:
  ./wwm_dmg_calc.py                    # base stats, active skill formula
  ./wwm_dmg_calc.py change.cnf         # compare base vs override

Config files:
  base.cnf              character stats
  skill_formulas.cnf    skill damage formulas
  marginal.cnf          equal-cost upgrade options
"""

import sys
import configparser
import random
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Dict, Optional

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
ATTRIBUTES           = ("bellstrike", "stonesplit", "bamboocut", "silkbind")
# BASE_CONFIG          = "pinda-stats.cnf"
BASE_CONFIG          = "base.cnf"
SKILL_FORMULAS_FILE  = "skill_formulas.cnf"
MARGINAL_CONFIG      = "marginal.cnf"
SKILL = "vagrant sword 3rd hit"

AFFINITY_CAP         = 0.40
DIRECT_AFFINITY_CAP  = 0.10
CRIT_CAP             = 0.80
DIRECT_CRIT_CAP      = 0.20
PRECISION_CAP        = 1.00

# ─────────────────────────────────────────────
# Descriptions
# ─────────────────────────────────────────────
DESC_DAMAGE = """
Damage types:
  ORANGE  = affinity hit  -> dmg = total_max * affinity_mult  (highest, fixed)
  YELLOW  = crit hit      -> dmg = roll(min,max) * crit_mult
  WHITE   = normal hit    -> dmg = roll(min,max)
  GRAY    = abrasion      -> dmg = total_min                  (lowest, fixed)

Rates:
  affinity_rate   cap 40%  + direct_affinity (cap 10%) => eff_affinity
  precision_rate  cap 100% (miss = GRAY)
  critical_rate   cap 80%  + direct_critical (cap 20%) => eff_crit
  direct_*        bypass the normal cap but have their own cap

Attribute DMG:
  total = physical * phys_coeff + phys_bonus + (main_attr * 1.5 + others) * attr_coeff + attr_bonus
  affinity dmg uses total_max, abrasion uses total_min

Assumptions & limitations:
  - Skill formula is read from skill_formulas.cnf
    phys_bonus/attr_bonus: flat added before coeff (e.g. skill-specific flat dmg)
    Examples: mystic skills have attr_coeff=0 (physical only)
  - Physical and attribute penetration are NOT modeled
    Results are pre-penetration dmg; scale accordingly
"""

DESC_STABILITY = """
Stability stats (from simulation):
  mean   average dmg over n rolls (should converge to theory E[DMG])
  std    standard deviation - how much each hit deviates from mean
           high std = swingy/inconsistent damage
  min    lowest dmg rolled (usually GRAY = total_min)
  max    highest dmg rolled (usually ORANGE = affinity dmg)
  p10    10% of hits fall below this value
  p90    90% of hits fall below this value
           -> p10~p90 spread shows the practical damage range
"""


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────
@dataclass
class AtkStat:
    atk_min: float
    atk_max: float

    def __post_init__(self):
        if self.atk_max < self.atk_min:
            self.atk_max = self.atk_min

    def e_roll(self) -> float:
        return (self.atk_min + self.atk_max) / 2

    def roll(self, rng: random.SystemRandom) -> float:
        return rng.uniform(self.atk_min, self.atk_max)


@dataclass
class SkillFormula:
    name:        str   = "default"
    phys_coeff:  float = 1.0
    attr_coeff:  float = 1.0
    phys_bonus:  float = 0.0   # added to physical before coeff
    attr_bonus:  float = 0.0   # added to attribute before coeff


@dataclass
class CharStats:
    """
    Character stats. Call .expected_dmg(formula) or .simulate(formula, n)
    to get damage results. All rate caps enforced internally.
    """
    physical:             AtkStat
    attributes:           Dict[str, AtkStat]
    main_attribute:       str
    affinity_rate:        float
    precision_rate:       float
    critical_rate:        float
    affinity_mult:        float = 1.35
    critical_mult:        float = 1.50
    direct_affinity_rate: float = 0.0
    direct_critical_rate: float = 0.0

    # ── effective rates ───────────────────────
    def eff_affinity(self) -> float:
        return min(self.affinity_rate, AFFINITY_CAP) + min(self.direct_affinity_rate, DIRECT_AFFINITY_CAP)

    def eff_precision(self) -> float:
        return min(self.precision_rate, PRECISION_CAP)

    def eff_crit(self) -> float:
        return min(self.critical_rate, CRIT_CAP) + min(self.direct_critical_rate, DIRECT_CRIT_CAP)

    # ── damage components (formula-aware) ─────
    def _phys_max(self, f: SkillFormula) -> float:
        return self.physical.atk_max * f.phys_coeff + f.phys_bonus

    def _phys_min(self, f: SkillFormula) -> float:
        return self.physical.atk_min * f.phys_coeff + f.phys_bonus

    def _phys_e_roll(self, f: SkillFormula) -> float:
        return self.physical.e_roll() * f.phys_coeff + f.phys_bonus

    def _phys_roll(self, rng: random.SystemRandom, f: SkillFormula) -> float:
        return self.physical.roll(rng) * f.phys_coeff + f.phys_bonus

    def _attr_max(self, f: SkillFormula) -> float:
        total = sum(
            stat.atk_max * (1.5 if name == self.main_attribute else 1.0)
            for name, stat in self.attributes.items()
        )
        return total * f.attr_coeff + f.attr_bonus

    def _attr_min(self, f: SkillFormula) -> float:
        total = sum(
            stat.atk_min * (1.5 if name == self.main_attribute else 1.0)
            for name, stat in self.attributes.items()
        )
        return total * f.attr_coeff + f.attr_bonus

    def _attr_e_roll(self, f: SkillFormula) -> float:
        total = sum(
            stat.e_roll() * (1.5 if name == self.main_attribute else 1.0)
            for name, stat in self.attributes.items()
        )
        return total * f.attr_coeff + f.attr_bonus

    def _attr_roll(self, rng: random.SystemRandom, f: SkillFormula) -> float:
        total = sum(
            stat.roll(rng) * (1.5 if name == self.main_attribute else 1.0)
            for name, stat in self.attributes.items()
        )
        return total * f.attr_coeff + f.attr_bonus

    def total_max(self, f: SkillFormula) -> float:
        return self._phys_max(f) + self._attr_max(f)

    def total_min(self, f: SkillFormula) -> float:
        return self._phys_min(f) + self._attr_min(f)

    def e_total_roll(self, f: SkillFormula) -> float:
        return self._phys_e_roll(f) + self._attr_e_roll(f)

    def total_roll(self, rng: random.SystemRandom, f: SkillFormula) -> float:
        return self._phys_roll(rng, f) + self._attr_roll(rng, f)

    # ── main interface ────────────────────────
    def expected_dmg(self, f: SkillFormula) -> dict:
        """Analytical expected damage + theory std (via law of total variance)."""
        a, p, c  = self.eff_affinity(), self.eff_precision(), self.eff_crit()
        am, cm   = self.affinity_mult, self.critical_mult

        dmg_orange = self.total_max(f) * am
        dmg_yellow = self.e_total_roll(f) * cm
        dmg_white  = self.e_total_roll(f)
        dmg_gray   = self.total_min(f)

        prob_orange = a
        prob_yellow = (1 - a) * p * c
        prob_white  = (1 - a) * p * (1 - c)
        prob_gray   = (1 - a) * (1 - p)

        assert abs(prob_orange + prob_yellow + prob_white + prob_gray - 1.0) < 1e-9

        e_dmg = (prob_orange * dmg_orange + prob_yellow * dmg_yellow
               + prob_white  * dmg_white  + prob_gray   * dmg_gray)

        # Var[X] = E[X²] - E[X]²  ;  Var[uniform(lo,hi)] = (hi-lo)²/12
        def var_uni(lo, hi): return (hi - lo) ** 2 / 12

        var_roll = var_uni(self._phys_min(f), self._phys_max(f))
        for name, stat in self.attributes.items():
            m = (1.5 if name == self.main_attribute else 1.0) * f.attr_coeff
            var_roll += var_uni(stat.atk_min * m, stat.atk_max * m)

        e2_roll = var_roll + self.e_total_roll(f) ** 2
        e_x2    = (prob_orange * dmg_orange ** 2
                 + prob_yellow * e2_roll * cm ** 2
                 + prob_white  * e2_roll
                 + prob_gray   * dmg_gray ** 2)
        std = (e_x2 - e_dmg ** 2) ** 0.5

        return {
            "E[DMG]":     round(e_dmg, 2),
            "std":        round(std, 2),
            "P_orange":   round(prob_orange * 100, 2),
            "P_yellow":   round(prob_yellow * 100, 2),
            "P_white":    round(prob_white  * 100, 2),
            "P_gray":     round(prob_gray   * 100, 2),
            "DMG_orange": round(dmg_orange, 2),
            "DMG_yellow": round(dmg_yellow, 2),
            "DMG_white":  round(dmg_white,  2),
            "DMG_gray":   round(dmg_gray,   2),
        }

    def simulate(self, f: SkillFormula, n_rolls: int = 70) -> dict:
        """Monte Carlo simulation using /dev/urandom (SystemRandom)."""
        rng = random.SystemRandom()
        a, p, c = self.eff_affinity(), self.eff_precision(), self.eff_crit()
        counts  = {"orange": 0, "yellow": 0, "white": 0, "gray": 0}
        dmg_log = []

        for _ in range(n_rolls):
            if rng.random() < a:
                dmg = self.total_max(f) * self.affinity_mult
                counts["orange"] += 1
            elif rng.random() >= p:
                dmg = self.total_min(f)
                counts["gray"] += 1
            else:
                roll = self.total_roll(rng, f)
                if rng.random() < c:
                    dmg = roll * self.critical_mult
                    counts["yellow"] += 1
                else:
                    dmg = roll
                    counts["white"] += 1
            dmg_log.append(dmg)

        dmg_log.sort()
        n    = len(dmg_log)
        mean = sum(dmg_log) / n
        std  = (sum((x - mean) ** 2 for x in dmg_log) / n) ** 0.5

        def pct(q): return dmg_log[int(q / 100 * n)]

        return {
            "E[DMG]_sim": round(mean, 2),
            "std":        round(std, 2),
            "min":        round(dmg_log[0], 2),
            "max":        round(dmg_log[-1], 2),
            "p10":        round(pct(10), 2),
            "p90":        round(pct(90), 2),
            "P_orange":   round(counts["orange"] / n * 100, 2),
            "P_yellow":   round(counts["yellow"] / n * 100, 2),
            "P_white":    round(counts["white"]  / n * 100, 2),
            "P_gray":     round(counts["gray"]   / n * 100, 2),
        }


# ─────────────────────────────────────────────
# Config loaders
# ─────────────────────────────────────────────
def _cfg(path: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read(path)
    if not cfg.sections():
        raise FileNotFoundError(f"Config not found or empty: {path}")
    return cfg


def load_stats(path: str) -> CharStats:
    cfg       = _cfg(path)
    main_attr = cfg["character"]["main_attribute"].strip()
    physical  = AtkStat(float(cfg["physical"]["min"]), float(cfg["physical"]["max"]))
    attributes = {}
    for attr in ATTRIBUTES:
        sec = f"attribute.{attr}"
        if sec in cfg:
            attributes[attr] = AtkStat(float(cfg[sec]["min"]), float(cfg[sec]["max"]))
    r = cfg["rates"]
    return CharStats(
        physical             = physical,
        attributes           = attributes,
        main_attribute       = main_attr,
        affinity_rate        = float(r["affinity_rate"]),
        precision_rate       = float(r["precision_rate"]),
        critical_rate        = float(r["critical_rate"]),
        affinity_mult        = float(r.get("affinity_mult",        "1.402")),
        critical_mult        = float(r.get("critical_mult",        "1.5")),
        direct_affinity_rate = float(r.get("direct_affinity_rate", "0.0")),
        direct_critical_rate = float(r.get("direct_critical_rate", "0.0")),
    )


def apply_override(base: CharStats, path: str) -> CharStats:
    """Return new CharStats with only the changed fields from override applied."""
    cfg = _cfg(path)
    s   = deepcopy(base)
    if "character" in cfg:
        s.main_attribute = cfg["character"]["main_attribute"].strip()
    if "physical" in cfg:
        p = cfg["physical"]
        s.physical = AtkStat(
            float(p.get("min", base.physical.atk_min)),
            float(p.get("max", base.physical.atk_max)),
        )
    for attr in ATTRIBUTES:
        sec = f"attribute.{attr}"
        if sec in cfg:
            b = base.attributes.get(attr, AtkStat(0, 0))
            s.attributes[attr] = AtkStat(
                float(cfg[sec].get("min", b.atk_min)),
                float(cfg[sec].get("max", b.atk_max)),
            )
    if "rates" in cfg:
        r = cfg["rates"]
        for key in ("affinity_rate", "precision_rate", "critical_rate",
                    "affinity_mult", "critical_mult",
                    "direct_affinity_rate", "direct_critical_rate"):
            if key in r:
                setattr(s, key, float(r[key]))
    return s


def load_skill_formulas(path: str) -> Dict[str, SkillFormula]:
    cfg = _cfg(path)
    return {
        name: SkillFormula(
            name           = name,
            phys_coeff  = float(cfg[name].get("phys_coeff",  cfg[name].get("physical_coeff", "1.0"))),
            attr_coeff  = float(cfg[name].get("attr_coeff",  "1.0")),
            phys_bonus  = float(cfg[name].get("phys_bonus",  "0")),
            attr_bonus  = float(cfg[name].get("attr_bonus",  "0")),
        )
        for name in cfg.sections()
    }


def load_marginal_opts(path: str, base: CharStats) -> Dict[str, CharStats]:
    """
    Each section = one equal-cost option. Values are deltas (+/-).
    Supported keys: affinity_rate, precision_rate, critical_rate,
                    direct_affinity_rate, direct_critical_rate,
                    physical_min, physical_max, <attr>_min, <attr>_max
    """
    cfg = _cfg(path)

    def apply_deltas(s: CharStats, items):
        for key, val in items:
            delta = float(val)
            if key in ("affinity_rate", "precision_rate", "critical_rate",
                       "direct_affinity_rate", "direct_critical_rate",
                       "affinity_mult", "critical_mult"):
                setattr(s, key, getattr(s, key) + delta)
            elif key == "physical_min":
                s.physical = AtkStat(s.physical.atk_min + delta, s.physical.atk_max)
            elif key == "physical_max":
                s.physical = AtkStat(s.physical.atk_min, s.physical.atk_max + delta)
            else:
                for attr in ATTRIBUTES:
                    if key == f"{attr}_min":
                        o = s.attributes[attr]
                        s.attributes[attr] = AtkStat(o.atk_min + delta, o.atk_max)
                        break
                    elif key == f"{attr}_max":
                        o = s.attributes[attr]
                        s.attributes[attr] = AtkStat(o.atk_min, o.atk_max + delta)
                        break

    opts = {}
    for name in cfg.sections():
        s = deepcopy(base)
        apply_deltas(s, cfg[name].items())
        opts[name] = s
    return opts


# ─────────────────────────────────────────────
# Printing
# ─────────────────────────────────────────────
SEP = "=" * 58

def print_stats(label: str, s: CharStats, f: SkillFormula,
                compare_to: Optional[CharStats] = None,
                simulate: bool = False, n_rolls: int = 70):
    print(f"\n{SEP}")
    print(f"  {label}")
    print(SEP)
    print(f"  Physical   : {s.physical.atk_min} - {s.physical.atk_max}")
    for name, stat in s.attributes.items():
        tag = " [MAIN x1.5]" if name == s.main_attribute else " [x1.0]"
        print(f"  {name:<12}: {stat.atk_min} - {stat.atk_max}{tag}")
    print(f"  Affinity   : {s.affinity_rate*100:.1f}% + {s.direct_affinity_rate*100:.1f}% direct"
          f"  => eff {s.eff_affinity()*100:.2f}%  (mult x{s.affinity_mult})")
    print(f"  Precision  : {s.precision_rate*100:.1f}%  => eff {s.eff_precision()*100:.2f}%")
    print(f"  Crit rate  : {s.critical_rate*100:.1f}% + {s.direct_critical_rate*100:.1f}% direct"
          f"  => eff {s.eff_crit()*100:.2f}%  (mult x{s.critical_mult})")
    print(f"  Skill      : {f.name}"
          f"  (phys×{f.phys_coeff} +{f.phys_bonus}bonus  attr×{f.attr_coeff} +{f.attr_bonus}bonus)")
    print(f"  Total max  : {s.total_max(f):.1f}"
          f"  |  E[roll]: {s.e_total_roll(f):.1f}"
          f"  |  Min: {s.total_min(f):.1f}")

    r   = s.expected_dmg(f)
    sim = s.simulate(f, n_rolls) if simulate else None

    diff_str = ""
    if compare_to:
        diff     = r["E[DMG]"] - compare_to.expected_dmg(f)["E[DMG]"]
        diff_str = f"  ({diff:+.2f} vs base)"

    print(f"\n  E[DMG]  theory = {r['E[DMG]']}  std={r['std']}{diff_str}")
    if sim:
        print(f"  E[DMG]  sim    = {sim['E[DMG]_sim']}  std={sim['std']}  (n={n_rolls})")

    hdr = f"  {'Type':<8}  {'DMG value':>10}  {'Theory%':>8}"
    div = f"  {'-'*30}"
    if sim:
        hdr += f"  {'Simulated%':>10}"
        div += "-" * 13
    print(hdr); print(div)
    for key, dmg_key in [("orange","DMG_orange"),("yellow","DMG_yellow"),
                         ("white","DMG_white"),("gray","DMG_gray")]:
        row = f"  {key:<8}  {r[dmg_key]:>10.2f}  {r[f'P_{key}']:>7.2f}%"
        if sim:
            row += f"  {sim[f'P_{key}']:>9.2f}%"
        print(row)

    if sim:
        print(f"\n  Stability:")
        print(f"  {'':15}  {'theory':>10}  {'sim':>10}")
        print(f"  {'-'*38}")
        for lbl, tk, sk in [
            ("mean/E[DMG]", "E[DMG]",     "E[DMG]_sim"),
            ("std",         "std",         "std"),
            ("min",         None,          "min"),
            ("p10",         None,          "p10"),
            ("p90",         None,          "p90"),
            ("max",         None,          "max"),
        ]:
            tv = r[tk] if tk else "—"
            sv = sim[sk]
            print(f"  {lbl:<15}  {str(tv):>10}  {sv:>10}")


def print_marginal(base: CharStats, f: SkillFormula, opts: Dict[str, CharStats], verbose: bool = False):
    print(f"\n{SEP}")
    print("  Marginal gain (equal-cost options)")
    print(SEP)
    base_e = base.expected_dmg(f)["E[DMG]"]

    for name, stat in opts.items():
        r    = stat.expected_dmg(f)
        gain = r["E[DMG]"] - base_e
        print(f"  {name:<26}  +{gain:.2f} DMG  (E={r['E[DMG]']}  std={r['std']})")

    if not verbose:
        return

    print(f"\n{SEP}")
    print("  Double marginal gain (all pairs)  [-v]")
    print(SEP)
    pair_results = []

    for a, b in combinations_with_replacement(list(opts.keys()), 2):
        sa, sb = opts[a], opts[b]
        sc     = deepcopy(base)
        for key in ("affinity_rate", "precision_rate", "critical_rate",
                    "direct_affinity_rate", "direct_critical_rate",
                    "affinity_mult", "critical_mult"):
            da = getattr(sa, key) - getattr(base, key)
            db = getattr(sb, key) - getattr(base, key)
            setattr(sc, key, getattr(base, key) + da + db)
        sc.physical = AtkStat(
            base.physical.atk_min + (sa.physical.atk_min - base.physical.atk_min)
                                  + (sb.physical.atk_min - base.physical.atk_min),
            base.physical.atk_max + (sa.physical.atk_max - base.physical.atk_max)
                                  + (sb.physical.atk_max - base.physical.atk_max),
        )
        for attr in base.attributes:
            sc.attributes[attr] = AtkStat(
                base.attributes[attr].atk_min
                    + (sa.attributes[attr].atk_min - base.attributes[attr].atk_min)
                    + (sb.attributes[attr].atk_min - base.attributes[attr].atk_min),
                base.attributes[attr].atk_max
                    + (sa.attributes[attr].atk_max - base.attributes[attr].atk_max)
                    + (sb.attributes[attr].atk_max - base.attributes[attr].atk_max),
            )
        rc   = sc.expected_dmg(f)
        gain = rc["E[DMG]"] - base_e
        lbl  = f"{a} + {b}"
        print(f"  {lbl:<52}  +{gain:.2f} DMG  (E={rc['E[DMG]']}  std={rc['std']})")
        pair_results.append((lbl, gain, rc["E[DMG]"], rc["std"]))

    print(f"\n  Top 5:")
    print(f"  {'-'*66}")
    for rank, (lbl, g, e, std) in enumerate(
            sorted(pair_results, key=lambda x: -x[1])[:5], 1):
        print(f"  #{rank}  {lbl:<52}  +{g:.2f}  (E={e}  std={std})")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
print(DESC_DAMAGE)
print(DESC_STABILITY)

try:
    base = load_stats(BASE_CONFIG)
except FileNotFoundError as e:
    print(f"Error: {e}"); sys.exit(1)

try:
    skill_formulas = load_skill_formulas(SKILL_FORMULAS_FILE)
    active_formula = skill_formulas.get(SKILL, SkillFormula())
    print(f"Skill formulas : {', '.join(skill_formulas)}  (active: {active_formula.name})")
except FileNotFoundError:
    active_formula = SkillFormula()
    print("Skill formulas : none found, using default (phys×1.0  attr×1.0  base+0)")

args    = sys.argv[1:]
verbose = "-v" in args
args    = [a for a in args if a != "-v"]

if args:
    try:
        changed = apply_override(base, args[0])
    except FileNotFoundError as e:
        print(f"Error: {e}"); sys.exit(1)
    print(f"\nBase   : {BASE_CONFIG}")
    print(f"Change : {args[0]}")
    print_stats("BASE",    base,    active_formula, simulate=True)
    print_stats("CHANGED", changed, active_formula, compare_to=base, simulate=True)
else:
    print(f"Base   : {BASE_CONFIG}")
    print_stats("Current stats", base, active_formula, simulate=True)
    try:
        opts = load_marginal_opts(MARGINAL_CONFIG, base)
        print_marginal(base, active_formula, opts, verbose=verbose)
    except FileNotFoundError:
        print(f"\n(no {MARGINAL_CONFIG} found, skipping marginal gain)")