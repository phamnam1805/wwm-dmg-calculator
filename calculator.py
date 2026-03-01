#!/usr/bin/env python3
"""
Where Winds Meet (WWM) - Damage Calculator
==========================================
Usage:
  ./calculator.py                      # base stats, default skill
  ./calculator.py change.cnf           # compare base vs override
  ./calculator.py -s <id|name>         # select skill by DB id or name search

Config files:
  base.cnf              character stats
  marginal.cnf          equal-cost upgrade options
  skills.db             skill formula database  (manage via skill_db.py)
"""

import sys
import configparser
import random
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Dict, List, Optional

from skill_db import (
    DB_PATH, SkillFormula,
    init_db, get_skill_by_id, search_skills, list_skills,
)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
ATTRIBUTES      = ("bellstrike", "stonesplit", "bamboocut", "silkbind")
BASE_CONFIG     = "base.cnf"
MARGINAL_CONFIG = "marginal.cnf"
DEFAULT_SKILL_ID: Optional[int] = None  # set to an int to pre-select a skill by ID

AFFINITY_CAP         = 0.40
DIRECT_AFFINITY_CAP  = 0.10
CRIT_CAP             = 0.80
DIRECT_CRIT_CAP      = 0.20
PRECISION_CAP        = 1.00


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────
@dataclass
class AtkStat:
    atk_min:   float
    atk_max:   float
    pen:       float = 0.0   # penetration
    dmg_bonus: float = 0.0   # damage bonus (additive %)

    def dmg_mult(self) -> float:
        return (1 + self.pen / 200) * (1 + self.dmg_bonus)

    def __post_init__(self):
        if self.atk_max < self.atk_min:
            self.atk_max = self.atk_min

    def e_roll(self) -> float:
        return (self.atk_min + self.atk_max) / 2

    def roll(self, rng: random.SystemRandom) -> float:
        return rng.uniform(self.atk_min, self.atk_max)


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
        return ( f.phys_coeff * self.physical.atk_max + f.phys_bonus) * self.physical.dmg_mult()

    def _phys_min(self, f: SkillFormula) -> float:
        return (self.physical.atk_min * f.phys_coeff + f.phys_bonus) * self.physical.dmg_mult()

    def _phys_e_roll(self, f: SkillFormula) -> float:
        return (self.physical.e_roll() * f.phys_coeff + f.phys_bonus) * self.physical.dmg_mult()

    def _phys_roll(self, rng: random.SystemRandom, f: SkillFormula) -> float:
        return (self.physical.roll(rng) * f.phys_coeff + f.phys_bonus) * self.physical.dmg_mult()

    def _attr_max(self, f: SkillFormula) -> float:
        total = 0.0
        for name, stat in self.attributes.items():
            main = name == self.main_attribute
            attr_bonus = 0.0 if f.is_mystic else (f.attr_bonus if main else 0.0)
            attr_coeff = f.attr_coeff if main else f.phys_coeff
            total += (attr_coeff * (stat.atk_max + attr_bonus)) * stat.dmg_mult()
        return total

    def _attr_min(self, f: SkillFormula) -> float:
        total = 0.0
        for name, stat in self.attributes.items():
            main = name == self.main_attribute
            attr_bonus = 0.0 if f.is_mystic else (f.attr_bonus if main else 0.0)
            attr_coeff = f.attr_coeff if main else f.phys_coeff
            total += (attr_coeff * (stat.atk_min + attr_bonus)) * stat.dmg_mult()
        return total

    def _attr_e_roll(self, f: SkillFormula) -> float:
        total = 0.0
        for name, stat in self.attributes.items():
            main = name == self.main_attribute
            attr_bonus = 0.0 if f.is_mystic else (f.attr_bonus if main else 0.0)
            attr_coeff = f.attr_coeff if main else f.phys_coeff
            total += (attr_coeff * (stat.e_roll() + attr_bonus)) * stat.dmg_mult()
        return total

    def _attr_roll(self, rng: random.SystemRandom, f: SkillFormula) -> float:
        total = 0.0
        for name, stat in self.attributes.items():
            main = name == self.main_attribute
            attr_bonus = 0.0 if f.is_mystic else (f.attr_bonus if main else 0.0)
            attr_coeff = f.attr_coeff if main else f.phys_coeff
            total += (attr_coeff * (stat.roll(rng) + attr_bonus)) * stat.dmg_mult()
        return total

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
            main = name == self.main_attribute
            attr_coeff = f.attr_coeff if main else f.phys_coeff
            m = attr_coeff * stat.dmg_mult()
            var_roll += var_uni(stat.atk_min * m, stat.atk_max * m)

        e2_roll = var_roll + self.e_total_roll(f) ** 2
        e_x2    = (prob_orange * dmg_orange ** 2
                 + prob_yellow * e2_roll * cm ** 2
                 + prob_white  * e2_roll
                 + prob_gray   * dmg_gray ** 2)
        std = (e_x2 - e_dmg ** 2) ** 0.5

        return {
            "E[DMG]":         round(e_dmg, 2),
            "std":            round(std, 2),
            "P_orange":       round(prob_orange * 100, 2),
            "P_yellow":       round(prob_yellow * 100, 2),
            "P_white":        round(prob_white  * 100, 2),
            "P_gray":         round(prob_gray   * 100, 2),
            "DMG_orange":     round(dmg_orange, 2),
            "DMG_yellow":     round(dmg_yellow, 2),
            "DMG_yellow_min": round(dmg_gray   * cm, 2),
            "DMG_yellow_max": round(dmg_orange / am * cm, 2),
            "DMG_white":      round(dmg_white,  2),
            "DMG_white_min":  round(dmg_gray,   2),
            "DMG_white_max":  round(dmg_orange / am, 2),
            "DMG_gray":       round(dmg_gray,   2),
        }

    def simulate(self, f: SkillFormula, n_rolls: int = 70) -> dict:
        """Monte Carlo simulation using /dev/urandom (SystemRandom)."""
        rng = random.SystemRandom()
        a, p, c = self.eff_affinity(), self.eff_precision(), self.eff_crit()
        counts   = {"orange": 0, "yellow": 0, "white": 0, "gray": 0}
        dmg_sum  = {"orange": 0.0, "yellow": 0.0, "white": 0.0, "gray": 0.0}
        dmg_log  = []

        for _ in range(n_rolls):
            if rng.random() < a:
                dmg   = self.total_max(f) * self.affinity_mult
                color = "orange"
            elif rng.random() >= p:
                dmg   = self.total_min(f)
                color = "gray"
            else:
                roll = self.total_roll(rng, f)
                if rng.random() < c:
                    dmg   = roll * self.critical_mult
                    color = "yellow"
                else:
                    dmg   = roll
                    color = "white"
            counts[color]  += 1
            dmg_sum[color] += dmg
            dmg_log.append(dmg)

        dmg_log.sort()
        n     = len(dmg_log)
        mean  = sum(dmg_log) / n
        std   = (sum((x - mean) ** 2 for x in dmg_log) / n) ** 0.5
        total = sum(dmg_log)

        def pct(q): return dmg_log[int(q / 100 * n)]

        return {
            "E[DMG]_sim": round(mean, 2),
            "std":        round(std, 2),
            "min":        round(dmg_log[0], 2),
            "max":        round(dmg_log[-1], 2),
            "p10":        round(pct(10), 2),
            "p90":        round(pct(90), 2),
            "total":      round(total, 2),
            "P_orange":   round(counts["orange"] / n * 100, 2),
            "P_yellow":   round(counts["yellow"] / n * 100, 2),
            "P_white":    round(counts["white"]  / n * 100, 2),
            "P_gray":     round(counts["gray"]   / n * 100, 2),
            "total_orange": round(dmg_sum["orange"], 2),
            "total_yellow": round(dmg_sum["yellow"], 2),
            "total_white":  round(dmg_sum["white"],  2),
            "total_gray":   round(dmg_sum["gray"],   2),
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
    p = cfg["physical"]
    physical  = AtkStat(
        atk_min   = float(p["min"]),
        atk_max   = float(p["max"]),
        pen       = float(p.get("pen",       "0.0")),
        dmg_bonus = float(p.get("dmg_bonus", "0.0")),
    )
    attributes = {}
    for attr in ATTRIBUTES:
        sec = f"attribute.{attr}"
        if sec in cfg:
            s = cfg[sec]
            attributes[attr] = AtkStat(
                atk_min   = float(s["min"]),
                atk_max   = float(s["max"]),
                pen       = float(s.get("pen",       "0.0")),
                dmg_bonus = float(s.get("dmg_bonus", "0.0")),
            )
    r = cfg["rates"]
    return CharStats(
        physical             = physical,
        attributes           = attributes,
        main_attribute       = main_attr,
        affinity_rate        = float(r["affinity_rate"]),
        precision_rate       = float(r["precision_rate"]),
        critical_rate        = float(r["critical_rate"]),
        affinity_mult        = float(r.get("affinity_mult",        "1.35")),
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
            atk_min   = float(p.get("min",       base.physical.atk_min)),
            atk_max   = float(p.get("max",       base.physical.atk_max)),
            pen       = float(p.get("pen",       base.physical.pen)),
            dmg_bonus = float(p.get("dmg_bonus", base.physical.dmg_bonus)),
        )
    for attr in ATTRIBUTES:
        sec = f"attribute.{attr}"
        if sec in cfg:
            b = base.attributes.get(attr, AtkStat(0, 0))
            s.attributes[attr] = AtkStat(
                atk_min   = float(cfg[sec].get("min",       b.atk_min)),
                atk_max   = float(cfg[sec].get("max",       b.atk_max)),
                pen       = float(cfg[sec].get("pen",       b.pen)),
                dmg_bonus = float(cfg[sec].get("dmg_bonus", b.dmg_bonus)),
            )
    if "rates" in cfg:
        r = cfg["rates"]
        for key in ("affinity_rate", "precision_rate", "critical_rate",
                    "affinity_mult", "critical_mult",
                    "direct_affinity_rate", "direct_critical_rate"):
            if key in r:
                setattr(s, key, float(r[key]))
    return s


def _resolve_skill(arg: str, db_path: str = DB_PATH) -> Optional[SkillFormula]:
    """
    Resolve a -s argument to a SkillFormula.
    - If arg is a pure integer → look up by ID.
    - Otherwise → search by name; if multiple matches, print list and return None.
    """
    if arg.isdigit():
        return get_skill_by_id(int(arg), db_path)
    results = search_skills(arg, db_path)
    if not results:
        print(f"  No skills found matching '{arg}'.")
        return None
    if len(results) == 1:
        return results[0]
    print(f"  Multiple matches for '{arg}' — pick an ID with -s <id>:\n")
    print(f"  {'ID':>4}  {'Name':<30}  {'Type':<12}  Weapon")
    print("  " + "-" * 60)
    for f in results:
        print(f"  {f.id:>4}  {f.name:<30}  {f.skill_type:<12}  {f.weapon_type or ''}")
    return None


def load_marginal_opts(path: str, base: CharStats) -> Dict[str, CharStats]:
    """
    Each section = one equal-cost option. Values are deltas (+/-).
    Supported keys: affinity_rate, precision_rate, critical_rate,
                    direct_affinity_rate, direct_critical_rate,
                    physical_min, physical_max, physical_pen, physical_dmg_bonus,
                    <attr>_min, <attr>_max, <attr>_pen, <attr>_dmg_bonus
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
                s.physical = AtkStat(s.physical.atk_min + delta, s.physical.atk_max, s.physical.pen, s.physical.dmg_bonus)
            elif key == "physical_max":
                s.physical = AtkStat(s.physical.atk_min, s.physical.atk_max + delta, s.physical.pen, s.physical.dmg_bonus)
            elif key == "physical_pen":
                s.physical = AtkStat(s.physical.atk_min, s.physical.atk_max, s.physical.pen + delta, s.physical.dmg_bonus)
            elif key == "physical_dmg_bonus":
                s.physical = AtkStat(s.physical.atk_min, s.physical.atk_max, s.physical.pen, s.physical.dmg_bonus + delta)
            else:
                for attr in ATTRIBUTES:
                    if key == f"{attr}_min":
                        o = s.attributes[attr]
                        s.attributes[attr] = AtkStat(o.atk_min + delta, o.atk_max, o.pen, o.dmg_bonus)
                        break
                    elif key == f"{attr}_max":
                        o = s.attributes[attr]
                        s.attributes[attr] = AtkStat(o.atk_min, o.atk_max + delta, o.pen, o.dmg_bonus)
                        break
                    elif key == f"{attr}_pen":
                        o = s.attributes[attr]
                        s.attributes[attr] = AtkStat(o.atk_min, o.atk_max, o.pen + delta, o.dmg_bonus)
                        break
                    elif key == f"{attr}_dmg_bonus":
                        o = s.attributes[attr]
                        s.attributes[attr] = AtkStat(o.atk_min, o.atk_max, o.pen, o.dmg_bonus + delta)
                        break

    opts = {}
    for name in cfg.sections():
        s = deepcopy(base)
        apply_deltas(s, cfg[name].items())
        # print(s)
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
    def pen_tag(stat: AtkStat) -> str:
        if stat.pen == 0.0 and stat.dmg_bonus == 0.0:
            return ""
        return f"  pen={stat.pen:.1f}  dmg_bonus={stat.dmg_bonus*100:.1f}%  (pen/dmg_bonus x{stat.dmg_mult():.4f})"

    print(f"  Physical   : {s.physical.atk_min} - {s.physical.atk_max}{pen_tag(s.physical)}")
    for name, stat in s.attributes.items():
        if name == s.main_attribute:
            tag = " [x1.0]" if f.is_mystic else " [MAIN x1.5]"
        else:
            tag = " [x1.0]"
        print(f"  {name:<12}: {stat.atk_min} - {stat.atk_max}{pen_tag(stat)}{tag}")
    print(f"  Affinity   : {s.affinity_rate*100:.1f}% + {s.direct_affinity_rate*100:.1f}% direct"
          f"  => eff {s.eff_affinity()*100:.2f}%  (mult x{s.affinity_mult})")
    print(f"  Precision  : {s.precision_rate*100:.1f}%  => eff {s.eff_precision()*100:.2f}%")
    print(f"  Crit rate  : {s.critical_rate*100:.1f}% + {s.direct_critical_rate*100:.1f}% direct"
          f"  => eff {s.eff_crit()*100:.2f}%  (mult x{s.critical_mult})")
    if f.is_mystic:
        formula_str = (f"phys_atk ×{f.phys_coeff} +{f.phys_bonus}flat"
                       f"  +  attr_atk ×{f.phys_coeff}  [MYSTIC]")
    else:
        formula_str = (f"phy_atk ×{f.phys_coeff} +{f.phys_bonus}flat"
                       f"  +  other_attr_atk ×{f.phys_coeff}"
                       f"  +  (main_attr_atk+{f.attr_bonus})×{f.attr_coeff}")
    print(f"  Skill      : {f.name}  ({formula_str})")
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

    hdr = f"  {'Type':<8}  {'avg DMG':>10}  {'range':>22}  {'hits% (theory)':>14}"
    div = f"  {'-'*58}"
    if sim:
        hdr += f"  {'hits% (sim)':>11}"
        div += "-" * 14
    print(hdr); print(div)
    ranges = {
        "orange": "fixed",
        "yellow": f"{r['DMG_yellow_min']:.0f} ~ {r['DMG_yellow_max']:.0f}",
        "white":  f"{r['DMG_white_min']:.0f} ~ {r['DMG_white_max']:.0f}",
        "gray":   "fixed",
    }
    for key, dmg_key in [("orange","DMG_orange"),("yellow","DMG_yellow"),
                         ("white","DMG_white"),("gray","DMG_gray")]:
        row = f"  {key:<8}  {r[dmg_key]:>10.2f}  {ranges[key]:>22}  {r[f'P_{key}']:>13.2f}%"
        if sim:
            row += f"  {sim[f'P_{key}']:>10.2f}%"
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

        total = sim["total"]
        print(f"\n  Total damage over {n_rolls} hits = {total:.0f}")
        print(f"  {'color':<8}  {'total dmg':>10}  {'% of total':>10}")
        print(f"  {'-'*32}")
        for color in ("orange", "yellow", "white", "gray"):
            cdmg = sim[f"total_{color}"]
            pct  = cdmg / total * 100 if total > 0 else 0
            print(f"  {color:<8}  {cdmg:>10.0f}  {pct:>9.1f}%")


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
        def combine(base_stat, sa_stat, sb_stat) -> AtkStat:
            def d(field):
                return (getattr(base_stat, field)
                        + (getattr(sa_stat, field) - getattr(base_stat, field))
                        + (getattr(sb_stat, field) - getattr(base_stat, field)))
            return AtkStat(
                atk_min   = d("atk_min"),
                atk_max   = d("atk_max"),
                pen       = d("pen"),
                dmg_bonus = d("dmg_bonus"),
            )
        sc.physical = combine(base.physical, sa.physical, sb.physical)
        for attr in base.attributes:
            sc.attributes[attr] = combine(
                base.attributes[attr], sa.attributes[attr], sb.attributes[attr]
            )
        rc   = sc.expected_dmg(f)
        gain = rc["E[DMG]"] - base_e
        lbl  = f"{a} + {b}"
        print(f"  {lbl:<52}  +{gain:.2f} DMG  (E={rc['E[DMG]']}  std={rc['std']})")
        pair_results.append((lbl, gain, rc["E[DMG]"], rc["std"]))

    print(f"\n  Top 10:")
    print(f"  {'-'*66}")
    for rank, (lbl, g, e, std) in enumerate(
            sorted(pair_results, key=lambda x: -x[1])[:10], 1):
        print(f"  #{rank:<2}  {lbl:<52}  +{g:.2f}  (E={e}  std={std})")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
try:
    base = load_stats(BASE_CONFIG)
except FileNotFoundError as e:
    print(f"Error: {e}"); sys.exit(1)

args    = sys.argv[1:]
verbose = "-v" in args
args    = [a for a in args if a != "-v"]

# -s <id|name> to select skill from DB
skill_arg = None
if "-s" in args:
    idx = args.index("-s")
    if idx + 1 < len(args):
        skill_arg = args[idx + 1]
        args = [a for i, a in enumerate(args) if i not in (idx, idx + 1)]
    else:
        print("Error: -s requires a skill id or name"); sys.exit(1)

init_db(DB_PATH)
active_formula: Optional[SkillFormula] = None

if skill_arg:
    active_formula = _resolve_skill(skill_arg, DB_PATH)
    if not active_formula:
        sys.exit(1)
    print(f"Active skill   : [{active_formula.id}] {active_formula.name}")
elif DEFAULT_SKILL_ID is not None:
    active_formula = get_skill_by_id(DEFAULT_SKILL_ID, DB_PATH)
    if not active_formula:
        print(f"Error: DEFAULT_SKILL_ID={DEFAULT_SKILL_ID} not found in DB."); sys.exit(1)
    print(f"Active skill   : [{active_formula.id}] {active_formula.name}")
else:
    all_skills = list_skills(DB_PATH)
    if all_skills:
        print(f"Available skills ({len(all_skills)}):")
        for f in all_skills:
            print(f"  [{f.id}] {f.name}  ({f.skill_type}{'  DOT' if f.is_dot else ''})")
        print("\nUse -s <id> or -s <name> to select a skill, or set DEFAULT_SKILL_ID.")
        sys.exit(0)
    else:
        print("No skills in DB. Use 'python skill_db.py add' to add one.")
        active_formula = SkillFormula()
        print("Using default formula (phys×1.0  attr×1.0  bonus=0)")

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