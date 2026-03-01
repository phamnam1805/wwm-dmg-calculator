#!/usr/bin/env python3
"""
dmg_resolver.py — Damage resolver (profile + skill → damage)
=============================================================
Bridges CharProfile (profile_db.py) and SkillFormula (skill_db.py).

Damage formula
--------------
  base_dmg   = phys_dmg + attr_dmg

  buff_mult  (martial art): 1 + all_martial_art_dmg_bonus + {weapon_type}_dmg_bonus
  buff_mult  (mystic):      1 + {mystic_type}_dmg_bonus

  affix_mult: 1.0  (placeholder — reserved for future affix system)

  scaled_dmg = base_dmg × buff_mult × affix_mult

  Hit types (applied to scaled_dmg):
    orange : affinity  — uses max-roll  × affinity_mult
    yellow : crit      — uses e[roll]   × critical_mult
    white  : normal    — uses e[roll]
    gray   : miss      — uses min-roll

Library API  : resolve(), simulate(), apply_gain(), compare_gains()
               — no print statements, safe to import from Web UI

Interactive CLI: python src/dmg_resolver.py
  Modes available per session:
    [1] Simulate            — Monte Carlo with configurable n_rolls
    [2] Marginal gain comparison — compare gains from marginal_gains.db
"""

import random
import sys
from dataclasses import dataclass, replace as dc_replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from profile_db import (
    CharProfile,
    init_db as init_profile_db,
    list_profiles, get_profile_by_id,
    ATTRIBUTES, WEAPON_TYPES, MYSTIC_TYPES,
)
from skill_db import (
    SkillFormula,
    init_all as init_skill_dbs,
    list_skills, get_skill_by_id,
)
from marginal_gain_db import (
    MarginalGain,
    init_db as init_gain_db,
    list_gains,
)

# ─────────────────────────────────────────────
# Rate caps  (mirrors calculator.py)
# ─────────────────────────────────────────────
AFFINITY_CAP        = 0.40
DIRECT_AFFINITY_CAP = 0.10
CRIT_CAP            = 0.80
DIRECT_CRIT_CAP     = 0.20
PRECISION_CAP       = 1.00


# ─────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────
@dataclass
class DamageResult:
    """Analytical expected damage breakdown."""
    # Expected value & dispersion
    e_dmg:          float
    std:            float
    # Hit-type probabilities (%)
    p_orange:       float
    p_yellow:       float
    p_white:        float
    p_gray:         float
    # Per-type representative damage
    dmg_orange:     float
    dmg_yellow:     float
    dmg_yellow_min: float
    dmg_yellow_max: float
    dmg_white:      float
    dmg_white_min:  float
    dmg_white_max:  float
    dmg_gray:       float
    # Multiplier breakdown
    buff_mult:      float
    affix_mult:     float
    # Base damage (before buff / affix / hit-type mult)
    base_max:       float
    base_e_roll:    float
    base_min:       float


@dataclass
class SimulateResult:
    """Monte Carlo simulation results."""
    n_rolls:      int
    mean:         float
    std:          float
    min_dmg:      float
    max_dmg:      float
    p10:          float
    p90:          float
    total:        float
    p_orange:     float
    p_yellow:     float
    p_white:      float
    p_gray:       float
    total_orange: float
    total_yellow: float
    total_white:  float
    total_gray:   float


# ─────────────────────────────────────────────
# Internal stat helpers
# ─────────────────────────────────────────────
def _eff_affinity(p: CharProfile) -> float:
    return min(p.affinity_rate, AFFINITY_CAP) + min(p.direct_affinity_rate, DIRECT_AFFINITY_CAP)


def _eff_precision(p: CharProfile) -> float:
    return min(p.precision_rate, PRECISION_CAP)


def _eff_crit(p: CharProfile) -> float:
    return min(p.critical_rate, CRIT_CAP) + min(p.direct_critical_rate, DIRECT_CRIT_CAP)


def _phys_dmg_mult(p: CharProfile) -> float:
    return (1 + p.physical_pen / 200) * (1 + p.physical_dmg_bonus)


def _attr_dmg_mult(p: CharProfile, attr: str) -> float:
    pen       = getattr(p, f"{attr}_pen")
    dmg_bonus = getattr(p, f"{attr}_dmg_bonus")
    return (1 + pen / 200) * (1 + dmg_bonus)


def _buff_mult(p: CharProfile, skill: SkillFormula) -> float:
    """
    buff_mult for martial art : 1 + all_martial_art_dmg_bonus + {weapon_type}_dmg_bonus
    buff_mult for mystic       : 1 + {mystic_type}_dmg_bonus
    """
    if skill.is_mystic:
        mt    = skill.mystic_type or ""
        bonus = getattr(p, f"{mt}_dmg_bonus", 0.0) if mt else 0.0
        return 1.0 + bonus
    else:
        wt    = skill.weapon_type or ""
        bonus = getattr(p, f"{wt}_dmg_bonus", 0.0) if wt else 0.0
        return 1.0 + p.all_martial_art_dmg_bonus + bonus


def _affix_mult() -> float:
    """Placeholder — reserved for future affix system."""
    return 1.0


def _base_phys(p: CharProfile, skill: SkillFormula, phys_roll: float) -> float:
    """Physical damage component for a given physical atk roll."""
    return (skill.phys_coeff * phys_roll + skill.phys_bonus) * _phys_dmg_mult(p)


def _base_attr(p: CharProfile, skill: SkillFormula, attr_rolls: Dict[str, float]) -> float:
    """Attribute damage component for given per-attribute atk rolls."""
    total = 0.0
    for attr in ATTRIBUTES:
        roll = attr_rolls[attr]
        if skill.is_mystic:
            coeff = skill.phys_coeff   # mystic: same coeff for all attrs
            bonus = 0.0
        else:
            main  = (attr == skill.attribute_type)
            coeff = skill.attr_coeff if main else skill.phys_coeff
            bonus = skill.attr_bonus if main else 0.0
        total += (coeff * (roll + bonus)) * _attr_dmg_mult(p, attr)
    return total


def _base_max(p: CharProfile, skill: SkillFormula) -> float:
    return _base_phys(p, skill, p.physical_max) + _base_attr(
        p, skill, {a: getattr(p, f"{a}_max") for a in ATTRIBUTES}
    )


def _base_min(p: CharProfile, skill: SkillFormula) -> float:
    return _base_phys(p, skill, p.physical_min) + _base_attr(
        p, skill, {a: getattr(p, f"{a}_min") for a in ATTRIBUTES}
    )


def _base_e_roll(p: CharProfile, skill: SkillFormula) -> float:
    return _base_phys(p, skill, (p.physical_min + p.physical_max) / 2) + _base_attr(
        p, skill,
        {a: (getattr(p, f"{a}_min") + getattr(p, f"{a}_max")) / 2 for a in ATTRIBUTES},
    )


def _base_roll_rng(p: CharProfile, skill: SkillFormula, rng: random.SystemRandom) -> float:
    phys_roll  = rng.uniform(p.physical_min, p.physical_max)
    attr_rolls = {a: rng.uniform(getattr(p, f"{a}_min"), getattr(p, f"{a}_max"))
                  for a in ATTRIBUTES}
    return _base_phys(p, skill, phys_roll) + _base_attr(p, skill, attr_rolls)


# ─────────────────────────────────────────────
# Library API  (no print statements — safe to import from Web UI)
# ─────────────────────────────────────────────
def resolve(profile: CharProfile, skill: SkillFormula) -> DamageResult:
    """
    Analytical expected damage.
    Returns a DamageResult with full breakdown.
    """
    bm  = _buff_mult(profile, skill)
    am  = _affix_mult()
    a   = _eff_affinity(profile)
    p   = _eff_precision(profile)
    c   = _eff_crit(profile)
    aff = profile.affinity_mult
    crt = profile.critical_mult

    base_mx   = _base_max(profile, skill)
    base_er   = _base_e_roll(profile, skill)
    base_mn   = _base_min(profile, skill)

    # Scale base damage by buff + affix multipliers
    scaled_max    = base_mx * bm * am
    scaled_e_roll = base_er * bm * am
    scaled_min    = base_mn * bm * am

    # Hit-type representative damage
    dmg_orange = scaled_max   * aff
    dmg_yellow = scaled_e_roll * crt
    dmg_white  = scaled_e_roll
    dmg_gray   = scaled_min

    # Hit-type probabilities
    prob_orange = a
    prob_yellow = (1 - a) * p * c
    prob_white  = (1 - a) * p * (1 - c)
    prob_gray   = (1 - a) * (1 - p)

    e_dmg = (prob_orange * dmg_orange + prob_yellow * dmg_yellow
           + prob_white  * dmg_white  + prob_gray   * dmg_gray)

    # ── Variance via Law of Total Variance ──────────────
    # Var[uniform(lo, hi)] = (hi - lo)² / 12
    def var_uni(lo: float, hi: float) -> float:
        return (hi - lo) ** 2 / 12

    # Physical contribution  (phys_bonus is constant → zero variance)
    var_roll = var_uni(
        skill.phys_coeff * profile.physical_min * _phys_dmg_mult(profile),
        skill.phys_coeff * profile.physical_max * _phys_dmg_mult(profile),
    )
    # Attribute contributions  (attr_bonus is constant → excluded from variance)
    for attr in ATTRIBUTES:
        if skill.is_mystic:
            coeff = skill.phys_coeff
        else:
            main  = (attr == skill.attribute_type)
            coeff = skill.attr_coeff if main else skill.phys_coeff
        m = coeff * _attr_dmg_mult(profile, attr)
        var_roll += var_uni(
            getattr(profile, f"{attr}_min") * m,
            getattr(profile, f"{attr}_max") * m,
        )

    # Scale variance by (buff × affix)²
    var_scaled = var_roll * (bm * am) ** 2
    e2_scaled  = var_scaled + scaled_e_roll ** 2

    e_x2 = (prob_orange * dmg_orange ** 2
          + prob_yellow * e2_scaled * crt ** 2
          + prob_white  * e2_scaled
          + prob_gray   * dmg_gray  ** 2)
    std = (e_x2 - e_dmg ** 2) ** 0.5

    return DamageResult(
        e_dmg          = round(e_dmg, 2),
        std            = round(std,   2),
        p_orange       = round(prob_orange * 100, 2),
        p_yellow       = round(prob_yellow * 100, 2),
        p_white        = round(prob_white  * 100, 2),
        p_gray         = round(prob_gray   * 100, 2),
        dmg_orange     = round(dmg_orange, 2),
        dmg_yellow     = round(dmg_yellow, 2),
        dmg_yellow_min = round(dmg_gray * crt, 2),
        dmg_yellow_max = round(scaled_max * crt, 2),
        dmg_white      = round(dmg_white,  2),
        dmg_white_min  = round(dmg_gray,   2),
        dmg_white_max  = round(scaled_max, 2),
        dmg_gray       = round(dmg_gray,   2),
        buff_mult      = round(bm, 6),
        affix_mult     = round(am, 6),
        base_max       = round(base_mx, 2),
        base_e_roll    = round(base_er, 2),
        base_min       = round(base_mn, 2),
    )


def simulate(profile: CharProfile, skill: SkillFormula, n_rolls: int = 100) -> SimulateResult:
    """
    Monte Carlo simulation using /dev/urandom (SystemRandom).
    Returns a SimulateResult with per-roll statistics.
    """
    bm  = _buff_mult(profile, skill)
    am  = _affix_mult()
    a   = _eff_affinity(profile)
    p   = _eff_precision(profile)
    c   = _eff_crit(profile)
    aff = profile.affinity_mult
    crt = profile.critical_mult

    base_mx = _base_max(profile, skill)
    base_mn = _base_min(profile, skill)

    rng     = random.SystemRandom()
    counts  = {"orange": 0, "yellow": 0, "white": 0, "gray": 0}
    dmg_sum = {"orange": 0.0, "yellow": 0.0, "white": 0.0, "gray": 0.0}
    dmg_log: List[float] = []

    for _ in range(n_rolls):
        if rng.random() < a:
            dmg   = base_mx * bm * am * aff
            color = "orange"
        elif rng.random() >= p:
            dmg   = base_mn * bm * am
            color = "gray"
        else:
            roll = _base_roll_rng(profile, skill, rng)
            scaled = roll * bm * am
            if rng.random() < c:
                dmg   = scaled * crt
                color = "yellow"
            else:
                dmg   = scaled
                color = "white"
        counts[color]  += 1
        dmg_sum[color] += dmg
        dmg_log.append(dmg)

    dmg_log.sort()
    n     = len(dmg_log)
    mean  = sum(dmg_log) / n
    std   = (sum((x - mean) ** 2 for x in dmg_log) / n) ** 0.5
    total = sum(dmg_log)

    def pct(q: float) -> float:
        return dmg_log[min(int(q / 100 * n), n - 1)]

    return SimulateResult(
        n_rolls      = n_rolls,
        mean         = round(mean, 2),
        std          = round(std,  2),
        min_dmg      = round(dmg_log[0],  2),
        max_dmg      = round(dmg_log[-1], 2),
        p10          = round(pct(10), 2),
        p90          = round(pct(90), 2),
        total        = round(total, 2),
        p_orange     = round(counts["orange"] / n * 100, 2),
        p_yellow     = round(counts["yellow"] / n * 100, 2),
        p_white      = round(counts["white"]  / n * 100, 2),
        p_gray       = round(counts["gray"]   / n * 100, 2),
        total_orange = round(dmg_sum["orange"], 2),
        total_yellow = round(dmg_sum["yellow"], 2),
        total_white  = round(dmg_sum["white"],  2),
        total_gray   = round(dmg_sum["gray"],   2),
    )


def apply_gain(profile: CharProfile, gain: MarginalGain) -> CharProfile:
    """
    Return a new CharProfile with gain's deltas applied to matching fields.
    Fields in gain.deltas that don't exist on CharProfile are silently ignored.
    Does NOT modify the original profile.
    """
    from dataclasses import fields as dc_fields
    known = {f.name for f in dc_fields(CharProfile)}
    kw: Dict[str, float] = {}
    for field, delta in gain.deltas.items():
        if field in known:
            kw[field] = getattr(profile, field) + delta
    return dc_replace(profile, **kw)


def compare_gains(
    profile: CharProfile,
    skill:   SkillFormula,
    gains:   List[MarginalGain],
) -> List[Tuple[MarginalGain, DamageResult, float]]:
    """
    Compare a list of marginal gains against the base profile.
    Returns [(gain, result_with_gain, delta_e_dmg), ...] sorted by delta descending.
    """
    base_e = resolve(profile, skill).e_dmg
    results = []
    for gain in gains:
        modified = apply_gain(profile, gain)
        result   = resolve(modified, skill)
        delta    = result.e_dmg - base_e
        results.append((gain, result, delta))
    results.sort(key=lambda x: -x[2])
    return results


# ─────────────────────────────────────────────
# Print helpers  (CLI-only — not part of library API)
# ─────────────────────────────────────────────
_SEP = "=" * 60
_DIV = "-" * 60


def _print_profile_line(p: CharProfile) -> None:
    eff_aff  = _eff_affinity(p)
    eff_crit = _eff_crit(p)
    print(f"  {p.id:>4}  {p.name:<28}  {eff_aff*100:>7.1f}%  {eff_crit*100:>8.1f}%")


def _print_skill_line(f: SkillFormula) -> None:
    stype = "mystic" if f.is_mystic else "martial"
    sub   = f.mystic_type or f.weapon_type or ""
    dot   = "  DOT" if f.is_dot else ""
    print(f"  {f.id:>4}  {f.name:<30}  {stype:<10}  {sub:<24}{dot}")


def _print_resolve(r: DamageResult, profile: CharProfile, skill: SkillFormula) -> None:
    print(f"\n{_SEP}")
    print(f"  {profile.name}  ·  {skill.name}  ({skill.skill_type})")
    print(_SEP)

    # buff_mult breakdown
    if skill.is_mystic:
        mt      = skill.mystic_type or ""
        mt_b    = getattr(profile, f"{mt}_dmg_bonus", 0.0) if mt else 0.0
        print(f"  buff_mult  = 1.00 + {mt_b:.4f} ({mt or '—'}) = {r.buff_mult:.6f}")
    else:
        wt      = skill.weapon_type or ""
        wt_b    = getattr(profile, f"{wt}_dmg_bonus", 0.0) if wt else 0.0
        all_b   = profile.all_martial_art_dmg_bonus
        print(f"  buff_mult  = 1.00"
              f" + {all_b:.4f} (all martial)"
              f" + {wt_b:.4f} ({wt or '—'}) = {r.buff_mult:.6f}")
    print(f"  affix_mult = {r.affix_mult:.6f}  (placeholder)")
    print()
    print(f"  Base DMG  :  max={r.base_max:.1f}  E[roll]={r.base_e_roll:.1f}  min={r.base_min:.1f}")
    print()
    print(f"  E[DMG]    = {r.e_dmg}  ±{r.std}")
    print()

    hdr = f"  {'Type':<8}  {'Avg DMG':>10}  {'Range':>26}  {'P%':>8}"
    div = f"  {'-'*57}"
    print(hdr); print(div)
    ranges = {
        "orange": "fixed",
        "yellow": f"{r.dmg_yellow_min:.0f} ~ {r.dmg_yellow_max:.0f}",
        "white":  f"{r.dmg_white_min:.0f} ~ {r.dmg_white_max:.0f}",
        "gray":   "fixed",
    }
    for color, dmg_attr in [("orange","dmg_orange"), ("yellow","dmg_yellow"),
                              ("white","dmg_white"),  ("gray","dmg_gray")]:
        dmg  = getattr(r, dmg_attr)
        prob = getattr(r, f"p_{color}")
        print(f"  {color:<8}  {dmg:>10.2f}  {ranges[color]:>26}  {prob:>7.2f}%")


def _print_simulate(r: DamageResult, sim: SimulateResult,
                    profile: CharProfile, skill: SkillFormula) -> None:
    _print_resolve(r, profile, skill)
    print()
    print(f"  Simulation  (n={sim.n_rolls})")
    print(f"  {'-'*57}")
    print(f"  E[DMG] sim  = {sim.mean}  ±{sim.std}")
    print()

    print(f"  {'Stat':<15}  {'Theory':>12}  {'Sim':>12}")
    print(f"  {'-'*43}")
    for lbl, tv, sv in [
        ("mean/E[DMG]", str(r.e_dmg),      str(sim.mean)),
        ("std",         str(r.std),         str(sim.std)),
        ("min",         str(r.dmg_gray),    str(sim.min_dmg)),
        ("p10",         "—",               str(sim.p10)),
        ("p90",         "—",               str(sim.p90)),
        ("max",         "—",               str(sim.max_dmg)),
    ]:
        print(f"  {lbl:<15}  {tv:>12}  {sv:>12}")

    print(f"\n  Total over {sim.n_rolls} hits = {sim.total:.0f}")
    print(f"  {'Color':<8}  {'Total DMG':>12}  {'Sim %':>8}  {'Theory %':>10}")
    print(f"  {'-'*44}")
    for color in ("orange", "yellow", "white", "gray"):
        tot_c  = getattr(sim, f"total_{color}")
        pct_s  = tot_c / sim.total * 100 if sim.total > 0 else 0
        pct_t  = getattr(r, f"p_{color}")
        print(f"  {color:<8}  {tot_c:>12.0f}  {pct_s:>7.1f}%  {pct_t:>9.1f}%")


def _print_comparison(
    base_result: DamageResult,
    comparison:  List[Tuple[MarginalGain, DamageResult, float]],
    profile:     CharProfile,
    skill:       SkillFormula,
) -> None:
    print(f"\n{_SEP}")
    print(f"  Marginal gain comparison")
    print(f"  Profile: {profile.name}  ·  Skill: {skill.name}")
    print(f"  Baseline  E[DMG] = {base_result.e_dmg}  ±{base_result.std}")
    print(_SEP)
    print(f"  {'Gain':<32}  {'E[DMG]':>10}  {'Δ DMG':>9}  {'Δ%':>7}")
    print(f"  {'-'*62}")
    base_e = base_result.e_dmg
    for gain, result, delta in comparison:
        pct = delta / base_e * 100 if base_e != 0 else 0.0
        print(f"  {gain.name:<32}  {result.e_dmg:>10.2f}  {delta:>+9.2f}  {pct:>+6.2f}%")


# ─────────────────────────────────────────────
# CLI interaction helpers
# ─────────────────────────────────────────────
def _select_profile() -> Optional[CharProfile]:
    """Print profile list and let user pick one by ID."""
    profiles = list_profiles()
    if not profiles:
        print("  No profiles found. Use 'profile_db.py add' to create one.")
        return None
    print(f"\n  {'ID':>4}  {'Name':<28}  {'Eff Aff%':>8}  {'Eff Crit%':>9}")
    print("  " + "-" * 56)
    for p in profiles:
        _print_profile_line(p)
    print()
    try:
        raw = input("  Profile ID (q=quit): ").strip()
    except (KeyboardInterrupt, EOFError):
        return None
    if raw.lower() in ("q", "quit", ""):
        return None
    try:
        pid = int(raw)
    except ValueError:
        print("  Not a valid ID."); return None
    p = get_profile_by_id(pid)
    if not p:
        print(f"  Profile ID={pid} not found."); return None
    return p


def _select_skill() -> Optional[SkillFormula]:
    """Print skill list with sequential row numbers; user picks by row #."""
    skills = list_skills()
    if not skills:
        print("  No skills found. Use 'skill_db.py martial add' to create one.")
        return None
    print(f"\n  {'#':>4}  {'Name':<30}  {'Type':<10}  {'Weapon / MysticType'}")
    print("  " + "-" * 72)
    for i, f in enumerate(skills, 1):
        stype = "mystic" if f.is_mystic else "martial"
        sub   = f.mystic_type or f.weapon_type or ""
        dot   = "  DOT" if f.is_dot else ""
        print(f"  {i:>4}  {f.name:<30}  {stype:<10}  {sub:<24}{dot}")
    print()
    try:
        raw = input(f"  Skill # 1-{len(skills)} (q=back): ").strip()
    except (KeyboardInterrupt, EOFError):
        return None
    if raw.lower() in ("q", "quit", ""):
        return None
    try:
        idx = int(raw)
    except ValueError:
        print("  Not a valid number."); return None
    if not (1 <= idx <= len(skills)):
        print(f"  Out of range (1–{len(skills)})."); return None
    return skills[idx - 1]


def _mode_simulate(profile: CharProfile, skill: SkillFormula) -> None:
    try:
        raw = input("  Number of rolls [100]: ").strip()
    except (KeyboardInterrupt, EOFError):
        return
    try:
        n_rolls = int(raw) if raw else 100
        if n_rolls <= 0:
            raise ValueError
    except ValueError:
        print("  Invalid number of rolls. Using 100.")
        n_rolls = 100

    r   = resolve(profile, skill)
    sim = simulate(profile, skill, n_rolls)
    _print_simulate(r, sim, profile, skill)


def _mode_marginal(profile: CharProfile, skill: SkillFormula) -> None:
    gains = list_gains()
    if not gains:
        print("  No marginal gains found. Use 'marginal_gain_db.py add' to create some.")
        return

    print(f"\n  {'ID':>4}  {'Name':<32}  Deltas")
    print("  " + "-" * 72)
    for g in gains:
        parts = [f"{f} {d:+.4g}" for f, d in sorted(g.deltas.items())]
        delta_str = ", ".join(parts) if parts else "(no deltas)"
        if len(delta_str) > 36:
            delta_str = delta_str[:35] + "…"
        print(f"  {g.id:>4}  {g.name:<32}  {delta_str}")
    print()

    try:
        raw = input("  Gain IDs (comma-separated, or 'all'): ").strip()
    except (KeyboardInterrupt, EOFError):
        return
    if not raw:
        return

    selected: List[MarginalGain] = []
    if raw.lower() == "all":
        selected = gains
    else:
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                gid = int(part)
            except ValueError:
                print(f"  '{part}' is not a valid ID, skipping."); continue
            g = next((g for g in gains if g.id == gid), None)
            if not g:
                print(f"  Gain ID={gid} not found, skipping.")
            else:
                selected.append(g)

    if not selected:
        print("  No gains selected."); return

    base_result = resolve(profile, skill)
    comparison  = compare_gains(profile, skill, selected)
    _print_comparison(base_result, comparison, profile, skill)


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────
def _cli() -> None:
    init_profile_db()
    init_skill_dbs()
    init_gain_db()

    print(f"\n{'═'*60}")
    print("  WWM Damage Resolver")
    print(f"{'═'*60}")

    while True:
        # ── Select profile ────────────────────
        print(f"\n{_SEP}\n  Select Profile\n{_SEP}")
        profile = _select_profile()
        if profile is None:
            print("  Goodbye."); return

        # ── Select skill ──────────────────────
        while True:
            print(f"\n{_SEP}\n  Select Skill\n{_SEP}")
            skill = _select_skill()
            if skill is None:
                break  # back to profile selection

            # ── Mode loop ─────────────────────
            while True:
                print(f"\n{_SEP}")
                print(f"  Profile : {profile.name}")
                print(f"  Skill   : {skill.name}  ({skill.skill_type})")
                print(_SEP)
                print("  [1] Simulate")
                print("  [2] Marginal gain comparison")
                print("  [3] Change skill")
                print("  [4] Change profile")
                print("  [0] Quit")
                print()
                try:
                    choice = input("  Mode: ").strip()
                except (KeyboardInterrupt, EOFError):
                    print("\n  Goodbye."); return

                if choice == "1":
                    _mode_simulate(profile, skill)
                elif choice == "2":
                    _mode_marginal(profile, skill)
                elif choice == "3":
                    break    # re-select skill
                elif choice == "4":
                    skill = None; break   # break inner, break outer to profile
                elif choice == "0":
                    print("  Goodbye."); return
                else:
                    print("  Invalid choice.")

            if skill is None:
                break   # back to outer while → profile selection


if __name__ == "__main__":
    _cli()
