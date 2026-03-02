#!/usr/bin/env python3
"""
profile_db.py — Character profile database (SQLite)
====================================================
Each profile stores a full set of character stats plus damage bonus modifiers:

  · all_martial_art_dmg_bonus  — bonus applied to all martial art skills
  · per-weapon-type dmg bonus  — sword, dual_blades, spear, fan, umbrella,
                                  heng_blade, mo_blade, rope_dart
  · per-mystic-type dmg bonus  — area_debuff, area_dmg,
                                  single_target_control, single_target_burst

Library API  : import and call functions directly (no print, safe for Web UI)
Interactive CLI: python src/profile_db.py [command] [args]

Commands:
  list                 list all profiles
  show      <id>       show full profile detail
  add                  interactive wizard to add a new profile
  edit      <id>       interactive wizard to edit a profile
  duplicate <id>       copy a profile to a new ID, with optional edits
  remove    <id>       delete profile by ID
  search    <name>     fuzzy search by name
  init                 initialise DB (runs automatically on first use)
"""

import sqlite3
import sys
from pathlib import Path
from dataclasses import dataclass, fields as dc_fields, replace as dc_replace
from typing import List, Optional

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DB_PATH      = str(Path(__file__).parent.parent / "dbs" / "profiles.db")
ATTRIBUTES   = ("bellstrike", "stonesplit", "bamboocut", "silkbind")
WEAPON_TYPES = ("sword", "dual_blades", "spear", "fan", "umbrella",
                "heng_blade", "mo_blade", "rope_dart")
MYSTIC_TYPES = ("area_debuff", "area_dmg",
                "single_target_control", "single_target_burst")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,

    -- Physical
    physical_min       REAL NOT NULL DEFAULT 0.0,
    physical_max       REAL NOT NULL DEFAULT 0.0,
    physical_pen       REAL NOT NULL DEFAULT 0.0,
    physical_dmg_bonus REAL NOT NULL DEFAULT 0.0,

    -- Attribute: bellstrike
    bellstrike_min       REAL NOT NULL DEFAULT 0.0,
    bellstrike_max       REAL NOT NULL DEFAULT 0.0,
    bellstrike_pen       REAL NOT NULL DEFAULT 0.0,
    bellstrike_dmg_bonus REAL NOT NULL DEFAULT 0.0,

    -- Attribute: stonesplit
    stonesplit_min       REAL NOT NULL DEFAULT 0.0,
    stonesplit_max       REAL NOT NULL DEFAULT 0.0,
    stonesplit_pen       REAL NOT NULL DEFAULT 0.0,
    stonesplit_dmg_bonus REAL NOT NULL DEFAULT 0.0,

    -- Attribute: bamboocut
    bamboocut_min       REAL NOT NULL DEFAULT 0.0,
    bamboocut_max       REAL NOT NULL DEFAULT 0.0,
    bamboocut_pen       REAL NOT NULL DEFAULT 0.0,
    bamboocut_dmg_bonus REAL NOT NULL DEFAULT 0.0,

    -- Attribute: silkbind
    silkbind_min       REAL NOT NULL DEFAULT 0.0,
    silkbind_max       REAL NOT NULL DEFAULT 0.0,
    silkbind_pen       REAL NOT NULL DEFAULT 0.0,
    silkbind_dmg_bonus REAL NOT NULL DEFAULT 0.0,

    -- Combat rates (stored as decimals: 0.374 = 37.4%)
    affinity_rate        REAL NOT NULL DEFAULT 0.0,
    precision_rate       REAL NOT NULL DEFAULT 1.0,
    critical_rate        REAL NOT NULL DEFAULT 0.0,
    affinity_mult        REAL NOT NULL DEFAULT 1.35,
    critical_mult        REAL NOT NULL DEFAULT 1.5,
    direct_affinity_rate REAL NOT NULL DEFAULT 0.0,
    direct_critical_rate REAL NOT NULL DEFAULT 0.0,

    -- Global martial-art DMG bonus (additive)
    all_martial_art_dmg_bonus REAL NOT NULL DEFAULT 0.0,

    -- Weapon-type DMG bonus
    sword_dmg_bonus       REAL NOT NULL DEFAULT 0.0,
    dual_blades_dmg_bonus REAL NOT NULL DEFAULT 0.0,
    spear_dmg_bonus       REAL NOT NULL DEFAULT 0.0,
    fan_dmg_bonus         REAL NOT NULL DEFAULT 0.0,
    umbrella_dmg_bonus    REAL NOT NULL DEFAULT 0.0,
    heng_blade_dmg_bonus  REAL NOT NULL DEFAULT 0.0,
    mo_blade_dmg_bonus    REAL NOT NULL DEFAULT 0.0,
    rope_dart_dmg_bonus   REAL NOT NULL DEFAULT 0.0,

    -- Mystic-type DMG bonus
    area_debuff_dmg_bonus           REAL NOT NULL DEFAULT 0.0,
    area_dmg_dmg_bonus              REAL NOT NULL DEFAULT 0.0,
    single_target_control_dmg_bonus REAL NOT NULL DEFAULT 0.0,
    single_target_burst_dmg_bonus   REAL NOT NULL DEFAULT 0.0,

    -- Target DMG bonus
    pvp_dmg_bonus  REAL NOT NULL DEFAULT 0.0,
    boss_dmg_bonus REAL NOT NULL DEFAULT 0.0
);
"""


# ─────────────────────────────────────────────
# CharProfile dataclass
# ─────────────────────────────────────────────
@dataclass
class CharProfile:

    def __post_init__(self):
        self.normalize()

    def normalize(self) -> None:
        # Ensure *_min <= *_max for all stat pairs
        for f in dc_fields(self):
            if not f.name.endswith("_min"):
                continue

            base = f.name[:-4]          # remove "_min"
            max_name = f"{base}_max"

            if not hasattr(self, max_name):
                continue

            vmin = getattr(self, f.name)
            vmax = getattr(self, max_name)

            if vmin > vmax:
                setattr(self, max_name, vmin)

        # Cap combat rates
        caps = {
            "affinity_rate":        0.40,
            "critical_rate":        0.80,
            "precision_rate":       1.00,
            "direct_affinity_rate": 0.10,
            "direct_critical_rate": 0.20,
        }

        for field, cap in caps.items():
            value = getattr(self, field)
            if value < 0:
                setattr(self, field, 0.0)
            elif value > cap:
                setattr(self, field, cap)

    # Metadata
    id:   Optional[int] = None
    name: str           = "default"

    # Physical
    physical_min:       float = 0.0
    physical_max:       float = 0.0
    physical_pen:       float = 0.0
    physical_dmg_bonus: float = 0.0

    # Attribute: bellstrike
    bellstrike_min:       float = 0.0
    bellstrike_max:       float = 0.0
    bellstrike_pen:       float = 0.0
    bellstrike_dmg_bonus: float = 0.0

    # Attribute: stonesplit
    stonesplit_min:       float = 0.0
    stonesplit_max:       float = 0.0
    stonesplit_pen:       float = 0.0
    stonesplit_dmg_bonus: float = 0.0

    # Attribute: bamboocut
    bamboocut_min:       float = 0.0
    bamboocut_max:       float = 0.0
    bamboocut_pen:       float = 0.0
    bamboocut_dmg_bonus: float = 0.0

    # Attribute: silkbind
    silkbind_min:       float = 0.0
    silkbind_max:       float = 0.0
    silkbind_pen:       float = 0.0
    silkbind_dmg_bonus: float = 0.0

    # Combat rates (decimals: 0.374 = 37.4%)
    affinity_rate:        float = 0.0
    precision_rate:       float = 1.0
    critical_rate:        float = 0.0
    affinity_mult:        float = 1.35
    critical_mult:        float = 1.5
    direct_affinity_rate: float = 0.0
    direct_critical_rate: float = 0.0

    # Global DMG bonus
    all_martial_art_dmg_bonus: float = 0.0

    # Weapon-type DMG bonus
    sword_dmg_bonus:       float = 0.0
    dual_blades_dmg_bonus: float = 0.0
    spear_dmg_bonus:       float = 0.0
    fan_dmg_bonus:         float = 0.0
    umbrella_dmg_bonus:    float = 0.0
    heng_blade_dmg_bonus:  float = 0.0
    mo_blade_dmg_bonus:    float = 0.0
    rope_dart_dmg_bonus:   float = 0.0

    # Mystic-type DMG bonus
    area_debuff_dmg_bonus:           float = 0.0
    area_dmg_dmg_bonus:              float = 0.0
    single_target_control_dmg_bonus: float = 0.0
    single_target_burst_dmg_bonus:   float = 0.0

    # Target DMG bonus (additive into buff_mult)
    pvp_dmg_bonus:  float = 0.0
    boss_dmg_bonus: float = 0.0


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────
def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _all_columns() -> List[str]:
    """All CharProfile field names except 'id' (used to build INSERT/UPDATE SQL)."""
    return [f.name for f in dc_fields(CharProfile) if f.name != "id"]


def _row_to_profile(row: sqlite3.Row) -> CharProfile:
    """Convert a DB row to CharProfile, ignoring any unknown/legacy columns."""
    known = {f.name for f in dc_fields(CharProfile)}
    return CharProfile(**{k: v for k, v in dict(row).items() if k in known})


# ─────────────────────────────────────────────
# Library API  (no print statements — safe to import from Web UI)
# ─────────────────────────────────────────────
def init_db(db_path: str = DB_PATH) -> None:
    """Create the profiles table if it does not exist. Safe to call multiple times."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA)
        existing = {r[1] for r in conn.execute("PRAGMA table_info(profiles)").fetchall()}
        # Auto-migrate: drop legacy column
        if "main_attribute" in existing:
            try:
                conn.execute("ALTER TABLE profiles DROP COLUMN main_attribute")
            except Exception:
                pass  # SQLite < 3.35.0 — column stays orphaned but is filtered on read
        # Auto-migrate: add new target DMG bonus columns
        for col in ("pvp_dmg_bonus", "boss_dmg_bonus"):
            if col not in existing:
                conn.execute(f"ALTER TABLE profiles ADD COLUMN {col} REAL NOT NULL DEFAULT 0.0")


def add_profile(profile: CharProfile, db_path: str = DB_PATH) -> int:
    """Insert a new profile. Returns the auto-assigned ID."""
    init_db(db_path)
    cols   = _all_columns()
    values = [getattr(profile, c) for c in cols]
    ph     = ", ".join("?" * len(cols))
    with _connect(db_path) as conn:
        cur = conn.execute(
            f"INSERT INTO profiles ({', '.join(cols)}) VALUES ({ph})",
            values,
        )
        return cur.lastrowid


def update_profile(profile: CharProfile, db_path: str = DB_PATH) -> None:
    """Update an existing profile by profile.id. Raises ValueError if id is None."""
    if profile.id is None:
        raise ValueError("Cannot update a profile without an ID.")
    cols       = _all_columns()
    set_clause = ", ".join(f"{c}=?" for c in cols)
    values     = [getattr(profile, c) for c in cols] + [profile.id]
    with _connect(db_path) as conn:
        conn.execute(f"UPDATE profiles SET {set_clause} WHERE id=?", values)


def remove_profile(profile_id: int, db_path: str = DB_PATH) -> None:
    """Delete a profile by ID. Silent if the ID does not exist."""
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM profiles WHERE id=?", (profile_id,))


def get_profile_by_id(profile_id: int, db_path: str = DB_PATH) -> Optional[CharProfile]:
    """Return a CharProfile by ID, or None if not found."""
    init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM profiles WHERE id=?", (profile_id,)).fetchone()
    return _row_to_profile(row) if row else None


def search_profiles(name: str, db_path: str = DB_PATH) -> List[CharProfile]:
    """Return profiles whose name contains `name` (case-insensitive)."""
    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM profiles WHERE name LIKE ? COLLATE NOCASE ORDER BY id",
            (f"%{name}%",),
        ).fetchall()
    return [_row_to_profile(r) for r in rows]


def list_profiles(db_path: str = DB_PATH) -> List[CharProfile]:
    """Return all profiles ordered by id."""
    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM profiles ORDER BY id").fetchall()
    return [_row_to_profile(r) for r in rows]


# ─────────────────────────────────────────────
# CLI helpers
# ─────────────────────────────────────────────
_COL = f"  {'ID':>4}  {'Name':<28}  {'Eff Aff%':>8}  {'Eff Crit%':>9}"
_DIV = "  " + "-" * 56


def _print_profile_row(p: CharProfile) -> None:
    eff_aff  = min(p.affinity_rate, 0.40) + min(p.direct_affinity_rate, 0.10)
    eff_crit = min(p.critical_rate, 0.80) + min(p.direct_critical_rate, 0.20)
    print(f"  {p.id:>4}  {p.name:<28}  {eff_aff*100:>7.1f}%  {eff_crit*100:>8.1f}%")


def _print_profile_detail(p: CharProfile) -> None:
    def pct(v: float) -> str:
        return f"{v*100:.1f}%"

    print(f"  ID           : {p.id}")
    print(f"  Name         : {p.name}")
    print()

    # Physical
    print("  [Physical]")
    print(f"    min={p.physical_min}  max={p.physical_max}"
          f"  pen={p.physical_pen}  dmg_bonus={pct(p.physical_dmg_bonus)}")
    print()

    # Attributes
    print("  [Attributes]")
    for attr in ATTRIBUTES:
        amin = getattr(p, f"{attr}_min")
        amax = getattr(p, f"{attr}_max")
        apen = getattr(p, f"{attr}_pen")
        admb = getattr(p, f"{attr}_dmg_bonus")
        print(f"    {attr:<12}: {amin} - {amax}  pen={apen}  dmg_bonus={pct(admb)}")
    print()

    # Combat rates
    eff_aff  = min(p.affinity_rate, 0.40) + min(p.direct_affinity_rate, 0.10)
    eff_crit = min(p.critical_rate, 0.80) + min(p.direct_critical_rate, 0.20)
    print("  [Combat rates]")
    print(f"    affinity  : {pct(p.affinity_rate)} + {pct(p.direct_affinity_rate)} direct"
          f"  => eff {pct(eff_aff)}  (mult x{p.affinity_mult})")
    print(f"    precision : {pct(p.precision_rate)}")
    print(f"    crit      : {pct(p.critical_rate)} + {pct(p.direct_critical_rate)} direct"
          f"  => eff {pct(eff_crit)}  (mult x{p.critical_mult})")
    print()

    # DMG bonus modifiers
    print("  [DMG bonus modifiers]")
    print(f"    all martial art     : {pct(p.all_martial_art_dmg_bonus)}")
    print()
    print("    Weapon type:")
    for wt in WEAPON_TYPES:
        val = getattr(p, f"{wt}_dmg_bonus")
        indicator = pct(val) if val != 0.0 else "—"
        print(f"      {wt:<22}: {indicator}")
    print()
    print("    Mystic type:")
    for mt in MYSTIC_TYPES:
        val = getattr(p, f"{mt}_dmg_bonus")
        indicator = pct(val) if val != 0.0 else "—"
        print(f"      {mt:<34}: {indicator}")
    print()
    print("    Target DMG bonus:")
    print(f"      {'pvp':<34}: {pct(p.pvp_dmg_bonus)}")
    print(f"      {'boss':<34}: {pct(p.boss_dmg_bonus)}")


def _prompt(label: str, default=None) -> str:
    """Show prompt with default value hint; return stripped input or default."""
    hint = f" [{default}]" if default is not None else ""
    val  = input(f"  {label}{hint}: ").strip()
    return val if val else (str(default) if default is not None else "")


def _pf(label: str, default: float) -> float:
    """Prompt for a float value, returning default on empty input or parse error."""
    raw = _prompt(label, default)
    try:
        return float(raw)
    except (ValueError, TypeError):
        print(f"  Invalid float, using {default}")
        return default


def _select_from(label: str, options: List[str], default: str) -> str:
    """Show numbered list and return the selected option string."""
    print(f"\n  {label}:")
    for i, opt in enumerate(options, 1):
        marker = "  ← current" if opt == default else ""
        print(f"    [{i}] {opt}{marker}")
    try:
        raw = input(f"  Choice [1-{len(options)}] (Enter=keep current): ").strip()
    except (KeyboardInterrupt, EOFError):
        raise
    if not raw:
        return default
    try:
        idx = int(raw)
        if 1 <= idx <= len(options):
            return options[idx - 1]
    except ValueError:
        pass
    print(f"  Invalid, keeping '{default}'.")
    return default


def _wizard(base: Optional[CharProfile] = None) -> Optional[CharProfile]:
    """Interactive field-by-field wizard. Returns a CharProfile or None on cancel."""
    b   = base or CharProfile()
    SEP = "  " + "-" * 52
    try:
        # ── Profile info ──────────────────────────────
        print(f"\n{SEP}\n  Profile info\n{SEP}")
        name = _prompt("name", b.name)
        if not name:
            print("  Name cannot be empty."); return None

        # ── Physical stats ─────────────────────────────
        print(f"\n{SEP}\n  Physical stats\n{SEP}")
        physical_min       = _pf("physical_min",       b.physical_min)
        physical_max       = _pf("physical_max",       b.physical_max)
        physical_pen       = _pf("physical_pen",       b.physical_pen)
        physical_dmg_bonus = _pf("physical_dmg_bonus", b.physical_dmg_bonus)

        # ── Attribute stats ────────────────────────────
        attr_vals: dict = {}
        for attr in ATTRIBUTES:
            print(f"\n{SEP}\n  Attribute: {attr}\n{SEP}")
            attr_vals[f"{attr}_min"]       = _pf(f"{attr}_min",       getattr(b, f"{attr}_min"))
            attr_vals[f"{attr}_max"]       = _pf(f"{attr}_max",       getattr(b, f"{attr}_max"))
            attr_vals[f"{attr}_pen"]       = _pf(f"{attr}_pen",       getattr(b, f"{attr}_pen"))
            attr_vals[f"{attr}_dmg_bonus"] = _pf(f"{attr}_dmg_bonus", getattr(b, f"{attr}_dmg_bonus"))

        # ── Combat rates ───────────────────────────────
        print(f"\n{SEP}\n  Combat rates  (decimal: 0.374 = 37.4%)\n{SEP}")
        affinity_rate        = _pf("affinity_rate",        b.affinity_rate)
        direct_affinity_rate = _pf("direct_affinity_rate", b.direct_affinity_rate)
        precision_rate       = _pf("precision_rate",       b.precision_rate)
        critical_rate        = _pf("critical_rate",        b.critical_rate)
        direct_critical_rate = _pf("direct_critical_rate", b.direct_critical_rate)
        affinity_mult        = _pf("affinity_mult",        b.affinity_mult)
        critical_mult        = _pf("critical_mult",        b.critical_mult)

        # ── DMG bonus modifiers ────────────────────────
        print(f"\n{SEP}\n  DMG bonus modifiers  (decimal: 0.10 = 10%)\n{SEP}")
        all_martial_art_dmg_bonus = _pf("all_martial_art_dmg_bonus",
                                        b.all_martial_art_dmg_bonus)

        print("\n  Weapon-type DMG bonus:")
        weapon_vals: dict = {}
        for wt in WEAPON_TYPES:
            key = f"{wt}_dmg_bonus"
            weapon_vals[key] = _pf(f"  {wt}", getattr(b, key))

        print("\n  Mystic-type DMG bonus:")
        mystic_vals: dict = {}
        for mt in MYSTIC_TYPES:
            key = f"{mt}_dmg_bonus"
            mystic_vals[key] = _pf(f"  {mt}", getattr(b, key))

        print("\n  Target DMG bonus:")
        pvp_dmg_bonus  = _pf("  pvp",  b.pvp_dmg_bonus)
        boss_dmg_bonus = _pf("  boss", b.boss_dmg_bonus)

        return CharProfile(
            id   = b.id,
            name = name,
            physical_min       = physical_min,
            physical_max       = physical_max,
            physical_pen       = physical_pen,
            physical_dmg_bonus = physical_dmg_bonus,
            **attr_vals,
            affinity_rate        = affinity_rate,
            precision_rate       = precision_rate,
            critical_rate        = critical_rate,
            affinity_mult        = affinity_mult,
            critical_mult        = critical_mult,
            direct_affinity_rate = direct_affinity_rate,
            direct_critical_rate = direct_critical_rate,
            all_martial_art_dmg_bonus = all_martial_art_dmg_bonus,
            **weapon_vals,
            **mystic_vals,
            pvp_dmg_bonus  = pvp_dmg_bonus,
            boss_dmg_bonus = boss_dmg_bonus,
        )

    except (KeyboardInterrupt, EOFError):
        print("\n  Cancelled.")
        return None
    except ValueError as e:
        print(f"  Invalid input: {e}")
        return None


def _pick_from_list(profiles: List[CharProfile]) -> Optional[CharProfile]:
    """Print a table of profiles and let user pick one by ID."""
    print(_COL); print(_DIV)
    for p in profiles:
        _print_profile_row(p)
    print()
    try:
        raw = input("  Select ID (Enter to cancel): ").strip()
    except (KeyboardInterrupt, EOFError):
        return None
    if not raw:
        return None
    try:
        chosen_id = int(raw)
    except ValueError:
        print("  Not a valid ID."); return None
    match = next((p for p in profiles if p.id == chosen_id), None)
    if not match:
        print(f"  ID {chosen_id} not in list."); return None
    return match


def _interact(p: CharProfile, db_path: str) -> None:
    """Show action menu for a selected profile."""
    print()
    _print_profile_detail(p)
    print()
    print("  [1] Edit")
    print("  [2] Remove")
    print("  [3] Cancel")
    print()
    try:
        choice = input("  Choice: ").strip()
    except (KeyboardInterrupt, EOFError):
        return

    if choice == "1":
        updated = _wizard(base=p)
        if updated:
            update_profile(updated, db_path)
            print(f"\n  Updated profile ID={updated.id}.")
    elif choice == "2":
        try:
            confirm = input(f"  Remove '{p.name}' (ID={p.id})? [y/N]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return
        if confirm in ("y", "yes"):
            remove_profile(p.id, db_path)
            print(f"  Removed profile ID={p.id}.")
        else:
            print("  Cancelled.")


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────
def _cli(argv: List[str], db_path: str = DB_PATH) -> None:
    SEP = "=" * 72

    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__); return

    cmd  = argv[0]
    rest = argv[1:]

    # ── init ──────────────────────────────────
    if cmd == "init":
        init_db(db_path)
        print(f"  DB initialised: {db_path}")
        return

    # ── list ──────────────────────────────────
    if cmd == "list":
        profiles = list_profiles(db_path)
        if not profiles:
            print("  (no profiles found)"); return
        print(_COL); print(_DIV)
        for p in profiles:
            _print_profile_row(p)
        return

    # ── show ──────────────────────────────────
    if cmd == "show":
        if not rest:
            print("  Usage: show <id>"); return
        p = get_profile_by_id(int(rest[0]), db_path)
        if not p:
            print(f"  Profile ID={rest[0]} not found."); return
        print(SEP); _print_profile_detail(p); print(SEP)
        return

    # ── search ────────────────────────────────
    if cmd == "search":
        if not rest:
            print("  Usage: search <name>"); return
        profiles = search_profiles(" ".join(rest), db_path)
        if not profiles:
            print("  No profiles found."); return
        p = _pick_from_list(profiles)
        if p:
            _interact(p, db_path)
        return

    # ── add ───────────────────────────────────
    if cmd == "add":
        print(f"\n{SEP}\n  Add new profile\n{SEP}")
        print("  (Enter to keep default, Ctrl-C to cancel)")
        p = _wizard()
        if p:
            new_id = add_profile(p, db_path)
            print(f"\n  Added profile '{p.name}' with ID={new_id}.")
        return

    # ── edit ──────────────────────────────────
    if cmd == "edit":
        if not rest:
            print("  Usage: edit <id>"); return
        p = get_profile_by_id(int(rest[0]), db_path)
        if not p:
            print(f"  Profile ID={rest[0]} not found."); return
        print(f"\n{SEP}\n  Edit profile ID={p.id}: {p.name}\n{SEP}")
        print("  (Enter to keep current value, Ctrl-C to cancel)")
        updated = _wizard(base=p)
        if updated:
            update_profile(updated, db_path)
            print(f"\n  Updated profile ID={updated.id}.")
        return

    # ── duplicate ─────────────────────────────
    if cmd == "duplicate":
        if not rest:
            print("  Usage: duplicate <id>"); return
        src = get_profile_by_id(int(rest[0]), db_path)
        if not src:
            print(f"  Profile ID={rest[0]} not found."); return
        # Show the source profile
        print(f"\n{SEP}\n  Duplicate profile ID={src.id}: {src.name}\n{SEP}")
        _print_profile_detail(src)
        print(f"\n{SEP}")
        # Ask whether to edit before saving
        try:
            ans = input("  Edit fields before saving? [y/N]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n  Cancelled."); return
        # Build the copy (id=None so add_profile assigns a new one)
        copy = dc_replace(src, id=None)
        if ans in ("y", "yes"):
            print(f"\n{SEP}\n  Edit duplicate (Enter=keep source value, Ctrl-C to cancel)\n{SEP}")
            copy = _wizard(base=copy)
            if copy is None:
                return
        new_id = add_profile(copy, db_path)
        print(f"\n  Duplicated '{copy.name}' → new profile ID={new_id}.")
        return

    # ── remove ────────────────────────────────
    if cmd == "remove":
        if not rest:
            print("  Usage: remove <id>"); return
        profile_id = int(rest[0])
        p = get_profile_by_id(profile_id, db_path)
        if not p:
            print(f"  Profile ID={profile_id} not found."); return
        try:
            confirm = input(f"  Remove '{p.name}' (ID={profile_id})? [y/N]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return
        if confirm in ("y", "yes"):
            remove_profile(profile_id, db_path)
            print(f"  Removed profile ID={profile_id}.")
        else:
            print("  Cancelled.")
        return

    print(f"  Unknown command '{cmd}'. Run with --help for usage.")


if __name__ == "__main__":
    _cli(sys.argv[1:])
