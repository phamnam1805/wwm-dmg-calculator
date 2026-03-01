#!/usr/bin/env python3
"""
skill_db.py — Skill formula database (SQLite)
==============================================
Library API  : import and call functions directly (usable from Web UI, CLI, etc.)
Interactive CLI: python src/skill_db.py [command] [args]

Commands:
  list                    list all skills
  list --type <type>      filter by skill_type  (martial_art | mystic)
  list --weapon <weapon>  filter by weapon_type
  search <name>           search by name (fuzzy), then pick ID to interact
  show   <id>             show skill detail
  add                     interactive wizard to add a new skill
  edit   <id>             interactive wizard to edit an existing skill
  remove <id>             delete skill by ID
  init                    initialise DB (runs automatically on first use)
"""

import sqlite3
import sys
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DB_PATH = str(Path(__file__).parent.parent / "dbs" / "skills.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    phys_coeff  REAL NOT NULL DEFAULT 1.0,
    attr_coeff  REAL NOT NULL DEFAULT 1.0,
    phys_bonus  REAL NOT NULL DEFAULT 0.0,
    attr_bonus  REAL NOT NULL DEFAULT 0.0,
    skill_type  TEXT NOT NULL DEFAULT 'martial_art'
                CHECK(skill_type IN ('martial_art', 'mystic')),
    is_dot      INTEGER NOT NULL DEFAULT 0,
    weapon_type TEXT,
    mystic_type TEXT,   -- NULL | area_debuff | area_dmg | single_target_control | single_target_burst
    CHECK(skill_type != 'mystic' OR weapon_type IS NULL)
);
"""


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────
class SkillType(str, Enum):
    MARTIAL_ART = "martial_art"
    MYSTIC      = "mystic"


class WeaponType(str, Enum):
    """Known weapon types. Extend freely — no DB schema change needed."""
    SWORD       = "sword"
    DUAL_BLADES = "dual_blades"
    SPEAR       = "spear"
    FAN         = "fan"
    UMBRELLA    = "umbrella"
    HENG_BLADE  = "heng_blade"
    MO_BLADE    = "mo_blade"
    ROPE_DART   = "rope_dart"


class MysticType(str, Enum):
    """Mystic skill sub-types, used for targeted DMG bonus in character profiles."""
    AREA_DEBUFF           = "area_debuff"
    AREA_DMG              = "area_dmg"
    SINGLE_TARGET_CONTROL = "single_target_control"
    SINGLE_TARGET_BURST   = "single_target_burst"

_MYSTIC_TYPE_VALUES = [t.value for t in MysticType]


# ─────────────────────────────────────────────
# SkillFormula dataclass
# ─────────────────────────────────────────────
@dataclass
class SkillFormula:
    name:        str           = "default"
    phys_coeff:  float         = 1.0
    attr_coeff:  float         = 1.0
    phys_bonus:  float         = 0.0   # flat bonus, added after phys_coeff × atk
    attr_bonus:  float         = 0.0   # atk bonus, added to attr_atk before coeff
    skill_type:  str           = SkillType.MARTIAL_ART
    is_dot:      bool          = False
    weapon_type: Optional[str] = None
    mystic_type: Optional[str] = None  # only for mystic skills; see MysticType enum
    id:          Optional[int] = None  # None if not yet saved to DB

    @property
    def is_mystic(self) -> bool:
        """Backward-compatible property — True when skill_type == 'mystic'."""
        return self.skill_type == SkillType.MYSTIC

    def __post_init__(self):
        if self.is_mystic:
            self.attr_bonus  = 0.0
            self.attr_coeff  = self.phys_coeff
            self.weapon_type = None            # mystic skills are not weapon-specific
        else:
            expected = self.phys_coeff * 1.5
            if abs(self.attr_coeff - expected) > 0.01 * expected:
                self.attr_coeff = expected  # enforce x1.5 for non-mystic main attr
            self.mystic_type = None           # not applicable for non-mystic skills


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────
def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_formula(row: sqlite3.Row) -> SkillFormula:
    return SkillFormula(
        id          = row["id"],
        name        = row["name"],
        phys_coeff  = row["phys_coeff"],
        attr_coeff  = row["attr_coeff"],
        phys_bonus  = row["phys_bonus"],
        attr_bonus  = row["attr_bonus"],
        skill_type  = row["skill_type"],
        is_dot      = bool(row["is_dot"]),
        weapon_type = row["weapon_type"],
        mystic_type = row["mystic_type"],
    )


# ─────────────────────────────────────────────
# Library API  (no print statements — safe to import from Web UI)
# ─────────────────────────────────────────────
def init_db(db_path: str = DB_PATH) -> None:
    """Create the skills table if it does not exist. Safe to call multiple times."""
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA)
        # Auto-migrate: add mystic_type column to existing DBs
        existing = {r[1] for r in conn.execute("PRAGMA table_info(skills)").fetchall()}
        if "mystic_type" not in existing:
            conn.execute("ALTER TABLE skills ADD COLUMN mystic_type TEXT")


def add_skill(formula: SkillFormula, db_path: str = DB_PATH) -> int:
    """Insert a new skill. Returns the auto-assigned ID."""
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO skills "
            "(name, phys_coeff, attr_coeff, phys_bonus, attr_bonus, skill_type, is_dot, weapon_type, mystic_type) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (formula.name, formula.phys_coeff, formula.attr_coeff,
             formula.phys_bonus, formula.attr_bonus,
             formula.skill_type, int(formula.is_dot), formula.weapon_type, formula.mystic_type),
        )
        return cur.lastrowid


def update_skill(formula: SkillFormula, db_path: str = DB_PATH) -> None:
    """Update an existing skill by formula.id. Raises ValueError if id is None."""
    if formula.id is None:
        raise ValueError("Cannot update a skill without an ID.")
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE skills SET name=?, phys_coeff=?, attr_coeff=?, phys_bonus=?, "
            "attr_bonus=?, skill_type=?, is_dot=?, weapon_type=?, mystic_type=? WHERE id=?",
            (formula.name, formula.phys_coeff, formula.attr_coeff,
             formula.phys_bonus, formula.attr_bonus,
             formula.skill_type, int(formula.is_dot), formula.weapon_type, formula.mystic_type,
             formula.id),
        )


def remove_skill(skill_id: int, db_path: str = DB_PATH) -> None:
    """Delete a skill by ID. Silent if the ID does not exist."""
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM skills WHERE id=?", (skill_id,))


def get_skill_by_id(skill_id: int, db_path: str = DB_PATH) -> Optional[SkillFormula]:
    """Return a SkillFormula by ID, or None if not found."""
    init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM skills WHERE id=?", (skill_id,)).fetchone()
    return _row_to_formula(row) if row else None


def search_skills(name: str, db_path: str = DB_PATH) -> List[SkillFormula]:
    """Return skills whose name contains `name` (case-insensitive)."""
    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM skills WHERE name LIKE ? COLLATE NOCASE ORDER BY id",
            (f"%{name}%",),
        ).fetchall()
    return [_row_to_formula(r) for r in rows]


def list_skills(
    db_path: str = DB_PATH,
    skill_type:  Optional[str] = None,
    weapon_type: Optional[str] = None,
) -> List[SkillFormula]:
    """Return all skills, optionally filtered by skill_type and/or weapon_type."""
    init_db(db_path)
    conditions, params = [], []
    if skill_type:
        conditions.append("skill_type = ?")
        params.append(skill_type)
    if weapon_type:
        conditions.append("weapon_type = ?")
        params.append(weapon_type)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT * FROM skills {where} ORDER BY id", params
        ).fetchall()
    return [_row_to_formula(r) for r in rows]


# ─────────────────────────────────────────────
# CLI helpers
# ─────────────────────────────────────────────
_COL = f"  {'ID':>4}  {'Name':<30}  {'Type':<12}  {'Weapon':<12}  DOT"
_DIV = "  " + "-" * 68


def _print_skill_row(f: SkillFormula) -> None:
    dot = "dot" if f.is_dot else ""
    wep = f.weapon_type or ""
    print(f"  {f.id:>4}  {f.name:<30}  {f.skill_type:<12}  {wep:<12}  {dot}")


def _print_skill_detail(f: SkillFormula) -> None:
    print(f"  ID          : {f.id}")
    print(f"  Name        : {f.name}")
    print(f"  skill_type  : {f.skill_type}")
    print(f"  mystic_type : {f.mystic_type or '(none)'}")
    print(f"  weapon_type : {f.weapon_type or '(none)'}")
    print(f"  is_dot      : {f.is_dot}")
    print(f"  phys_coeff  : {f.phys_coeff}")
    print(f"  attr_coeff  : {f.attr_coeff}  (enforced to phys×1.5 for non-mystic)")
    print(f"  phys_bonus  : {f.phys_bonus}")
    print(f"  attr_bonus  : {f.attr_bonus}")


def _prompt(label: str, default=None) -> str:
    """Show prompt with default value hint; return stripped input or default."""
    hint = f" [{default}]" if default is not None else ""
    val  = input(f"  {label}{hint}: ").strip()
    return val if val else (str(default) if default is not None else "")


def _select_option(label: str, options: List[str], default: Optional[str] = None) -> Optional[str]:
    """Show a numbered selection menu. Returns the chosen option, or None for '(none)'."""
    print(f"\n  {label}:")
    marker0 = "  ← current" if default is None else ""
    print(f"    [0] (none){marker0}")
    for i, opt in enumerate(options, 1):
        marker = "  ← current" if opt == default else ""
        print(f"    [{i}] {opt}{marker}")
    try:
        raw = input(f"  Choice [0-{len(options)}] (Enter=keep current): ").strip()
    except (KeyboardInterrupt, EOFError):
        raise
    if not raw:
        return default
    try:
        idx = int(raw)
    except ValueError:
        print("  Invalid, keeping current.")
        return default
    if idx == 0:
        return None
    if 1 <= idx <= len(options):
        return options[idx - 1]
    print("  Out of range, keeping current.")
    return default


def _wizard(base: Optional[SkillFormula] = None) -> Optional[SkillFormula]:
    """Interactive field-by-field wizard. Returns a SkillFormula or None on cancel."""
    print("  (Enter to keep current value, Ctrl-C to cancel)\n")
    try:
        name       = _prompt("name",        base.name       if base else "")
        if not name:
            print("  Name cannot be empty."); return None
        phys_coeff = float(_prompt("phys_coeff",  base.phys_coeff if base else 1.0))
        phys_bonus = float(_prompt("phys_bonus",  base.phys_bonus if base else 0.0))
        attr_bonus = float(_prompt("attr_bonus",  base.attr_bonus if base else 0.0))

        stype_raw  = _prompt("skill_type (martial_art/mystic)",
                             base.skill_type if base else SkillType.MARTIAL_ART)
        if stype_raw not in (SkillType.MARTIAL_ART, SkillType.MYSTIC):
            print(f"  Invalid skill_type '{stype_raw}'. Use 'martial_art' or 'mystic'.")
            return None

        # mystic_type — selection menu, only shown for mystic skills
        mystic_type = base.mystic_type if base else None
        if stype_raw == SkillType.MYSTIC:
            mystic_type = _select_option(
                "mystic_type", _MYSTIC_TYPE_VALUES,
                default=base.mystic_type if base else None,
            )

        is_dot_raw = _prompt("is_dot (y/n)", "y" if (base and base.is_dot) else "n")
        is_dot     = is_dot_raw.lower() in ("y", "yes", "1", "true")

        weapon     = _prompt("weapon_type (sword/dual_blades/spear/umbrella/fan/heng_blade/mo_blade/rope_dart or blank)",
                             base.weapon_type if base else "")
        weapon     = weapon if weapon else None

        return SkillFormula(
            id          = base.id if base else None,
            name        = name,
            phys_coeff  = phys_coeff,
            attr_coeff  = 1.0,       # __post_init__ will enforce correct value
            phys_bonus  = phys_bonus,
            attr_bonus  = attr_bonus,
            skill_type  = stype_raw,
            is_dot      = is_dot,
            weapon_type = weapon,
            mystic_type = mystic_type,
        )
    except (KeyboardInterrupt, EOFError):
        print("\n  Cancelled.")
        return None
    except ValueError as e:
        print(f"  Invalid input: {e}")
        return None


def _pick_from_list(skills: List[SkillFormula]) -> Optional[SkillFormula]:
    """Print a table of skills and let user pick one by ID. Returns None on cancel."""
    print(_COL)
    print(_DIV)
    for f in skills:
        _print_skill_row(f)
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
    match = next((f for f in skills if f.id == chosen_id), None)
    if not match:
        print(f"  ID {chosen_id} not in list."); return None
    return match


def _interact(f: SkillFormula, db_path: str) -> None:
    """Show action menu for a selected skill."""
    print()
    _print_skill_detail(f)
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
        updated = _wizard(base=f)
        if updated:
            update_skill(updated, db_path)
            print(f"\n  Updated skill ID={updated.id}.")
    elif choice == "2":
        try:
            confirm = input(f"  Remove '{f.name}' (ID={f.id})? [y/N]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return
        if confirm in ("y", "yes"):
            remove_skill(f.id, db_path)
            print(f"  Removed skill ID={f.id}.")
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
        stype  = None
        weapon = None
        i = 0
        while i < len(rest):
            if rest[i] == "--type"   and i + 1 < len(rest): stype  = rest[i+1]; i += 2
            elif rest[i] == "--weapon" and i + 1 < len(rest): weapon = rest[i+1]; i += 2
            else: i += 1
        skills = list_skills(db_path, skill_type=stype, weapon_type=weapon)
        if not skills:
            print("  (no skills found)"); return
        print(_COL); print(_DIV)
        for f in skills: _print_skill_row(f)
        return

    # ── show ──────────────────────────────────
    if cmd == "show":
        if not rest:
            print("  Usage: show <id>"); return
        f = get_skill_by_id(int(rest[0]), db_path)
        if not f:
            print(f"  Skill ID={rest[0]} not found."); return
        print(SEP); _print_skill_detail(f); print(SEP)
        return

    # ── search ────────────────────────────────
    if cmd == "search":
        if not rest:
            print("  Usage: search <name>"); return
        skills = search_skills(" ".join(rest), db_path)
        if not skills:
            print("  No skills found."); return
        f = _pick_from_list(skills)
        if f:
            _interact(f, db_path)
        return

    # ── add ───────────────────────────────────
    if cmd == "add":
        print(f"\n{SEP}\n  Add new skill\n{SEP}")
        f = _wizard()
        if f:
            new_id = add_skill(f, db_path)
            print(f"\n  Added skill '{f.name}' with ID={new_id}.")
        return

    # ── edit ──────────────────────────────────
    if cmd == "edit":
        if not rest:
            print("  Usage: edit <id>"); return
        f = get_skill_by_id(int(rest[0]), db_path)
        if not f:
            print(f"  Skill ID={rest[0]} not found."); return
        print(f"\n{SEP}\n  Edit skill ID={f.id}: {f.name}\n{SEP}")
        updated = _wizard(base=f)
        if updated:
            update_skill(updated, db_path)
            print(f"\n  Updated skill ID={updated.id}.")
        return

    # ── remove ────────────────────────────────
    if cmd == "remove":
        if not rest:
            print("  Usage: remove <id>"); return
        skill_id = int(rest[0])
        f = get_skill_by_id(skill_id, db_path)
        if not f:
            print(f"  Skill ID={skill_id} not found."); return
        try:
            confirm = input(f"  Remove '{f.name}' (ID={skill_id})? [y/N]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return
        if confirm in ("y", "yes"):
            remove_skill(skill_id, db_path)
            print(f"  Removed skill ID={skill_id}.")
        else:
            print("  Cancelled.")
        return

    print(f"  Unknown command '{cmd}'. Run with --help for usage.")


if __name__ == "__main__":
    _cli(sys.argv[1:])
