#!/usr/bin/env python3
"""
skill_db.py — Skill formula database (SQLite)
==============================================
Two separate databases, one file:
  dbs/martial_art_skills.db  — martial art skills
  dbs/mystic_skills.db       — mystic skills

Library API  : import and call functions directly (usable from Web UI, CLI, etc.)
Interactive CLI: python src/skill_db.py [subcommand] [command] [args]

Subcommand-scoped (work on one DB):
  martial list                   list all martial art skills
  martial list --weapon <type>   filter by weapon type
  martial show   <id>            show detail
  martial add                    wizard: add new martial art skill
  martial edit   <id>            wizard: edit existing skill
  martial remove <id>            delete skill
  martial search <name>          search by name (pick to edit/remove)
  martial init                   initialise DB

  mystic  list                   list all mystic skills
  mystic  show   <id>            show detail
  mystic  add                    wizard: add new mystic skill
  mystic  edit   <id>            wizard: edit existing skill
  mystic  remove <id>            delete skill
  mystic  search <name>          search by name (pick to edit/remove)
  mystic  init                   initialise DB

Global commands (work across both DBs):
  list                           list ALL skills (both DBs)
  search <name>                  search across both DBs
  init                           initialise both DBs
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
_DBDIR = Path(__file__).parent.parent / "dbs"

MARTIAL_ART_DB_PATH = str(_DBDIR / "martial_art_skills.db")
MYSTIC_DB_PATH      = str(_DBDIR / "mystic_skills.db")

_MARTIAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    name           TEXT    NOT NULL,
    phys_coeff     REAL    NOT NULL DEFAULT 1.0,
    phys_bonus     REAL    NOT NULL DEFAULT 0.0,
    attr_bonus     REAL    NOT NULL DEFAULT 0.0,
    weapon_type    TEXT,
    attribute_type TEXT,   -- bellstrike | stonesplit | bamboocut | silkbind
    is_dot         INTEGER NOT NULL DEFAULT 0
);
"""

_MYSTIC_SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    phys_coeff  REAL    NOT NULL DEFAULT 1.0,
    phys_bonus  REAL    NOT NULL DEFAULT 0.0,
    mystic_type TEXT,
    is_dot      INTEGER NOT NULL DEFAULT 0
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

_WEAPON_TYPE_VALUES = [t.value for t in WeaponType]


class AttributeType(str, Enum):
    """Martial art attribute types — which attribute pool the skill draws from."""
    BELLSTRIKE = "bellstrike"
    STONESPLIT = "stonesplit"
    BAMBOOCUT  = "bamboocut"
    SILKBIND   = "silkbind"

_ATTRIBUTE_TYPE_VALUES = [t.value for t in AttributeType]


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
    attr_coeff:  float         = 1.0   # derived in __post_init__, NOT stored in DB
    phys_bonus:  float         = 0.0   # flat bonus added after phys_coeff × atk
    attr_bonus:  float         = 0.0   # atk bonus added to main attr before coeff
    skill_type:  str           = SkillType.MARTIAL_ART
    is_dot:      bool          = False
    weapon_type:    Optional[str] = None  # martial art only; see WeaponType enum
    attribute_type: Optional[str] = None  # martial art only; see AttributeType enum
    mystic_type:    Optional[str] = None  # mystic only; see MysticType enum
    id:          Optional[int] = None  # None if not yet saved to DB

    @property
    def is_mystic(self) -> bool:
        return self.skill_type == SkillType.MYSTIC

    def __post_init__(self):
        # Normalize skill_type to plain str value (str, Enum format changed in Python 3.11+)
        if isinstance(self.skill_type, Enum):
            self.skill_type = self.skill_type.value
        if self.is_mystic:
            self.attr_coeff     = self.phys_coeff  # mystic: same coeff for all attrs
            self.attr_bonus     = 0.0              # no attr bonus for mystic
            self.weapon_type    = None             # mystic skills not weapon-specific
            self.attribute_type = None             # not applicable to mystic
        else:
            self.mystic_type = None  # not applicable to martial art
            if self.is_dot:
                # DOT: no phys_bonus, no attr_bonus, no x1.5 scaling on main attribute
                self.attr_coeff  = self.phys_coeff
                self.phys_bonus  = 0.0
                self.attr_bonus  = 0.0
            else:
                self.attr_coeff  = self.phys_coeff * 1.5


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────
def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _martial_row_to_formula(row: sqlite3.Row) -> SkillFormula:
    return SkillFormula(
        id             = row["id"],
        name           = row["name"],
        phys_coeff     = row["phys_coeff"],
        phys_bonus     = row["phys_bonus"],
        attr_bonus     = row["attr_bonus"],
        skill_type     = SkillType.MARTIAL_ART,
        is_dot         = bool(row["is_dot"]),
        weapon_type    = row["weapon_type"],
        attribute_type = row["attribute_type"],
        # attr_coeff derived in __post_init__; mystic_type = None via __post_init__
    )


def _mystic_row_to_formula(row: sqlite3.Row) -> SkillFormula:
    return SkillFormula(
        id          = row["id"],
        name        = row["name"],
        phys_coeff  = row["phys_coeff"],
        phys_bonus  = row["phys_bonus"],
        skill_type  = SkillType.MYSTIC,
        is_dot      = bool(row["is_dot"]),
        mystic_type = row["mystic_type"],
        # attr_coeff derived in __post_init__; attr_bonus=0, weapon_type=None via __post_init__
    )


# ─────────────────────────────────────────────
# Library API  (no print statements — safe to import from Web UI)
# ─────────────────────────────────────────────
def init_martial_db(db_path: str = MARTIAL_ART_DB_PATH) -> None:
    """Create martial art skills table if it does not exist. Safe to call multiple times."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.executescript(_MARTIAL_SCHEMA)
        # Auto-migrate: add attribute_type column to existing DBs
        existing = {r[1] for r in conn.execute("PRAGMA table_info(skills)").fetchall()}
        if "attribute_type" not in existing:
            conn.execute("ALTER TABLE skills ADD COLUMN attribute_type TEXT")


def init_mystic_db(db_path: str = MYSTIC_DB_PATH) -> None:
    """Create mystic skills table if it does not exist. Safe to call multiple times."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.executescript(_MYSTIC_SCHEMA)


def init_all(
    martial_db: str = MARTIAL_ART_DB_PATH,
    mystic_db:  str = MYSTIC_DB_PATH,
) -> None:
    """Initialise both databases. Safe to call multiple times."""
    init_martial_db(martial_db)
    init_mystic_db(mystic_db)


def add_skill(
    formula:    SkillFormula,
    martial_db: str = MARTIAL_ART_DB_PATH,
    mystic_db:  str = MYSTIC_DB_PATH,
) -> int:
    """Insert a new skill into the appropriate DB. Returns the auto-assigned ID."""
    if formula.is_mystic:
        init_mystic_db(mystic_db)
        with _connect(mystic_db) as conn:
            cur = conn.execute(
                "INSERT INTO skills (name, phys_coeff, phys_bonus, mystic_type, is_dot) "
                "VALUES (?, ?, ?, ?, ?)",
                (formula.name, formula.phys_coeff, formula.phys_bonus,
                 formula.mystic_type, int(formula.is_dot)),
            )
            return cur.lastrowid
    else:
        init_martial_db(martial_db)
        with _connect(martial_db) as conn:
            cur = conn.execute(
                "INSERT INTO skills "
                "(name, phys_coeff, phys_bonus, attr_bonus, weapon_type, attribute_type, is_dot) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (formula.name, formula.phys_coeff, formula.phys_bonus,
                 formula.attr_bonus, formula.weapon_type, formula.attribute_type,
                 int(formula.is_dot)),
            )
            return cur.lastrowid


def update_skill(
    formula:    SkillFormula,
    martial_db: str = MARTIAL_ART_DB_PATH,
    mystic_db:  str = MYSTIC_DB_PATH,
) -> None:
    """Update an existing skill by formula.id. Raises ValueError if id is None."""
    if formula.id is None:
        raise ValueError("Cannot update a skill without an ID.")
    if formula.is_mystic:
        with _connect(mystic_db) as conn:
            conn.execute(
                "UPDATE skills SET name=?, phys_coeff=?, phys_bonus=?, mystic_type=?, is_dot=? "
                "WHERE id=?",
                (formula.name, formula.phys_coeff, formula.phys_bonus,
                 formula.mystic_type, int(formula.is_dot), formula.id),
            )
    else:
        with _connect(martial_db) as conn:
            conn.execute(
                "UPDATE skills SET name=?, phys_coeff=?, phys_bonus=?, attr_bonus=?, "
                "weapon_type=?, attribute_type=?, is_dot=? WHERE id=?",
                (formula.name, formula.phys_coeff, formula.phys_bonus,
                 formula.attr_bonus, formula.weapon_type, formula.attribute_type,
                 int(formula.is_dot), formula.id),
            )


def remove_skill(
    skill_id:   int,
    skill_type: str,
    martial_db: str = MARTIAL_ART_DB_PATH,
    mystic_db:  str = MYSTIC_DB_PATH,
) -> None:
    """Delete a skill by ID and type. Silent if the ID does not exist."""
    db_path = mystic_db if skill_type == SkillType.MYSTIC else martial_db
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM skills WHERE id=?", (skill_id,))


def get_skill_by_id(
    skill_id:   int,
    skill_type: Optional[str] = None,
    martial_db: str = MARTIAL_ART_DB_PATH,
    mystic_db:  str = MYSTIC_DB_PATH,
) -> Optional[SkillFormula]:
    """
    Return a SkillFormula by ID, or None if not found.
    If skill_type is given, searches only that DB.
    If skill_type is None, searches both; raises ValueError if found in both
    (IDs are independent per DB — caller must specify skill_type to disambiguate).
    """
    if skill_type == SkillType.MYSTIC:
        init_mystic_db(mystic_db)
        with _connect(mystic_db) as conn:
            row = conn.execute("SELECT * FROM skills WHERE id=?", (skill_id,)).fetchone()
        return _mystic_row_to_formula(row) if row else None

    if skill_type == SkillType.MARTIAL_ART:
        init_martial_db(martial_db)
        with _connect(martial_db) as conn:
            row = conn.execute("SELECT * FROM skills WHERE id=?", (skill_id,)).fetchone()
        return _martial_row_to_formula(row) if row else None

    # Search both DBs
    init_all(martial_db, mystic_db)
    with _connect(martial_db) as conn:
        mr = conn.execute("SELECT * FROM skills WHERE id=?", (skill_id,)).fetchone()
    with _connect(mystic_db) as conn:
        yr = conn.execute("SELECT * FROM skills WHERE id=?", (skill_id,)).fetchone()

    if mr and yr:
        raise ValueError(
            f"ID {skill_id} exists in both martial_art_skills.db and mystic_skills.db. "
            f"Use 'martial show {skill_id}' or 'mystic show {skill_id}' to disambiguate."
        )
    if mr:
        return _martial_row_to_formula(mr)
    if yr:
        return _mystic_row_to_formula(yr)
    return None


def search_skills(
    name:       str,
    skill_type: Optional[str] = None,
    martial_db: str = MARTIAL_ART_DB_PATH,
    mystic_db:  str = MYSTIC_DB_PATH,
) -> List[SkillFormula]:
    """Return skills whose name contains `name` (case-insensitive), across one or both DBs."""
    results: List[SkillFormula] = []
    if skill_type != SkillType.MYSTIC:
        init_martial_db(martial_db)
        with _connect(martial_db) as conn:
            rows = conn.execute(
                "SELECT * FROM skills WHERE name LIKE ? COLLATE NOCASE ORDER BY id",
                (f"%{name}%",),
            ).fetchall()
        results.extend(_martial_row_to_formula(r) for r in rows)
    if skill_type != SkillType.MARTIAL_ART:
        init_mystic_db(mystic_db)
        with _connect(mystic_db) as conn:
            rows = conn.execute(
                "SELECT * FROM skills WHERE name LIKE ? COLLATE NOCASE ORDER BY id",
                (f"%{name}%",),
            ).fetchall()
        results.extend(_mystic_row_to_formula(r) for r in rows)
    return results


def list_skills(
    skill_type:  Optional[str] = None,
    weapon_type: Optional[str] = None,
    martial_db:  str = MARTIAL_ART_DB_PATH,
    mystic_db:   str = MYSTIC_DB_PATH,
) -> List[SkillFormula]:
    """Return all skills, optionally filtered by skill_type and/or weapon_type."""
    results: List[SkillFormula] = []
    if skill_type != SkillType.MYSTIC:
        init_martial_db(martial_db)
        conditions, params = [], []
        if weapon_type:
            conditions.append("weapon_type = ?")
            params.append(weapon_type)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        with _connect(martial_db) as conn:
            rows = conn.execute(
                f"SELECT * FROM skills {where} ORDER BY id", params
            ).fetchall()
        results.extend(_martial_row_to_formula(r) for r in rows)
    if skill_type != SkillType.MARTIAL_ART:
        init_mystic_db(mystic_db)
        with _connect(mystic_db) as conn:
            rows = conn.execute("SELECT * FROM skills ORDER BY id").fetchall()
        results.extend(_mystic_row_to_formula(r) for r in rows)
    return results


# ─────────────────────────────────────────────
# CLI display helpers
# ─────────────────────────────────────────────
_COL_TYPED  = f"  {'ID':>4}  {'Name':<30}  {'Weapon / MysticType':<24}  DOT"
_COL_GLOBAL = f"  {'ID':>4}  {'Name':<30}  {'Type':<12}  {'Weapon / MysticType':<24}  DOT"
_DIV_TYPED  = "  " + "-" * 66
_DIV_GLOBAL = "  " + "-" * 78


def _print_typed_row(f: SkillFormula) -> None:
    dot = "dot" if f.is_dot else ""
    sub = f.mystic_type or f.weapon_type or ""
    print(f"  {f.id:>4}  {f.name:<30}  {sub:<24}  {dot}")


def _print_global_row(f: SkillFormula) -> None:
    dot   = "dot" if f.is_dot else ""
    sub   = f.mystic_type or f.weapon_type or ""
    stype = "mystic" if f.is_mystic else "martial_art"
    print(f"  {f.id:>4}  {f.name:<30}  {stype:<12}  {sub:<24}  {dot}")


def _print_skill_detail(f: SkillFormula) -> None:
    print(f"  ID          : {f.id}")
    print(f"  Name        : {f.name}")
    print(f"  skill_type  : {f.skill_type}")
    if f.is_mystic:
        print(f"  mystic_type    : {f.mystic_type or '(none)'}")
        print(f"  phys_coeff     : {f.phys_coeff}")
        print(f"  attr_coeff     : {f.attr_coeff}  (= phys_coeff for mystic)")
        print(f"  phys_bonus     : {f.phys_bonus}")
    else:
        print(f"  weapon_type    : {f.weapon_type or '(none)'}")
        print(f"  attribute_type : {f.attribute_type or '(none)'}")
        print(f"  phys_coeff     : {f.phys_coeff}")
        print(f"  attr_coeff     : {f.attr_coeff}  (enforced to phys×1.5)")
        print(f"  phys_bonus     : {f.phys_bonus}")
        print(f"  attr_bonus     : {f.attr_bonus}")
    print(f"  is_dot         : {f.is_dot}")


# ─────────────────────────────────────────────
# CLI input helpers
# ─────────────────────────────────────────────
def _prompt(label: str, default=None) -> str:
    """Show prompt with default value hint; return stripped input or default."""
    hint = f" [{default}]" if default is not None else ""
    val  = input(f"  {label}{hint}: ").strip()
    return val if val else (str(default) if default is not None else "")


def _select_option(
    label:   str,
    options: List[str],
    default: Optional[str] = None,
) -> Optional[str]:
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


# ─────────────────────────────────────────────
# Typed wizards
# ─────────────────────────────────────────────
def _martial_wizard(base: Optional[SkillFormula] = None) -> Optional[SkillFormula]:
    """Interactive wizard for martial art skills. Returns SkillFormula or None on cancel."""
    print("  (Enter to keep current value, Ctrl-C to cancel)\n")
    try:
        name = _prompt("name", base.name if base else "")
        if not name:
            print("  Name cannot be empty."); return None

        phys_coeff = float(_prompt("phys_coeff", base.phys_coeff if base else 1.0))
        phys_bonus = float(_prompt("phys_bonus", base.phys_bonus if base else 0.0))
        attr_bonus = float(_prompt("attr_bonus (added to main_attr before coeff)",
                                   base.attr_bonus if base else 0.0))

        weapon_type = _select_option(
            "weapon_type", _WEAPON_TYPE_VALUES,
            default=base.weapon_type if base else None,
        )
        if weapon_type is None:
            print("  weapon_type is required for martial art skills.")
            return None

        attribute_type = _select_option(
            "attribute_type", _ATTRIBUTE_TYPE_VALUES,
            default=base.attribute_type if base else None,
        )
        if attribute_type is None:
            print("  attribute_type is required for martial art skills.")
            return None

        is_dot_raw = _prompt("is_dot (y/n)", "y" if (base and base.is_dot) else "n")
        is_dot     = is_dot_raw.lower() in ("y", "yes", "1", "true")

        return SkillFormula(
            id             = base.id if base else None,
            name           = name,
            phys_coeff     = phys_coeff,
            phys_bonus     = phys_bonus,
            attr_bonus     = attr_bonus,
            skill_type     = SkillType.MARTIAL_ART,
            is_dot         = is_dot,
            weapon_type    = weapon_type,
            attribute_type = attribute_type,
        )
    except (KeyboardInterrupt, EOFError):
        print("\n  Cancelled."); return None
    except ValueError as e:
        print(f"  Invalid input: {e}"); return None


def _mystic_wizard(base: Optional[SkillFormula] = None) -> Optional[SkillFormula]:
    """Interactive wizard for mystic skills. Returns SkillFormula or None on cancel."""
    print("  (Enter to keep current value, Ctrl-C to cancel)\n")
    try:
        name = _prompt("name", base.name if base else "")
        if not name:
            print("  Name cannot be empty."); return None

        phys_coeff = float(_prompt("phys_coeff", base.phys_coeff if base else 1.0))
        phys_bonus = float(_prompt("phys_bonus", base.phys_bonus if base else 0.0))

        mystic_type = _select_option(
            "mystic_type", _MYSTIC_TYPE_VALUES,
            default=base.mystic_type if base else None,
        )

        is_dot_raw = _prompt("is_dot (y/n)", "y" if (base and base.is_dot) else "n")
        is_dot     = is_dot_raw.lower() in ("y", "yes", "1", "true")

        return SkillFormula(
            id          = base.id if base else None,
            name        = name,
            phys_coeff  = phys_coeff,
            phys_bonus  = phys_bonus,
            skill_type  = SkillType.MYSTIC,
            is_dot      = is_dot,
            mystic_type = mystic_type,
        )
    except (KeyboardInterrupt, EOFError):
        print("\n  Cancelled."); return None
    except ValueError as e:
        print(f"  Invalid input: {e}"); return None


# ─────────────────────────────────────────────
# CLI interaction helpers
# ─────────────────────────────────────────────
def _pick_from_list(
    skills:      List[SkillFormula],
    global_view: bool = False,
) -> Optional[SkillFormula]:
    """Print skill table and let user pick one by ID. Returns None on cancel."""
    if global_view:
        print(_COL_GLOBAL); print(_DIV_GLOBAL)
        for f in skills: _print_global_row(f)
    else:
        print(_COL_TYPED); print(_DIV_TYPED)
        for f in skills: _print_typed_row(f)
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


def _interact_typed(f: SkillFormula, martial_db: str, mystic_db: str) -> None:
    """Show action menu (edit/remove) for a selected skill."""
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

    wizard = _mystic_wizard if f.is_mystic else _martial_wizard

    if choice == "1":
        updated = wizard(base=f)
        if updated:
            update_skill(updated, martial_db, mystic_db)
            print(f"\n  Updated skill ID={updated.id}.")
    elif choice == "2":
        try:
            confirm = input(f"  Remove '{f.name}' (ID={f.id})? [y/N]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return
        if confirm in ("y", "yes"):
            remove_skill(f.id, f.skill_type, martial_db, mystic_db)
            print(f"  Removed skill ID={f.id}.")
        else:
            print("  Cancelled.")


# ─────────────────────────────────────────────
# CLI entry points
# ─────────────────────────────────────────────
def _cli_typed(
    skill_type: str,
    db_path:    str,
    cmd:        str,
    rest:       List[str],
    SEP:        str,
) -> None:
    """Handle martial/mystic subcommand: list, show, add, edit, remove, search, init."""
    is_mystic = skill_type == SkillType.MYSTIC
    wizard    = _mystic_wizard if is_mystic else _martial_wizard

    # Both DB paths — one is the specific typed DB, other is the default for the other type
    if is_mystic:
        kw = {"martial_db": MARTIAL_ART_DB_PATH, "mystic_db": db_path}
    else:
        kw = {"martial_db": db_path, "mystic_db": MYSTIC_DB_PATH}

    if cmd == "init":
        (init_mystic_db if is_mystic else init_martial_db)(db_path)
        print(f"  DB initialised: {db_path}")
        return

    if cmd == "list":
        weapon = None
        i = 0
        while i < len(rest):
            if rest[i] == "--weapon" and i + 1 < len(rest):
                weapon = rest[i + 1]; i += 2
            else:
                i += 1
        skills = list_skills(skill_type=skill_type, weapon_type=weapon, **kw)
        if not skills:
            print("  (no skills found)"); return
        print(_COL_TYPED); print(_DIV_TYPED)
        for f in skills: _print_typed_row(f)
        return

    if cmd == "show":
        if not rest:
            print("  Usage: show <id>"); return
        f = get_skill_by_id(int(rest[0]), skill_type=skill_type, **kw)
        if not f:
            print(f"  Skill ID={rest[0]} not found."); return
        print(SEP); _print_skill_detail(f); print(SEP)
        return

    if cmd == "search":
        if not rest:
            print("  Usage: search <name>"); return
        skills = search_skills(" ".join(rest), skill_type=skill_type, **kw)
        if not skills:
            print("  No skills found."); return
        f = _pick_from_list(skills)
        if f:
            _interact_typed(f, **kw)
        return

    if cmd == "add":
        label = "mystic" if is_mystic else "martial art"
        print(f"\n{SEP}\n  Add new {label} skill\n{SEP}")
        f = wizard()
        if f:
            new_id = add_skill(f, **kw)
            print(f"\n  Added '{f.name}' with ID={new_id}.")
        return

    if cmd == "edit":
        if not rest:
            print("  Usage: edit <id>"); return
        f = get_skill_by_id(int(rest[0]), skill_type=skill_type, **kw)
        if not f:
            print(f"  Skill ID={rest[0]} not found."); return
        print(f"\n{SEP}\n  Edit skill ID={f.id}: {f.name}\n{SEP}")
        updated = wizard(base=f)
        if updated:
            update_skill(updated, **kw)
            print(f"\n  Updated skill ID={updated.id}.")
        return

    if cmd == "remove":
        if not rest:
            print("  Usage: remove <id>"); return
        skill_id = int(rest[0])
        f = get_skill_by_id(skill_id, skill_type=skill_type, **kw)
        if not f:
            print(f"  Skill ID={skill_id} not found."); return
        try:
            confirm = input(f"  Remove '{f.name}' (ID={skill_id})? [y/N]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return
        if confirm in ("y", "yes"):
            remove_skill(skill_id, skill_type, **kw)
            print(f"  Removed skill ID={skill_id}.")
        else:
            print("  Cancelled.")
        return

    print(f"  Unknown command '{cmd}'. Run with --help for usage.")


def _cli(argv: List[str]) -> None:
    SEP = "=" * 72

    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__); return

    cmd  = argv[0]
    rest = argv[1:]

    # ── Subcommand: martial / mystic ───────────
    if cmd in ("martial", "mystic"):
        skill_type = SkillType.MARTIAL_ART if cmd == "martial" else SkillType.MYSTIC
        db_path    = MARTIAL_ART_DB_PATH   if cmd == "martial" else MYSTIC_DB_PATH
        if not rest:
            print(f"  Usage: {cmd} <command> [args]  (use --help to see all commands)")
            return
        _cli_typed(skill_type, db_path, rest[0], rest[1:], SEP)
        return

    # ── Global: init ───────────────────────────
    if cmd == "init":
        init_all()
        print(f"  Both DBs initialised:")
        print(f"    {MARTIAL_ART_DB_PATH}")
        print(f"    {MYSTIC_DB_PATH}")
        return

    # ── Global: list ───────────────────────────
    if cmd == "list":
        skills = list_skills()
        if not skills:
            print("  (no skills found)"); return
        print(_COL_GLOBAL); print(_DIV_GLOBAL)
        for f in skills: _print_global_row(f)
        return

    # ── Global: search ─────────────────────────
    if cmd == "search":
        if not rest:
            print("  Usage: search <name>"); return
        skills = search_skills(" ".join(rest))
        if not skills:
            print("  No skills found."); return
        f = _pick_from_list(skills, global_view=True)
        if f:
            _interact_typed(f, MARTIAL_ART_DB_PATH, MYSTIC_DB_PATH)
        return

    print(f"  Unknown command '{cmd}'. Run with --help for usage.")


if __name__ == "__main__":
    _cli(sys.argv[1:])
