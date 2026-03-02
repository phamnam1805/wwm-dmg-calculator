#!/usr/bin/env python3
"""
marginal_gain_db.py — Marginal gain database (SQLite)
======================================================
A marginal gain is a named set of stat deltas applied on top of a CharProfile
to measure how much E[DMG] changes per upgrade option.

Library API  : import and call functions directly (no print, safe for Web UI)
Interactive CLI: python src/marginal_gain_db.py [command] [args]

Commands:
  list             list all gains
  show  <id>       show gain detail (name + all non-zero deltas)
  add              interactive wizard to add a new gain
  edit  <id>       interactive wizard to edit a gain
  remove <id>      delete a gain by ID
  init             initialise DB
"""

import sqlite3
import sys
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Dict, List, Optional

from profile_db import ATTRIBUTES, WEAPON_TYPES, MYSTIC_TYPES

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DB_PATH = str(Path(__file__).parent.parent / "dbs" / "marginal_gains.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS gains (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS gain_deltas (
    gain_id INTEGER NOT NULL,
    field   TEXT    NOT NULL,
    delta   REAL    NOT NULL,
    PRIMARY KEY (gain_id, field),
    FOREIGN KEY (gain_id) REFERENCES gains(id) ON DELETE CASCADE
);
"""

# ─────────────────────────────────────────────
# All adjustable fields (mirrors CharProfile float fields)
# ─────────────────────────────────────────────
_FIELD_GROUPS: List[tuple] = [
    ("Physical", [
        "physical_min", "physical_max", "physical_pen", "physical_dmg_bonus",
    ]),
    *[
        (f"Attribute · {attr}", [
            f"{attr}_min", f"{attr}_max", f"{attr}_pen", f"{attr}_dmg_bonus",
        ])
        for attr in ATTRIBUTES
    ],
    ("Combat rates", [
        "affinity_rate", "direct_affinity_rate",
        "precision_rate",
        "critical_rate", "direct_critical_rate",
        "affinity_mult", "critical_mult",
    ]),
    ("Martial art DMG bonus", [
        "all_martial_art_dmg_bonus",
        *(f"{w}_dmg_bonus" for w in WEAPON_TYPES),
    ]),
    ("Mystic type DMG bonus", [
        f"{m}_dmg_bonus" for m in MYSTIC_TYPES
    ]),
    ("Target DMG bonus", [
        "pvp_dmg_bonus",
        "boss_dmg_bonus",
    ]),
]

_ALL_FIELDS: List[str] = [f for _, fields in _FIELD_GROUPS for f in fields]
_FIELD_SET: frozenset    = frozenset(_ALL_FIELDS)
_FIELD_INDEX: Dict[str, int] = {f: i + 1 for i, f in enumerate(_ALL_FIELDS)}


# ─────────────────────────────────────────────
# Data class
# ─────────────────────────────────────────────
@dataclass
class MarginalGain:
    id:     Optional[int]    = None
    name:   str              = "default"
    deltas: Dict[str, float] = dc_field(default_factory=dict)


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────
def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _load_deltas(conn: sqlite3.Connection, gain_id: int) -> Dict[str, float]:
    rows = conn.execute(
        "SELECT field, delta FROM gain_deltas WHERE gain_id=? ORDER BY field",
        (gain_id,),
    ).fetchall()
    return {r["field"]: r["delta"] for r in rows}


def _save_deltas(conn: sqlite3.Connection, gain_id: int, deltas: Dict[str, float]) -> None:
    conn.execute("DELETE FROM gain_deltas WHERE gain_id=?", (gain_id,))
    for field, delta in deltas.items():
        if delta != 0.0:
            conn.execute(
                "INSERT INTO gain_deltas (gain_id, field, delta) VALUES (?, ?, ?)",
                (gain_id, field, delta),
            )


# ─────────────────────────────────────────────
# Library API  (no print statements — safe to import from Web UI)
# ─────────────────────────────────────────────
def init_db(db_path: str = DB_PATH) -> None:
    """Create gains tables if they do not exist. Safe to call multiple times."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA)


def add_gain(gain: MarginalGain, db_path: str = DB_PATH) -> int:
    """Insert a new gain. Returns the auto-assigned ID."""
    init_db(db_path)
    with _connect(db_path) as conn:
        cur    = conn.execute("INSERT INTO gains (name) VALUES (?)", (gain.name,))
        new_id = cur.lastrowid
        _save_deltas(conn, new_id, gain.deltas)
        return new_id


def update_gain(gain: MarginalGain, db_path: str = DB_PATH) -> None:
    """Update an existing gain by gain.id. Raises ValueError if id is None."""
    if gain.id is None:
        raise ValueError("Cannot update a gain without an ID.")
    with _connect(db_path) as conn:
        conn.execute("UPDATE gains SET name=? WHERE id=?", (gain.name, gain.id))
        _save_deltas(conn, gain.id, gain.deltas)


def remove_gain(gain_id: int, db_path: str = DB_PATH) -> None:
    """Delete a gain by ID (cascades to deltas). Silent if not found."""
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM gains WHERE id=?", (gain_id,))


def get_gain_by_id(gain_id: int, db_path: str = DB_PATH) -> Optional[MarginalGain]:
    """Return a MarginalGain by ID, or None if not found."""
    init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM gains WHERE id=?", (gain_id,)).fetchone()
        if not row:
            return None
        deltas = _load_deltas(conn, gain_id)
    return MarginalGain(id=row["id"], name=row["name"], deltas=deltas)


def list_gains(db_path: str = DB_PATH) -> List[MarginalGain]:
    """Return all gains ordered by id."""
    init_db(db_path)
    with _connect(db_path) as conn:
        rows   = conn.execute("SELECT * FROM gains ORDER BY id").fetchall()
        result = []
        for row in rows:
            deltas = _load_deltas(conn, row["id"])
            result.append(MarginalGain(id=row["id"], name=row["name"], deltas=deltas))
    return result


# ─────────────────────────────────────────────
# CLI helpers
# ─────────────────────────────────────────────
def _prompt(label: str, default=None) -> str:
    hint = f" [{default}]" if default is not None else ""
    val  = input(f"  {label}{hint}: ").strip()
    return val if val else (str(default) if default is not None else "")


def _delta_str(deltas: Dict[str, float], max_chars: int = 72) -> str:
    if not deltas:
        return "(no deltas)"
    parts = [f"{f} {d:+.4g}" for f, d in sorted(deltas.items())]
    joined = ", ".join(parts)
    return joined if len(joined) <= max_chars else joined[:max_chars - 1] + "…"


def _print_gain_row(g: MarginalGain) -> None:
    print(f"  {g.id:>4}  {g.name:<30}  {_delta_str(g.deltas)}")


def _print_gain_detail(g: MarginalGain) -> None:
    print(f"  ID   : {g.id}")
    print(f"  Name : {g.name}")
    if g.deltas:
        print("  Deltas:")
        for field, delta in sorted(g.deltas.items()):
            print(f"    {field:<40}  {delta:+.6g}")
    else:
        print("  Deltas: (none)")


def _print_field_list(current_deltas: Dict[str, float]) -> None:
    """Print all fields numbered and grouped, with current delta values annotated."""
    idx = 1
    for group_name, fields in _FIELD_GROUPS:
        print(f"\n  [{group_name}]")
        for field in fields:
            cur = current_deltas.get(field, 0.0)
            indicator = f"  {cur:+.4g}" if cur != 0.0 else ""
            print(f"    [{idx:>2}] {field:<40}{indicator}")
            idx += 1


def _delta_wizard(base_deltas: Dict[str, float]) -> Optional[Dict[str, float]]:
    """
    Interactive loop to build/edit a delta dict.
    Commands: field-name | field-number | 'list' | 'clear' | 'done'
    Returns updated deltas, or None on cancel.
    """
    deltas = dict(base_deltas)
    try:
        while True:
            # Show current state
            if deltas:
                print(f"\n  Current deltas:")
                for field, delta in sorted(deltas.items()):
                    print(f"    {field:<40}  {delta:+.6g}")
            else:
                print("\n  Current deltas: (none)")

            print()
            print("  Enter field name or number · 'list' to see all · 'clear' · 'done'")
            try:
                raw = input("  > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n  Cancelled."); return None

            if not raw or raw.lower() == "done":
                break

            if raw.lower() == "clear":
                deltas.clear()
                print("  All deltas cleared.")
                continue

            if raw.lower() == "list":
                _print_field_list(deltas)
                continue

            # Resolve field by number or name
            field: Optional[str] = None
            if raw.isdigit():
                n = int(raw)
                if 1 <= n <= len(_ALL_FIELDS):
                    field = _ALL_FIELDS[n - 1]
                else:
                    print(f"  Out of range (1–{len(_ALL_FIELDS)})."); continue
            elif raw in _FIELD_SET:
                field = raw
            else:
                # Partial name match
                matches = [f for f in _ALL_FIELDS if raw.lower() in f.lower()]
                if len(matches) == 1:
                    field = matches[0]
                elif len(matches) > 1:
                    print(f"  Ambiguous '{raw}'. Matches: {', '.join(matches)}")
                    continue
                else:
                    print(f"  Unknown field '{raw}'. Type 'list' to see all fields.")
                    continue

            cur_val = deltas.get(field, 0.0)
            raw_delta = _prompt(f"  {field} delta", f"{cur_val:+g}" if cur_val != 0.0 else "0")
            try:
                new_val = float(raw_delta)
            except (ValueError, TypeError):
                print("  Invalid float, skipping."); continue

            if new_val == 0.0:
                if field in deltas:
                    del deltas[field]
                    print(f"  {field}: removed (delta = 0).")
                else:
                    print(f"  {field}: no change (already 0).")
            else:
                deltas[field] = new_val
                print(f"  {field}: {new_val:+.6g}")

    except (KeyboardInterrupt, EOFError):
        print("\n  Cancelled."); return None

    return deltas


def _wizard(base: Optional[MarginalGain] = None) -> Optional[MarginalGain]:
    """Interactive wizard for a MarginalGain. Returns MarginalGain or None on cancel."""
    b   = base or MarginalGain()
    SEP = "  " + "-" * 52
    try:
        print(f"\n{SEP}\n  Gain info\n{SEP}")
        name = _prompt("name", b.name)
        if not name:
            print("  Name cannot be empty."); return None

        print(f"\n{SEP}\n  Set stat deltas\n{SEP}")
        print("  Tip: type field name or number; 'list' to browse; 'done' to finish.")
        deltas = _delta_wizard(b.deltas)
        if deltas is None:
            return None

        return MarginalGain(id=b.id, name=name, deltas=deltas)

    except (KeyboardInterrupt, EOFError):
        print("\n  Cancelled."); return None


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────
def _cli(argv: List[str], db_path: str = DB_PATH) -> None:
    SEP = "=" * 72

    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__); return

    cmd  = argv[0]
    rest = argv[1:]

    if cmd == "init":
        init_db(db_path)
        print(f"  DB initialised: {db_path}"); return

    if cmd == "list":
        gains = list_gains(db_path)
        if not gains:
            print("  (no gains found)"); return
        print(f"  {'ID':>4}  {'Name':<30}  Deltas")
        print("  " + "-" * 72)
        for g in gains:
            _print_gain_row(g)
        return

    if cmd == "show":
        if not rest:
            print("  Usage: show <id>"); return
        g = get_gain_by_id(int(rest[0]), db_path)
        if not g:
            print(f"  Gain ID={rest[0]} not found."); return
        print(SEP); _print_gain_detail(g); print(SEP); return

    if cmd == "add":
        print(f"\n{SEP}\n  Add new marginal gain\n{SEP}")
        g = _wizard()
        if g:
            new_id = add_gain(g, db_path)
            print(f"\n  Added '{g.name}' with ID={new_id}.")
        return

    if cmd == "edit":
        if not rest:
            print("  Usage: edit <id>"); return
        g = get_gain_by_id(int(rest[0]), db_path)
        if not g:
            print(f"  Gain ID={rest[0]} not found."); return
        print(f"\n{SEP}\n  Edit gain ID={g.id}: {g.name}\n{SEP}")
        updated = _wizard(base=g)
        if updated:
            update_gain(updated, db_path)
            print(f"\n  Updated gain ID={updated.id}.")
        return

    if cmd == "remove":
        if not rest:
            print("  Usage: remove <id>"); return
        gain_id = int(rest[0])
        g = get_gain_by_id(gain_id, db_path)
        if not g:
            print(f"  Gain ID={gain_id} not found."); return
        try:
            confirm = input(f"  Remove '{g.name}' (ID={gain_id})? [y/N]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return
        if confirm in ("y", "yes"):
            remove_gain(gain_id, db_path)
            print(f"  Removed gain ID={gain_id}.")
        else:
            print("  Cancelled.")
        return

    print(f"  Unknown command '{cmd}'. Run with --help for usage.")


if __name__ == "__main__":
    _cli(sys.argv[1:])
