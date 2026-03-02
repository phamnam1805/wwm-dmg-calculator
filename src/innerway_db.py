#!/usr/bin/env python3
"""
innerway_db.py — Innerway DMG bonus database (SQLite)
======================================================
Each entry represents one innerway that grants a direct additive DMG bonus.

  buff_mult (with innerway) = 1 + … + sum(selected innerway dmg_bonus values)

Library API  : import and call functions directly (no print, safe for Web UI)
Interactive CLI: python src/innerway_db.py [command] [args]

Commands:
  list                 list all entries
  show      <id>       show detail
  add                  interactive wizard to add a new entry
  edit      <id>       edit an entry
  remove    <id>       delete an entry
  init                 initialise DB (runs automatically on first use)
"""

import sqlite3
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DB_PATH = str(Path(__file__).parent.parent / "dbs" / "innerway.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS innerwaydb (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT    NOT NULL,
    desc      TEXT    NOT NULL DEFAULT '',
    dmg_bonus REAL    NOT NULL DEFAULT 0.0
);
"""

# ─────────────────────────────────────────────
# Dataclass
# ─────────────────────────────────────────────
@dataclass
class InnerwayEntry:
    id:        Optional[int] = None
    name:      str           = "default"
    desc:      str           = ""
    dmg_bonus: float         = 0.0  # additive decimal: 0.05 = +5%


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────
def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_entry(row: sqlite3.Row) -> InnerwayEntry:
    return InnerwayEntry(
        id=row["id"], name=row["name"],
        desc=row["desc"], dmg_bonus=row["dmg_bonus"],
    )


# ─────────────────────────────────────────────
# Library API
# ─────────────────────────────────────────────
def init_db(db_path: str = DB_PATH) -> None:
    """Create table if it does not exist."""
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA)


def add_entry(entry: InnerwayEntry, db_path: str = DB_PATH) -> int:
    """Insert a new entry. Returns the new ID."""
    init_db(db_path)
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO innerwaydb (name, desc, dmg_bonus) VALUES (?, ?, ?)",
            (entry.name, entry.desc, entry.dmg_bonus),
        )
        return cur.lastrowid


def update_entry(entry: InnerwayEntry, db_path: str = DB_PATH) -> None:
    """Update an existing entry."""
    if entry.id is None:
        raise ValueError("InnerwayEntry.id must be set to update")
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE innerwaydb SET name = ?, desc = ?, dmg_bonus = ? WHERE id = ?",
            (entry.name, entry.desc, entry.dmg_bonus, entry.id),
        )


def remove_entry(entry_id: int, db_path: str = DB_PATH) -> None:
    """Delete an entry by ID."""
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM innerwaydb WHERE id = ?", (entry_id,))


def get_entry_by_id(entry_id: int, db_path: str = DB_PATH) -> Optional[InnerwayEntry]:
    """Return an entry by ID, or None."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM innerwaydb WHERE id = ?", (entry_id,)
        ).fetchone()
    return _row_to_entry(row) if row else None


def get_entries_by_ids(
    ids: List[int], db_path: str = DB_PATH
) -> List[InnerwayEntry]:
    """Return entries for a list of IDs (preserves order)."""
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT * FROM innerwaydb WHERE id IN ({placeholders})", ids
        ).fetchall()
    by_id = {r["id"]: _row_to_entry(r) for r in rows}
    return [by_id[i] for i in ids if i in by_id]


def list_entries(db_path: str = DB_PATH) -> List[InnerwayEntry]:
    """Return all entries ordered by ID."""
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM innerwaydb ORDER BY id").fetchall()
    return [_row_to_entry(r) for r in rows]


# ─────────────────────────────────────────────
# CLI helpers
# ─────────────────────────────────────────────
_SEP = "─" * 60


def _pct(v: float) -> str:
    return f"{v * 100:+.2f}%"


def print_entry_row(e: InnerwayEntry) -> None:
    print(f"  [{e.id}]  {e.name:<36}  {_pct(e.dmg_bonus):>8}  {e.desc}")


def print_entry_detail(e: InnerwayEntry) -> None:
    print(f"\n{_SEP}")
    print(f"  [{e.id}]  {e.name}")
    print(_SEP)
    print(f"  DMG bonus : {_pct(e.dmg_bonus)}")
    print(f"  Desc      : {e.desc or '(none)'}")
    print()


def _wizard(base: Optional[InnerwayEntry] = None) -> Optional[InnerwayEntry]:
    b = base or InnerwayEntry()
    try:
        print()
        raw = input(f"  Name [{b.name}]: ").strip()
        name = raw if raw else b.name

        raw = input(f"  Desc [{b.desc}]: ").strip()
        desc = raw if raw else b.desc

        raw = input(f"  DMG bonus (5% → 0.05) [{b.dmg_bonus}]: ").strip()
        try:
            dmg_bonus = float(raw) if raw else b.dmg_bonus
        except ValueError:
            print("  Invalid number, using 0.0")
            dmg_bonus = 0.0

        return InnerwayEntry(id=b.id, name=name, desc=desc, dmg_bonus=dmg_bonus)
    except (KeyboardInterrupt, EOFError):
        print("\n  Cancelled.")
        return None


# ─────────────────────────────────────────────
# CLI commands
# ─────────────────────────────────────────────
def _cmd_list() -> None:
    entries = list_entries()
    if not entries:
        print("  (no entries)")
        return
    print(f"  {'ID':<4}  {'Name':<36}  {'Bonus':>8}  Desc")
    print(f"  {'-'*4}  {'-'*36}  {'-'*8}  {'-'*20}")
    for e in entries:
        print_entry_row(e)


def _cmd_show(args: List[str]) -> None:
    if not args:
        print("Usage: show <id>")
        return
    e = get_entry_by_id(int(args[0]))
    if e is None:
        print(f"  Entry #{args[0]} not found.")
        return
    print_entry_detail(e)


def _cmd_add() -> None:
    e = _wizard()
    if e is None:
        return
    new_id = add_entry(e)
    print(f"  Added innerway #{new_id}: {e.name} ({_pct(e.dmg_bonus)})")


def _cmd_edit(args: List[str]) -> None:
    if not args:
        print("Usage: edit <id>")
        return
    e = get_entry_by_id(int(args[0]))
    if e is None:
        print(f"  Entry #{args[0]} not found.")
        return
    updated = _wizard(base=e)
    if updated is None:
        return
    update_entry(updated)
    print(f"  Updated innerway #{updated.id}.")


def _cmd_remove(args: List[str]) -> None:
    if not args:
        print("Usage: remove <id>")
        return
    entry_id = int(args[0])
    e = get_entry_by_id(entry_id)
    if e is None:
        print(f"  Entry #{entry_id} not found.")
        return
    confirm = input(f"  Delete '{e.name}' (#{entry_id})? [y/N]: ").strip().lower()
    if confirm in ("y", "yes"):
        remove_entry(entry_id)
        print(f"  Removed innerway #{entry_id}.")
    else:
        print("  Cancelled.")


def _cmd_init() -> None:
    init_db()
    print(f"  DB initialised at {DB_PATH}")


_USAGE = """\
Usage: python src/innerway_db.py <command> [args]

Commands:
  list                 list all entries
  show      <id>       show detail
  add                  interactive wizard to add a new entry
  edit      <id>       edit an entry
  remove    <id>       delete an entry
  init                 initialise DB
"""

if __name__ == "__main__":
    init_db()
    argv = sys.argv[1:]
    if not argv:
        print(_USAGE)
        sys.exit(0)

    cmd = argv[0].lower()
    rest = argv[1:]
    dispatch = {
        "list":   lambda: _cmd_list(),
        "show":   lambda: _cmd_show(rest),
        "add":    lambda: _cmd_add(),
        "edit":   lambda: _cmd_edit(rest),
        "remove": lambda: _cmd_remove(rest),
        "init":   lambda: _cmd_init(),
    }
    if cmd not in dispatch:
        print(f"Unknown command: {cmd}\n")
        print(_USAGE)
        sys.exit(1)
    dispatch[cmd]()
