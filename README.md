# WWM Damage Calculator

Damage calculator for **Where Winds Meet**.

Computes the **expected (average) damage of a specific skill** based on a character's stats. From there:

- **Marginal gain analysis** — shows how much E[DMG] changes when each stat is increased by a fixed amount, giving actionable guidance on which stat to prioritize next.
- **Profile comparison** — compares two character profiles (e.g. different gear sets) side-by-side, making it easy to see which equipment upgrade actually yields higher damage output.

**Requires:** Python 3.9+ · No third-party dependencies (stdlib only: `sqlite3`, `random`, `dataclasses`, `pathlib`)

> **Note:** `calculator.py` in the repo root is deprecated and no longer used. It is kept for reference only.

---

## Architecture

The calculator consists of five modules, each with a library API (importable, no prints) and an interactive CLI:

| Module | DB file | Purpose |
|---|---|---|
| `src/profile_db.py` | `dbs/profiles.db` | Character profiles (ATK stats + all bonuses) |
| `src/skill_db.py` | `dbs/martial_art_skills.db` · `dbs/mystic_skills.db` | Skill damage formulas |
| `src/marginal_gain_db.py` | `dbs/marginal_gains.db` | Stat delta entries for upgrade comparison |
| `src/innerway_db.py` | `dbs/innerway.db` | Innerway DMG bonus entries |
| `src/dmg_resolver.py` | *(no own DB)* | Interactive damage resolver — ties everything together |

Database files are stored in `dbs/` and committed to the repository.

---

## Quick start

```bash
# 1. Create a character profile
python src/profile_db.py add

# 2. Add a skill formula
python src/skill_db.py martial add   # martial art skill
python src/skill_db.py mystic  add   # mystic skill

# 3. (Optional) Add marginal gain entries for upgrade comparison
python src/marginal_gain_db.py add

# 4. Run the damage resolver
python src/dmg_resolver.py
```

---

## Module reference

### `src/profile_db.py` — Character profiles

```bash
python src/profile_db.py list
python src/profile_db.py show   <id>
python src/profile_db.py add
python src/profile_db.py edit   <id>
python src/profile_db.py remove <id>
python src/profile_db.py search <name>
python src/profile_db.py init           # initialise DB (auto-called on first use)
```

A **CharProfile** stores:

| Group | Fields |
|---|---|
| Physical ATK | `physical_min`, `physical_max`, `physical_pen`, `physical_dmg_bonus` |
| Attribute ATK × 4 | `{attr}_min/max/pen/dmg_bonus` for each of: `bellstrike` `stonesplit` `bamboocut` `silkbind` |
| Combat rates | `affinity_rate`, `direct_affinity_rate`, `precision_rate`, `critical_rate`, `direct_critical_rate`, `affinity_mult`, `critical_mult` |
| Martial art DMG bonus | `all_martial_art_dmg_bonus`, `{weapon}_dmg_bonus` for each weapon type |
| Mystic DMG bonus | `{mystic_type}_dmg_bonus` for each mystic type |
| Target DMG bonus | `boss_dmg_bonus`, `pvp_dmg_bonus` |

Rates are stored as decimals (`0.374` = 37.4%). Rate caps are enforced automatically on load.

---

### `src/skill_db.py` — Skill formulas

```bash
python src/skill_db.py list                    # all skills (martial + mystic)
python src/skill_db.py martial list
python src/skill_db.py martial show   <id>
python src/skill_db.py martial add
python src/skill_db.py martial edit   <id>
python src/skill_db.py martial remove <id>
python src/skill_db.py mystic  list
python src/skill_db.py mystic  show   <id>
python src/skill_db.py mystic  add
python src/skill_db.py mystic  edit   <id>
python src/skill_db.py mystic  remove <id>
```

A **SkillFormula** stores:

| Field | Description |
|---|---|
| `phys_coeff` | Physical ATK multiplier |
| `phys_bonus` | Flat physical ATK bonus (added before `phys_coeff`) |
| `attr_coeff` | Main attribute multiplier (derived: `phys_coeff × 1.5`; stored for reference) |
| `attr_bonus` | Flat main-attribute ATK bonus |
| `skill_type` | `martial_art` or `mystic` |
| `attribute_type` | Main attribute (martial art only): `bellstrike` / `stonesplit` / `bamboocut` / `silkbind` |
| `weapon_type` | Weapon type (martial art only, required) |
| `mystic_type` | Sub-type (mystic only): `area_debuff` / `area_dmg` / `single_target_control` / `single_target_burst` |
| `is_dot` | DOT flag — disables `phys_bonus` and attribute x1.5 scaling |

---

### `src/marginal_gain_db.py` — Marginal gains

```bash
python src/marginal_gain_db.py list
python src/marginal_gain_db.py show   <id>
python src/marginal_gain_db.py add
python src/marginal_gain_db.py edit   <id>
python src/marginal_gain_db.py remove <id>
python src/marginal_gain_db.py init
```

Each **MarginalGain** has a name and a set of stat deltas (e.g. `bellstrike_max +26.6`, `sword_dmg_bonus +0.038`). Deltas are applied on top of a profile to measure E[DMG] change per upgrade option.

---

### `src/innerway_db.py` — Innerway DMG bonuses

```bash
python src/innerway_db.py list
python src/innerway_db.py show   <id>
python src/innerway_db.py add
python src/innerway_db.py edit   <id>
python src/innerway_db.py remove <id>
python src/innerway_db.py init
```

Each **InnerwayEntry** stores a `name`, a human-readable `desc`, and a `dmg_bonus` (additive decimal, e.g. `0.20` = +20%). When resolving damage, selected entries are summed and added into `buff_mult`. Note: only the direct DMG bonus effect is modeled — innerway stat modifiers are not (see [Not modeled](#not-modeled)).

---

### `src/dmg_resolver.py` — Damage resolver

```bash
python src/dmg_resolver.py
```

Interactive CLI with the following flow:

```
Select Profile → Select Skill → Set combat context → Mode loop
```

**Combat context** (asked once per skill selection, can be changed via `[4]`):
- **Target**: `boss` (default) or `pvp` — determines which target DMG bonus is added
- **Attunement affix DMG bonus** (martial art skills only): enter as decimal (`0.058` = 5.8%)
- **Innerway DMG bonus**: shows list from `innerway.db`, select by ID (space-separated) or Enter to skip — selected bonuses are summed and added to `buff_mult`

**Mode loop options:**

| Option | Description |
|---|---|
| `[1] Simulate` | Monte Carlo simulation. Asks number of rolls (default 100). Shows theory vs simulation comparison with hit-type breakdown. |
| `[2] Marginal gain comparison` | Lists all gains in DB, lets you pick by ID (comma-separated or `all`). Shows E[DMG] ↑ and std ↓ for each option. |
| `[3] Compare with another profile` | Pick a second profile (each with its own affix and innerway bonus). Shows side-by-side Δ and Δ% for all stats. |
| `[4] Change context` | Re-ask target, affix bonus, and innerway. |
| `[5] Change skill` | Back to skill selection. |
| `[6] Change profile` | Back to profile selection. |

---

## Damage formula

### Base damage

```
phys_dmg = (phys_coeff × phys_atk + phys_bonus) × phys_dmg_mult

attr_dmg  = attr_coeff × (main_attr_atk + attr_bonus) × main_attr_dmg_mult   # main attr
          + phys_coeff × other_attr_atk                × other_attr_dmg_mult  # other attrs

base_dmg = phys_dmg + attr_dmg
```

Where `phys_dmg_mult = (1 + pen/200) × (1 + dmg_bonus)` for each stat.

For **mystic skills**: same `phys_coeff` for all attributes, no x1.5 scaling, `attr_bonus = 0`.

For **DOT skills**: no `phys_bonus`, no x1.5 on main attribute.

### Buff multiplier

```
# Martial art skill:
buff_mult = 1 + all_martial_art_dmg_bonus + {weapon_type}_dmg_bonus + target_bonus
          + innerway_bonus

# Mystic skill:
buff_mult = 1 + {mystic_type}_dmg_bonus + target_bonus + innerway_bonus

# target_bonus  = boss_dmg_bonus  (when target = boss)
#               = pvp_dmg_bonus   (when target = pvp)
# innerway_bonus = sum of selected innerway dmg_bonus values (0.0 if none selected)
```

### Affix multiplier

```
affix_mult = 1 + attunement_affix_bonus   # martial art skills
affix_mult = 1.0                           # mystic skills (no attunement affix)
```

### Scaled damage

```
scaled_max    = base_max    × buff_mult × affix_mult
scaled_e_roll = base_e_roll × buff_mult × affix_mult   # e_roll = (min + max) / 2
scaled_min    = base_min    × buff_mult × affix_mult
```

### Hit type probabilities (after rate caps)

```
eff_affinity  = min(affinity_rate, 40%) + min(direct_affinity_rate, 10%)
eff_precision = min(precision_rate, 100%)
eff_crit      = min(critical_rate, 80%) + min(direct_critical_rate, 20%)

P(orange) = eff_affinity
P(yellow) = (1 - eff_affinity) × eff_precision × eff_crit
P(white)  = (1 - eff_affinity) × eff_precision × (1 - eff_crit)
P(gray)   = (1 - eff_affinity) × (1 - eff_precision)
```

### Damage per hit type

| Color | Type | Damage formula |
|---|---|---|
| **ORANGE** | affinity | `scaled_max × affinity_mult` (fixed — always max roll) |
| **YELLOW** | crit | `roll(scaled_min, scaled_max) × critical_mult` |
| **WHITE** | normal | `roll(scaled_min, scaled_max)` |
| **GRAY** | abrasion | `scaled_min` (fixed — always min roll) |

### Expected damage & standard deviation

```
E[DMG] = P(orange) × scaled_max × affinity_mult
       + P(yellow) × scaled_e_roll × critical_mult
       + P(white)  × scaled_e_roll
       + P(gray)   × scaled_min
```

Variance is computed analytically via the **Law of Total Variance** (Var[Uniform(lo,hi)] = (hi−lo)²/12). Simulation uses `random.SystemRandom` (/dev/urandom).

---

## Rate caps

| Rate | Normal cap | Direct cap |
|---|---|---|
| `affinity_rate` | 40% | `direct_affinity_rate` +10% |
| `precision_rate` | 100% | — |
| `critical_rate` | 80% | `direct_critical_rate` +20% |

## Attributes

`bellstrike` · `stonesplit` · `bamboocut` · `silkbind`

## Weapon types

`sword` · `dual_blades` · `spear` · `fan` · `umbrella` · `heng_blade` · `mo_blade` · `rope_dart`

## Mystic sub-types

`area_debuff` · `area_dmg` · `single_target_control` · `single_target_burst`

---

## Not modeled

- Target defense / resistance
- Healer healing formula
- Innerway **stat modifiers** in combat — e.g. Yi River stacks granted by Morale Chant that increase ATK stats during the fight. Only the direct DMG bonus effect of innerway is currently supported.
