# WWM Damage Calculator

Damage calculator for **Where Winds Meet**.

## Usage

```bash
python calculator.py                    # current stats + active skill
python calculator.py change.cnf         # compare base vs override
python calculator.py -s "skill name"    # select skill formula
python calculator.py pinda-stats.cnf -s "soaring spin 2nd hit"    # Compare base vs pinda-stats.cnf over skill Soaring Spin 2nd hit
python calculator.py change.cnf -v      # include double marginal gain
```

## Config files

| File                 | Description                            |
|----------------------|----------------------------------------|
| `base.cnf`           | Character stats                        |
| `skill_formulas.cnf` | Damage formula for each skill          |
| `marginal.cnf`       | Equal-cost upgrade options to compare  |

Override files (partial stats applied on top of base):
`affi.cnf`, `crit.cnf`, `prec.cnf`, `phys.cnf`, `bellstrike.cnf`, `bamboocut.cnf`, `pendant.cnf`, etc.

### Config format — character stats

```ini
[character]
main_attribute = bellstrike

[physical]
min = 665
max = 1885
pen       = 5.1    # optional, default 0
dmg_bonus = 0.0    # optional, default 0

[attribute.bellstrike]
min = 231
max = 457
pen       = 41     # optional, default 0
dmg_bonus = 0.066  # optional, default 0

[attribute.stonesplit]
min = 50
max = 50

[rates]
affinity_rate        = 0.374
precision_rate       = 0.942
critical_rate        = 0.382
affinity_mult        = 1.402
critical_mult        = 1.5
direct_affinity_rate = 0.023
direct_critical_rate = 0.046
```

### Config format — skill formulas

```ini
[vagrant sword 3rd hit]
phys_coeff = 1.8283
phys_bonus = 320
attr_coeff = 2.7425
attr_bonus = 185

[soaring spin 2nd hit]
phys_coeff = 3.6831
phys_bonus = 454
attr_coeff = 1
attr_bonus = 0
is_mystic  = true    # no x1.5 on main attribute, attr_bonus only on main attr
```

### Config format — marginal gain

```ini
# Each section = one equal-cost option. Values are deltas (+/-)
[+2.8% affinity]
affinity_rate = +0.028

[+47 max physical ATK]
physical_max = +47

[+31 max bellstrike ATK]
bellstrike_max = +31

# Also supported: physical_pen, physical_dmg_bonus, <attr>_pen, <attr>_dmg_bonus
```

---

## Damage Formula

### Per-stat multiplier

Each attack stat (physical, bellstrike, stonesplit, …) has its own `pen` and `dmg_bonus`:

```
dmg_mult = (1 + pen / 200) * (1 + dmg_bonus)
```

When both are 0 (default), `dmg_mult = 1.0` — no effect.

### Physical component

```
phys_dmg = (phys_atk * phys_coeff + phys_bonus) * physical.dmg_mult()
```

### Attribute component

Each attribute is computed independently then summed:

```
# Normal skill:
attr_contribution = (1.5 * attr_coeff * main_attr_atk + attr_bonus) * main_attr.dmg_mult()
                  + (1.0 * attr_coeff * other_attr_atk            ) * other_attr.dmg_mult()

# Mystic skill (is_mystic = true):
attr_contribution = (1.0 * attr_coeff * attr_atk + attr_bonus) * attr.dmg_mult()
                  # x1.5 removed for main attr; attr_bonus only on main attr
```

### Total damage range

```
total_max = phys_max + attr_max
total_min = phys_min + attr_min
E[roll]   = (total_min + total_max) / 2
```

### Effective rates (after cap)

```
eff_affinity  = min(affinity_rate,  40%) + min(direct_affinity, 10%)
eff_precision = min(precision_rate, 100%)
eff_crit      = min(crit_rate,      80%) + min(direct_crit,     20%)
```

### Hit type probabilities

```
P(orange) = eff_affinity
P(yellow) = (1 - eff_affinity) * eff_precision * eff_crit
P(white)  = (1 - eff_affinity) * eff_precision * (1 - eff_crit)
P(gray)   = (1 - eff_affinity) * (1 - eff_precision)
```

### Damage per hit type

| Color  | Type     | Damage                        | Range                          |
|--------|----------|-------------------------------|--------------------------------|
| ORANGE | affinity | `total_max × affinity_mult`   | fixed — always max             |
| YELLOW | crit     | `roll(min,max) × crit_mult`   | `total_min×cm ~ total_max×cm`  |
| WHITE  | normal   | `roll(min, max)`              | `total_min ~ total_max`        |
| GRAY   | abrasion | `total_min`                   | fixed — always min             |

### Expected damage

```
E[DMG] = P(orange) × total_max × affinity_mult
       + P(yellow) × E[roll]   × crit_mult
       + P(white)  × E[roll]
       + P(gray)   × total_min
```

### Skill types

| Type         | `phys_bonus` | main attr x1.5 | `attr_bonus` applies to |
|--------------|:------------:|:--------------:|-------------------------|
| Normal skill | yes          | yes            | main attr only          |
| Mystic skill | yes          | **no**         | main attr only          |
| DOT          | **no**       | **no**         | all attrs (x1.0)        |

---

## Stability stats (from simulation)

| Stat | Description                                                       |
|------|-------------------------------------------------------------------|
| mean | Average damage over n rolls (converges to theoretical E[DMG])     |
| std  | Standard deviation — high std means inconsistent/swingy damage    |
| min  | Lowest damage rolled (usually GRAY = total_min)                   |
| max  | Highest damage rolled (usually ORANGE = affinity dmg)             |
| p10  | 10% of hits fall below this value                                 |
| p90  | 90% of hits fall below this value (p10~p90 = practical range)     |

---

## Attributes

`bellstrike` · `stonesplit` · `bamboocut` · `silkbind`

## Rate caps

| Rate           | Normal cap | Direct cap |
|----------------|------------|------------|
| affinity_rate  | 40%        | +10%       |
| precision_rate | 100%       | —          |
| critical_rate  | 80%        | +20%       |

## Notes

The following modifiers are **not modeled** and must be accounted for manually:

- Target defense
- Damage boost multipliers (增伤) in general, including:
  - All Martial Art Skill DMG Boost
  - Specified Weapon Martial Art Boost
  - Mystic Skill DMG Boost
  - DMG Boost vs Boss Units
  - PvP Boost
- **DOT damage** (bleed, burn, etc.) uses a different formula — no `phys_bonus`, no x1.5 on main attr — not supported yet
- **Healer damage** follows a separate formula — not supported

Simulation uses `/dev/urandom` (SystemRandom)
