# WWM Damage Calculator

Damage calculator for **Where Winds Meet**.

## Usage

```bash
python calculator.py                    # current stats + active skill
python calculator.py change.cnf         # compare base vs override
python calculator.py -s "skill name"    # select skill formula
python calculator.py change.cnf -v      # include double marginal gain
```

## Config files

| File                 | Description                              |
|----------------------|------------------------------------------|
| `base.cnf`           | Character stats                          |
| `skill_formulas.cnf` | Damage formula for each skill            |
| `marginal.cnf`       | Equal-cost upgrade options to compare    |

Override files (partial stats applied on top of base):
`affi.cnf`, `crit.cnf`, `prec.cnf`, `phys.cnf`, `bellstrike.cnf`, `bamboocut.cnf`, `pendant.cnf`, etc.

## Damage Formula

### Raw damage

```
phys_dmg = physical_atk * phys_coeff + phys_bonus

attr_dmg = (main_attr * 1.5 + Σ other_attrs) * attr_coeff + attr_bonus
         # mystic skill: main_attr * 1.0 (no x1.5 bonus)

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

| Color  | Type     | Damage                         | Notes               |
|--------|----------|--------------------------------|---------------------|
| ORANGE | affinity | `total_max × affinity_mult`    | fixed, always max   |
| YELLOW | crit     | `roll(min, max) × crit_mult`   | random              |
| WHITE  | normal   | `roll(min, max)`               | random              |
| GRAY   | abrasion | `total_min`                    | fixed, always min   |

### Expected damage

```
E[DMG] = P(orange) × total_max × affinity_mult
       + P(yellow) × E[roll]   × crit_mult
       + P(white)  × E[roll]
       + P(gray)   × total_min
```

## Stability stats (from simulation)

| Stat | Description                                                        |
|------|--------------------------------------------------------------------|
| mean | Average damage over n rolls (converges to theoretical E[DMG])      |
| std  | Standard deviation — high std means inconsistent/swingy damage     |
| min  | Lowest damage rolled (usually GRAY = total_min)                    |
| max  | Highest damage rolled (usually ORANGE = affinity dmg)              |
| p10  | 10% of hits fall below this value                                  |
| p90  | 90% of hits fall below this value (p10~p90 = practical range)      |

## Attributes

`bellstrike` · `stonesplit` · `bamboocut` · `silkbind`

## Rate caps

| Rate           | Normal cap | Direct cap |
|----------------|------------|------------|
| affinity_rate  | 40%        | +10%       |
| precision_rate | 100%       | —          |
| critical_rate  | 80%        | +20%       |

## Notes

- Physical and attribute penetration are **not modeled** — results are pre-penetration damage
- Simulation uses `/dev/urandom` (SystemRandom)
