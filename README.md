# WWM Damage Calculator

Công cụ tính damage cho game **Where Winds Meet**.

## Cách dùng

```bash
python calculator.py                    # stats hiện tại + active skill
python calculator.py change.cnf         # so sánh base vs override
python calculator.py -s "skill name"    # chọn skill formula
python calculator.py change.cnf -v      # kèm double marginal gain
```

## Config files

| File                | Mô tả                                      |
|---------------------|--------------------------------------------|
| `base.cnf`          | Stats nhân vật                             |
| `skill_formulas.cnf`| Công thức damage của từng skill            |
| `marginal.cnf`      | Các option upgrade ngang giá để so sánh   |

Override files (partial stats, áp đè lên base):
`affi.cnf`, `crit.cnf`, `prec.cnf`, `phys.cnf`, `bellstrike.cnf`, `bamboocut.cnf`, `pendant.cnf`, v.v.

## Công thức damage

### Raw damage

```
phys_dmg = physical_atk * phys_coeff + phys_bonus

attr_dmg = (main_attr * 1.5 + Σ other_attrs) * attr_coeff + attr_bonus
         # mystic skill: main_attr * 1.0 (không có x1.5)

total_max = phys_max + attr_max
total_min = phys_min + attr_min
E[roll]   = (total_min + total_max) / 2
```

### Effective rates (sau cap)

```
eff_affinity  = min(affinity_rate,  40%) + min(direct_affinity, 10%)
eff_precision = min(precision_rate, 100%)
eff_crit      = min(crit_rate,      80%) + min(direct_crit,     20%)
```

### Xác suất từng loại hit

```
P(orange) = eff_affinity
P(yellow) = (1 - eff_affinity) * eff_precision * eff_crit
P(white)  = (1 - eff_affinity) * eff_precision * (1 - eff_crit)
P(gray)   = (1 - eff_affinity) * (1 - eff_precision)
```

### Damage từng loại hit

| Màu    | Tên       | Damage                         | Ghi chú              |
|--------|-----------|--------------------------------|----------------------|
| ORANGE | affinity  | `total_max × affinity_mult`    | fixed, luôn là max   |
| YELLOW | crit      | `roll(min, max) × crit_mult`   | random               |
| WHITE  | normal    | `roll(min, max)`               | random               |
| GRAY   | abrasion  | `total_min`                    | fixed, luôn là min   |

### Expected damage

```
E[DMG] = P(orange) × total_max × affinity_mult
       + P(yellow) × E[roll]   × crit_mult
       + P(white)  × E[roll]
       + P(gray)   × total_min
```

## Stability stats (từ simulation)

| Stat  | Ý nghĩa                                                  |
|-------|----------------------------------------------------------|
| mean  | Trung bình damage qua n lần (converge về E[DMG] lý thuyết) |
| std   | Độ lệch chuẩn — std cao = damage không ổn định           |
| min   | Damage thấp nhất (thường là GRAY = total_min)            |
| max   | Damage cao nhất (thường là ORANGE = affinity dmg)        |
| p10   | 10% số hit thấp hơn giá trị này                         |
| p90   | 90% số hit thấp hơn giá trị này (p10~p90 = practical range) |

## Attributes

`bellstrike` · `stonesplit` · `bamboocut` · `silkbind`

## Rate caps

| Rate              | Cap thường | Cap direct |
|-------------------|-----------|------------|
| affinity_rate     | 40%       | +10%       |
| precision_rate    | 100%      | —          |
| critical_rate     | 80%       | +20%       |

## Lưu ý

- Physical và attribute penetration **chưa được mô phỏng** — kết quả là pre-penetration damage
- Simulation dùng `/dev/urandom` (SystemRandom)
