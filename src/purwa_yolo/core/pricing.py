"""ERP pricing logic: dataclasses + pure functions (no Streamlit).

rate per vehicle = base rate x min(period x density, max_mult) x plate multiplier

The plate multiplier is the odd-even rule, applied per vehicle AFTER the cap,
so a surcharge of x1.5 is always exactly x1.5. It is only used when plate
readings are passed in (after the OCR step).
"""
from collections import Counter
from dataclasses import dataclass, field
from datetime import date

from purwa_yolo import app_config as cfg
from purwa_yolo.core.plate_rules import (date_parity, plate_multiplier,
                                         plate_parity, today_wib)


@dataclass
class ErpSettings:
    """Everything the sidebar controls. Defaults make it easy to use in tests."""
    currency: str = "USD"
    fx_rate: float = 1.0
    base_rates: dict = field(
        default_factory=lambda: dict(cfg.DEFAULT_RATES_USD))
    period_mults: dict = field(
        default_factory=lambda: dict(cfg.DEFAULT_PERIODS))
    period: str = "Normal"
    density_on: bool = True
    busy_min: int = 5
    congested_min: int = 10
    mult_busy: float = 1.25
    mult_congested: float = 1.5
    max_mult: float = 3.0
    # odd-even plate rule
    plate_date: date = field(default_factory=today_wib)
    plate_surcharge: float = cfg.DEFAULT_PLATE_SURCHARGE
    plate_min_conf: float = cfg.DEFAULT_PLATE_MIN_CONF
    show_full_plate: bool = False


@dataclass
class ClassLine:
    key: str
    label: str
    count: int
    base_rate: float
    final_rate: float
    surcharged: int
    charge: float


@dataclass
class PlateRow:
    vehicle_id: int
    label: str
    text: str | None
    conf: float
    readable: bool
    parity: str | None
    multiplier: float


@dataclass
class ErpBreakdown:
    lines: list
    total_vehicles: int
    total_charge: float
    total_static: float
    period_mult: float
    density_level: str
    density_mult: float
    raw_mult: float
    total_mult: float
    use_plates: bool
    plate_warning: str | None
    plate_rows: list
    unread: int
    other_classes: dict


def fmt_money(usd, settings):
    """Format a USD amount in the selected currency."""
    cur = cfg.CURRENCIES[settings.currency]
    value = usd * settings.fx_rate
    return f"{cur['symbol']}{value:,.{cur['decimals']}f}"


def density_level(n_vehicles, settings):
    """Return (density level name, multiplier)."""
    if not settings.density_on:
        return "Disabled", 1.0
    if n_vehicles >= max(settings.congested_min, settings.busy_min):
        return "Congested", settings.mult_congested
    if n_vehicles >= settings.busy_min:
        return "Busy", settings.mult_busy
    return "Free-flowing", 1.0


def compute_erp(counts, density, settings, plates=None):
    """Compute the whole ERP breakdown.

    counts:  {class name: number of vehicles}
    density: vehicles used for the density level (image: all; video: most in one frame)
    plates:  list[PlateReading] after the OCR step, or None (no odd-even rule)
    """
    norm = Counter()
    for name, n in counts.items():
        norm[name.lower()] += n

    period_mult = settings.period_mults[settings.period]
    level, dens_mult = density_level(density, settings)
    raw_mult = period_mult * dens_mult
    total_mult = min(raw_mult, settings.max_mult)

    # ---- odd-even plate rule (per vehicle) ----
    use_plates = plates is not None
    plate_warning = None
    if use_plates and len(plates) != sum(norm.values()):
        use_plates = False
        plate_warning = ("Plate readings do not match the detected vehicles, "
                         "so the odd-even rule was not applied.")

    mult_sum, surcharged = Counter(), Counter()
    plate_rows, unread = [], 0
    if use_plates:
        for p in plates:
            cls = p.cls.lower()
            readable = bool(p.text) and p.conf >= settings.plate_min_conf
            parity = plate_parity(p.text) if readable else None
            mult = plate_multiplier(parity, settings.plate_date,
                                    settings.plate_surcharge)
            mult_sum[cls] += mult
            if mult > 1.0:
                surcharged[cls] += 1
            if parity is None:
                unread += 1
            plate_rows.append(PlateRow(
                vehicle_id=p.vehicle_id, label=cfg.ERP_LABELS.get(cls, cls),
                text=p.text, conf=p.conf, readable=readable, parity=parity,
                multiplier=mult))

    lines = []
    total_vehicles, total_charge, total_static = 0, 0.0, 0.0
    for key, base in settings.base_rates.items():
        n = norm.get(key, 0)
        effective = mult_sum[key] if use_plates else n
        final_rate = base * total_mult
        charge = effective * final_rate
        lines.append(ClassLine(key, cfg.ERP_LABELS[key], n, base, final_rate,
                               surcharged[key], charge))
        total_vehicles += n
        total_charge += charge
        total_static += n * base

    return ErpBreakdown(
        lines=lines, total_vehicles=total_vehicles, total_charge=total_charge,
        total_static=total_static, period_mult=period_mult,
        density_level=level, density_mult=dens_mult, raw_mult=raw_mult,
        total_mult=total_mult, use_plates=use_plates,
        plate_warning=plate_warning, plate_rows=plate_rows, unread=unread,
        other_classes={k: v for k, v in norm.items()
                       if k not in settings.base_rates})


def plate_rule_summary(settings):
    """(date parity, parity that pays the surcharge) for the selected date."""
    dp = date_parity(settings.plate_date)
    return dp, ("even" if dp == "odd" else "odd")
