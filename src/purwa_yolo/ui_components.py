"""Streamlit widgets and rendering for the ERP part of the app."""
import pandas as pd
import streamlit as st

from . import app_config as cfg
from . import plate_ocr
from .plate_rules import mask_plate, today_wib
from .pricing import ErpSettings, compute_erp, fmt_money, plate_rule_summary


def erp_settings():
    """Sidebar controls. Return an ErpSettings."""
    with st.sidebar:
        st.header("⚙️ ERP Rate Settings")

        currency = st.radio("Currency", list(cfg.CURRENCIES.keys()),
                            horizontal=True)
        cur = cfg.CURRENCIES[currency]
        fx_rate = cur["default_rate"]
        if currency != "USD":
            fx_rate = st.number_input(
                f"Exchange rate ({currency} per 1 USD), enter manually",
                min_value=0.0001, value=cur["default_rate"],
                step=cur["step"], format="%.4f")

        st.subheader("Base rates (USD)")
        base_rates = {}
        for key, default in cfg.DEFAULT_RATES_USD.items():
            base_rates[key] = st.number_input(
                f"{cfg.ERP_LABELS[key]} (USD)", min_value=0.0,
                value=default, step=0.5)

        st.subheader("Time period")
        with st.expander("Adjust period multipliers"):
            period_mults = {}
            for name, default in cfg.DEFAULT_PERIODS.items():
                period_mults[name] = st.number_input(
                    f"{name} multiplier", min_value=0.0,
                    value=default, step=0.1)
        period = st.selectbox("Current period", list(period_mults.keys()))

        st.subheader("Traffic density")
        density_on = st.checkbox("Raise rates when busy", value=True)
        busy_min, congested_min = 5, 10
        mult_busy, mult_congested = 1.25, 1.5
        if density_on:
            busy_min = st.number_input("Busy starts at (vehicles)",
                                       min_value=1, value=5, step=1)
            congested_min = st.number_input("Congested starts at (vehicles)",
                                            min_value=1, value=10, step=1)
            mult_busy = st.number_input("Busy multiplier", min_value=1.0,
                                        value=1.25, step=0.05)
            mult_congested = st.number_input("Congested multiplier",
                                             min_value=1.0, value=1.5,
                                             step=0.05)

        max_mult = st.number_input(
            "Maximum total multiplier", min_value=1.0,
            value=3.0, step=0.5,
            help="Period multiplier × density multiplier will not "
                 "exceed this value.")

        st.subheader("Odd-even plate rule")
        if plate_ocr.PLATE_OCR_AVAILABLE:
            plate_on = st.checkbox(
                "Read plates and apply the odd-even rule", value=True,
                help="Uses fast-alpr (plate detector + fast-plate-ocr). "
                     "Changing this needs a new detection run.")
        else:
            plate_on = False
            st.info("Plate reading needs fast-alpr. Install it with:\n\n"
                    "`python -m pip install \"fast-alpr[onnx]\"`")
        plate_date = st.date_input(
            "Date to apply", value=today_wib(),
            help="Even date: odd-numbered plates pay the surcharge. "
                 "Odd date: even-numbered plates pay it.")
        plate_surcharge = st.number_input(
            "Surcharge multiplier", min_value=1.0,
            value=cfg.DEFAULT_PLATE_SURCHARGE, step=0.1,
            help="Applied to vehicles whose plate number parity differs "
                 "from the date parity. Others pay ×1.0.")
        plate_min_conf = st.slider(
            "Minimum OCR confidence", 0.0, 1.0, cfg.DEFAULT_PLATE_MIN_CONF,
            0.05, help="Plates read below this confidence are treated as "
                       "unreadable (charged ×1.0).")
        show_full_plate = st.checkbox("Show full plate text in the table",
                                      value=False)

    return ErpSettings(
        currency=currency, fx_rate=fx_rate, base_rates=base_rates,
        period_mults=period_mults, period=period, density_on=density_on,
        busy_min=int(busy_min),
        congested_min=max(int(congested_min), int(busy_min)),
        mult_busy=mult_busy, mult_congested=mult_congested,
        max_mult=max_mult, plate_on=plate_on, plate_date=plate_date,
        plate_surcharge=plate_surcharge, plate_min_conf=plate_min_conf,
        show_full_plate=show_full_plate)


def render_erp(counts, density, density_basis, settings, plates=None):
    """Compute and display the ERP breakdown."""
    b = compute_erp(counts, density, settings, plates)
    if b.plate_warning:
        st.warning(b.plate_warning)

    rows = []
    for ln in b.lines:
        row = {"Type": ln.label, "Count": ln.count,
               "Base rate": fmt_money(ln.base_rate, settings),
               "Final rate": fmt_money(ln.final_rate, settings)}
        if b.use_plates:
            row["Surcharged"] = ln.surcharged
        row["Subtotal"] = fmt_money(ln.charge, settings)
        rows.append(row)
    total_row = {"Type": "TOTAL", "Count": b.total_vehicles,
                 "Base rate": "", "Final rate": ""}
    if b.use_plates:
        total_row["Surcharged"] = sum(ln.surcharged for ln in b.lines)
    total_row["Subtotal"] = fmt_money(b.total_charge, settings)
    rows.append(total_row)

    st.subheader("🚦 ERP Calculation (dynamic pricing)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total vehicles", b.total_vehicles)
    diff = b.total_charge - b.total_static
    if abs(diff) > 1e-9:
        sign = "+" if diff > 0 else "-"
        c2.metric("Total charge", fmt_money(b.total_charge, settings),
                  delta=f"{sign}{fmt_money(abs(diff), settings)} vs base rate",
                  delta_color="inverse")
    else:
        c2.metric("Total charge", fmt_money(b.total_charge, settings))
    c3.metric("Rate multiplier", f"×{b.total_mult:.2f}")

    st.table(pd.DataFrame(rows).set_index("Type"))

    st.caption(
        f"Period: **{settings.period}** (×{b.period_mult:.2f}) · "
        f"Density: **{b.density_level}** (×{b.density_mult:.2f}; {density} "
        f"vehicles {density_basis}) · Total multiplier ×{b.total_mult:.2f}")
    if b.raw_mult > settings.max_mult:
        st.caption(f"Multiplier capped from ×{b.raw_mult:.2f} to "
                   f"×{settings.max_mult:.2f} (maximum limit).")

    if b.use_plates:
        date_par, hit_par = plate_rule_summary(settings)
        st.caption(
            f"Odd-even rule for **{settings.plate_date:%d %b %Y}** "
            f"({date_par} date): {hit_par}-numbered plates "
            f"×{settings.plate_surcharge:.2f}, {date_par}-numbered plates "
            f"×1.00. Unreadable plates are charged ×1.00 ({b.unread} "
            "vehicle(s)). The plate multiplier is applied on top of the rate "
            "shown in 'Final rate'.")
        if b.plate_rows:
            with st.expander("Plate readings"):
                st.dataframe(pd.DataFrame([{
                    "Type": r.label,
                    "Plate": (mask_plate(r.text, settings.show_full_plate)
                              if r.readable else "—"),
                    "OCR confidence": f"{r.conf:.2f}" if r.text else "—",
                    "Parity": r.parity or "unread",
                    "Multiplier": f"×{r.multiplier:.2f}",
                } for r in b.plate_rows]), hide_index=True)
                st.caption("Plates are masked unless 'Show full plate text' "
                           "is ticked in the sidebar.")

    if b.other_classes:
        st.caption("Classes without a rate (not counted): "
                   + ", ".join(f"{k} ({v})" for k, v in b.other_classes.items()))
    if b.total_vehicles == 0:
        st.info("No vehicles detected. Try lowering the confidence threshold.")


def show_counts(counts, title, total_label):
    """Generic counts for non-vehicle use cases."""
    if counts:
        st.subheader(title)
        st.metric(label=total_label, value=sum(counts.values()))
        cols = st.columns(len(counts))
        for col, (name, n) in zip(cols, counts.items()):
            col.metric(label=name, value=n)
    else:
        st.info("No objects detected. Try lowering the confidence threshold.")
