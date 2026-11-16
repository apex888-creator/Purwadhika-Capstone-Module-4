"""Sidebar: ERP rate settings (returns an ErpSettings)."""
import streamlit as st

from purwa_yolo import app_config as cfg
from purwa_yolo.core.plate_rules import today_wib
from purwa_yolo.core.pricing import ErpSettings


def erp_settings():
    """Draw the sidebar controls and return the chosen settings."""
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
            "Maximum total multiplier", min_value=1.0, value=3.0, step=0.5,
            help="Period multiplier × density multiplier will not "
                 "exceed this value.")

        st.subheader("Odd-even plate rule")
        st.caption("Used after you press 'Read Plates' (section 2).")
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
        max_mult=max_mult, plate_date=plate_date,
        plate_surcharge=plate_surcharge, plate_min_conf=plate_min_conf,
        show_full_plate=show_full_plate)
