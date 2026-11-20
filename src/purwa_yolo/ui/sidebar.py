"""Sidebar: ERP rate settings (returns an ErpSettings).

Every widget has a fixed key= so its value survives a language switch
(Streamlit would otherwise treat a relabelled widget as a new one).
"""
import streamlit as st

from purwa_yolo import app_config as cfg
from purwa_yolo.core.plate_rules import today_wib
from purwa_yolo.core.pricing import ErpSettings
from purwa_yolo.ui import theme
from purwa_yolo.ui.i18n import lang_button, t, tr


def erp_settings():
    """Draw the sidebar controls and return the chosen settings."""
    with st.sidebar:
        lang_button("lang_sidebar")
        theme.brand(t("sb_title"))
        st.caption(t("sb_caption"))

        # ---- currency ----
        currency = st.radio(t("currency"), list(cfg.CURRENCIES.keys()),
                            horizontal=True, key="sb_currency")
        cur = cfg.CURRENCIES[currency]
        fx_rate = cur["default_rate"]
        if currency != "USD":
            fx_rate = st.number_input(
                t("fx", cur=currency),
                min_value=0.0001, value=cur["default_rate"],
                step=cur["step"], format="%.4f", key=f"sb_fx_{currency}")

        # ---- base rates ----
        with st.expander(t("sb_rates"), expanded=True):
            base_rates = {}
            for key, default in cfg.DEFAULT_RATES_USD.items():
                base_rates[key] = st.number_input(
                    t("rate_label", name=tr(cfg.ERP_LABELS[key])),
                    min_value=0.0, value=default, step=0.5,
                    key=f"sb_rate_{key}")

        # ---- time period ----
        with st.expander(t("sb_period"), expanded=True):
            period_mults = {}
            with st.popover(t("adj_period"), width="stretch"):
                for name, default in cfg.DEFAULT_PERIODS.items():
                    period_mults[name] = st.number_input(
                        t("period_mult", name=tr(name)), min_value=0.0,
                        value=default, step=0.1, key=f"sb_pm_{name}")
            period = st.selectbox(t("cur_period"), list(period_mults.keys()),
                                  format_func=tr, key="sb_period")

        # ---- traffic density ----
        with st.expander(t("sb_density"), expanded=False):
            density_on = st.checkbox(t("raise_busy"), value=True,
                                     key="sb_density_on")
            busy_min, congested_min = 5, 10
            mult_busy, mult_congested = 1.25, 1.5
            if density_on:
                c1, c2 = st.columns(2)
                busy_min = c1.number_input(t("busy_from"), min_value=1,
                                           value=5, step=1, key="sb_busy_min")
                congested_min = c2.number_input(t("cong_from"), min_value=1,
                                                value=10, step=1,
                                                key="sb_cong_min")
                mult_busy = c1.number_input(t("busy_x"), min_value=1.0,
                                            value=1.25, step=0.05,
                                            key="sb_busy_x")
                mult_congested = c2.number_input(t("cong_x"), min_value=1.0,
                                                 value=1.5, step=0.05,
                                                 key="sb_cong_x")

            max_mult = st.number_input(
                t("max_mult"), min_value=1.0, value=3.0, step=0.5,
                help=t("max_mult_help"), key="sb_max_mult")

        # ---- odd-even rule ----
        with st.expander(t("sb_oddeven"), expanded=False):
            st.caption(t("sb_oddeven_cap"))
            plate_date = st.date_input(
                t("plate_date"), value=today_wib(),
                help=t("plate_date_help"), key="sb_plate_date")
            plate_surcharge = st.number_input(
                t("surcharge"), min_value=1.0,
                value=cfg.DEFAULT_PLATE_SURCHARGE, step=0.1,
                help=t("surcharge_help"), key="sb_surcharge")
            plate_min_conf = st.slider(
                t("min_conf"), 0.0, 1.0, cfg.DEFAULT_PLATE_MIN_CONF, 0.05,
                help=t("min_conf_help"), key="sb_min_conf")
            show_full_plate = st.checkbox(t("show_full"), value=False,
                                          key="sb_show_full")

    return ErpSettings(
        currency=currency, fx_rate=fx_rate, base_rates=base_rates,
        period_mults=period_mults, period=period, density_on=density_on,
        busy_min=int(busy_min),
        congested_min=max(int(congested_min), int(busy_min)),
        mult_busy=mult_busy, mult_congested=mult_congested,
        max_mult=max_mult, plate_date=plate_date,
        plate_surcharge=plate_surcharge, plate_min_conf=plate_min_conf,
        show_full_plate=show_full_plate)
