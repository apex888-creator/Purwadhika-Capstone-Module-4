"""Rendering of results: counts, ERP charge table and plate readings."""
import streamlit as st

from purwa_yolo.core.plate_rules import mask_plate
from purwa_yolo.core.pricing import fmt_money, plate_rule_summary
from purwa_yolo.ui import theme
from purwa_yolo.ui.i18n import fmt_date, t, tr
from purwa_yolo.ui.tables import show_table


def show_counts(counts, title, total_label):
    """Generic counts (non-vehicle use cases)."""
    if counts:
        st.subheader(title)
        st.metric(label=total_label, value=sum(counts.values()))
        cols = st.columns(len(counts))
        for col, (name, n) in zip(cols, counts.items()):
            col.metric(label=tr(name), value=n)
    else:
        st.info(t("no_objects"))


def _density_kind(level):
    level = str(level).lower()
    if "congest" in level:
        return "bad"
    if "busy" in level:
        return "warn"
    return "ok"


def show_erp(b, settings, density, density_basis, title):
    """Show an ErpBreakdown: metrics, table and the rules that were applied."""
    if b.plate_warning:
        st.warning(tr(b.plate_warning))

    k_type, k_count = t("col_type"), t("col_count")
    k_base, k_final = t("col_base"), t("col_final")
    k_sur, k_sub = t("col_surcharged"), t("col_subtotal")

    rows = []
    for ln in b.lines:
        row = {k_type: tr(ln.label), k_count: ln.count,
               k_base: fmt_money(ln.base_rate, settings),
               k_final: fmt_money(ln.final_rate, settings)}
        if b.use_plates:
            row[k_sur] = ln.surcharged
        row[k_sub] = fmt_money(ln.charge, settings)
        rows.append(row)
    total_row = {k_type: t("col_total"), k_count: b.total_vehicles,
                 k_base: "", k_final: ""}
    if b.use_plates:
        total_row[k_sur] = sum(ln.surcharged for ln in b.lines)
    total_row[k_sub] = fmt_money(b.total_charge, settings)
    rows.append(total_row)

    st.subheader(title)
    c1, c2, c3 = st.columns(3)
    c1.metric(t("total_vehicles"), b.total_vehicles)
    diff = b.total_charge - b.total_static
    if abs(diff) > 1e-9:
        sign = "+" if diff > 0 else "-"
        c2.metric(t("total_charge"), fmt_money(b.total_charge, settings),
                  delta=t("delta_vs_base", sign=sign,
                          amt=fmt_money(abs(diff), settings)),
                  delta_color="inverse")
    else:
        c2.metric(t("total_charge"), fmt_money(b.total_charge, settings))
    c3.metric(t("rate_mult"), f"×{b.total_mult:.2f}")

    st.write("")
    show_table(rows, highlight_last=True)
    st.write("")

    # what was applied, as chips
    theme.chips(
        (t("chip_period", p=tr(settings.period), m=b.period_mult), ""),
        (t("chip_density", lvl=tr(b.density_level), m=b.density_mult,
           n=density, basis=tr(density_basis)),
         _density_kind(b.density_level)),
        (t("chip_total", m=b.total_mult), "warn"),
    )
    if b.raw_mult > settings.max_mult:
        st.caption(t("capped", raw=b.raw_mult, cap=settings.max_mult))

    if b.use_plates:
        date_par, hit_par = plate_rule_summary(settings)
        st.info(
            t("oddeven_info",
              date=fmt_date(settings.plate_date),
              dp=tr(date_par), hp=tr(hit_par),
              sur=settings.plate_surcharge, unread=b.unread),
            icon="🔢")

    if b.other_classes:
        st.caption(t("other_classes")
                   + ", ".join(f"{tr(k)} ({v})"
                               for k, v in b.other_classes.items()))
    if b.total_vehicles == 0:
        st.info(t("no_vehicles_found"))


def show_plate_table(b, settings):
    """Plate readings per vehicle (masked unless the user asks for the full text)."""
    if not b.plate_rows:
        return
    st.markdown(f"#### {t('plate_readings')}")
    show_table([{
        t("col_vehicle"): f"#{r.vehicle_id}",
        t("col_type"): tr(r.label),
        t("col_plate"): (mask_plate(r.text, settings.show_full_plate)
                         if r.readable else "—"),
        t("col_ocr_conf"): f"{r.conf:.2f}" if r.text else "—",
        t("col_parity"): tr(r.parity or "unread"),
        t("col_mult"): f"×{r.multiplier:.2f}",
    } for r in b.plate_rows])
    st.caption(t("plate_note"))
    st.write("")
