"""Rendering of results: counts, ERP charge table and plate readings."""
import streamlit as st

from purwa_yolo.core.plate_rules import mask_plate
from purwa_yolo.core.pricing import fmt_money, plate_rule_summary
from purwa_yolo.ui import theme
from purwa_yolo.ui.tables import show_table


def show_counts(counts, title, total_label):
    """Generic counts (non-vehicle use cases)."""
    if counts:
        st.subheader(title)
        st.metric(label=total_label, value=sum(counts.values()))
        cols = st.columns(len(counts))
        for col, (name, n) in zip(cols, counts.items()):
            col.metric(label=name, value=n)
    else:
        st.info("No objects detected. Try lowering the confidence threshold.")


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

    st.subheader(title)
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

    st.write("")
    show_table(rows, highlight_last=True)
    st.write("")

    # what was applied, as chips
    theme.chips(
        (f"Period: {settings.period} (×{b.period_mult:.2f})", ""),
        (f"Density: {b.density_level} (×{b.density_mult:.2f}), "
         f"{density} vehicles {density_basis}",
         _density_kind(b.density_level)),
        (f"Total ×{b.total_mult:.2f}", "warn"),
    )
    if b.raw_mult > settings.max_mult:
        st.caption(f"Multiplier capped from ×{b.raw_mult:.2f} to "
                   f"×{settings.max_mult:.2f} (maximum limit).")

    if b.use_plates:
        date_par, hit_par = plate_rule_summary(settings)
        st.info(
            f"**Odd-even rule for {settings.plate_date:%d %b %Y}** "
            f"({date_par} date): {hit_par}-numbered plates "
            f"×{settings.plate_surcharge:.2f}, {date_par}-numbered plates "
            f"×1.00. Unreadable plates are charged ×1.00 ({b.unread} "
            "vehicle(s)). The plate multiplier is applied on top of the rate "
            "shown in 'Final rate'.", icon="🔢")

    if b.other_classes:
        st.caption("Classes without a rate (not counted): "
                   + ", ".join(f"{k} ({v})" for k, v in b.other_classes.items()))
    if b.total_vehicles == 0:
        st.info("No vehicles detected. Try lowering the confidence threshold.")


def show_plate_table(b, settings):
    """Plate readings per vehicle (masked unless the user asks for the full text)."""
    if not b.plate_rows:
        return
    st.markdown("#### Plate readings")
    show_table([{
        "Vehicle": f"#{r.vehicle_id}",
        "Type": r.label,
        "Plate": (mask_plate(r.text, settings.show_full_plate)
                  if r.readable else "—"),
        "OCR confidence": f"{r.conf:.2f}" if r.text else "—",
        "Parity": r.parity or "unread",
        "Multiplier": f"×{r.multiplier:.2f}",
    } for r in b.plate_rows])
    st.caption("Vehicle numbers match the labels on the detection result. "
               "Plates are masked unless 'Show full plate text' is ticked "
               "in the sidebar.")
    st.write("")
