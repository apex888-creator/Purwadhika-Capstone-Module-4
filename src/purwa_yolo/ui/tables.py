"""Simple HTML tables, so the app does not need to import pandas.

(pandas ships compiled DLLs; on some Windows laptops Application Control blocks
a freshly installed pandas, and this app only needed it to draw two tables.)

The look (zebra rows, hover, highlighted TOTAL row) lives in ui/theme.py
under the `.erp-table` class.
"""
from html import escape

import streamlit as st


def html_table(rows, highlight_last=False):
    """Build a table from a list of dicts that all share the same keys.
    Every value is HTML-escaped. highlight_last styles the last row (TOTAL)."""
    if not rows:
        return ""
    headers = list(rows[0])
    head = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
    body = []
    for i, row in enumerate(rows):
        cls = ' class="total"' if highlight_last and i == len(rows) - 1 else ""
        cells = "".join(f"<td>{escape(str(row.get(h, '')))}</td>"
                        for h in headers)
        body.append(f"<tr{cls}>{cells}</tr>")
    return ('<div style="overflow-x:auto"><table class="erp-table">'
            f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody>"
            "</table></div>")


def show_table(rows, highlight_last=False):
    # st.html does not receive the global CSS reliably in every version,
    # st.markdown with unsafe_allow_html does.
    st.markdown(html_table(rows, highlight_last), unsafe_allow_html=True)
