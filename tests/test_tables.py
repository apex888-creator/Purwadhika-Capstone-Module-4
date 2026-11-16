"""HTML tables used instead of pandas."""
from purwa_yolo.ui.tables import html_table


def test_header_and_row_count():
    html = html_table([{"Type": "Car", "Count": 2}, {"Type": "Van", "Count": 1}])
    assert html.count("<th ") == 2 and html.count("<tr>") == 3     # 2 header cells; header row + 2 rows
    assert ">Car<" in html and ">2<" in html


def test_values_are_html_escaped():
    html = html_table([{"Plate": "<script>alert(1)</script>", "Note": "A&B"}])
    assert "<script>" not in html and "&lt;script&gt;" in html and "A&amp;B" in html


def test_masked_plates_and_currency_symbols_stay_readable():
    html = html_table([{"Plate": "B ***4 ***", "Total": "$9.50", "Mult": "×1.50"}])
    assert "B ***4 ***" in html and "$9.50" in html and "×1.50" in html


def test_last_row_can_be_bold():
    html = html_table([{"Type": "Car"}, {"Type": "TOTAL"}], highlight_last=True)
    rows = html.split("<tr>")
    assert "font-weight:700" not in rows[2] and "font-weight:700" in rows[3]


def test_empty_table():
    assert html_table([]) == ""
