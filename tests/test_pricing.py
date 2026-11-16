"""ERP price calculation, including the odd-even rule (pure logic)."""
from datetime import date

from purwa_yolo.core.data import PlateReading
from purwa_yolo.core.pricing import (ErpSettings, compute_erp, density_level,
                                     fmt_money)

EVEN_DAY, ODD_DAY = date(2026, 11, 14), date(2026, 11, 15)


def close(a, b):
    return abs(a - b) < 1e-9


def plates():
    """car even, car odd, car unreadable, van odd, bus even."""
    return [
        PlateReading(1, "car", "B1234XYZ", 0.9),
        PlateReading(2, "car", "D5671AB", 0.9),
        PlateReading(3, "car", None, 0.0),
        PlateReading(4, "van", "L777AB", 0.9),
        PlateReading(5, "bus", "F1002CD", 0.9),
    ]


COUNTS = {"car": 3, "van": 1, "bus": 1}


def test_without_plates_the_charge_is_count_times_rate_times_multiplier():
    b = compute_erp(COUNTS, 5, ErpSettings(plate_date=EVEN_DAY))
    assert b.density_level == "Busy" and b.density_mult == 1.25
    assert close(b.total_charge, (3 * 2 + 3 + 5) * 1.25)
    assert close(b.total_static, 14.0)
    assert b.use_plates is False


def test_even_date_surcharges_odd_plates():
    s = ErpSettings(plate_date=EVEN_DAY, density_on=False)
    b = compute_erp(COUNTS, 5, s, plates())
    # car: 1.0 + 1.5 + 1.0 = 3.5 x 2 = 7.0 ; van odd 1.5 x 3 = 4.5 ; bus even 1.0 x 5 = 5.0
    assert close(b.total_charge, 16.5)
    assert [ln.surcharged for ln in b.lines] == [1, 1, 0]
    assert b.unread == 1


def test_odd_date_reverses_the_mechanism():
    s = ErpSettings(plate_date=ODD_DAY, density_on=False)
    b = compute_erp(COUNTS, 5, s, plates())
    # car: 1.5 + 1.0 + 1.0 = 3.5 x 2 = 7.0 ; van odd 1.0 x 3 = 3.0 ; bus even 1.5 x 5 = 7.5
    assert close(b.total_charge, 17.5)
    assert [ln.surcharged for ln in b.lines] == [1, 0, 1]


def test_cap_is_applied_before_the_plate_multiplier():
    s = ErpSettings(plate_date=EVEN_DAY, period="Peak hours", max_mult=1.5)
    b = compute_erp({"car": 1}, 10, s, [PlateReading(1, "car", "D5671AB", 0.9)])
    # peak 1.5 x congested 1.5 = 2.25 -> capped to 1.5, then plate x1.5: 2 x 1.5 x 1.5
    assert b.raw_mult > b.total_mult
    assert close(b.total_charge, 4.5)


def test_low_confidence_reading_counts_as_unreadable():
    low = [PlateReading(1, "car", "D5671AB", 0.30)]
    strict = ErpSettings(plate_date=EVEN_DAY, density_on=False, plate_min_conf=0.5)
    lenient = ErpSettings(plate_date=EVEN_DAY, density_on=False, plate_min_conf=0.2)
    b1 = compute_erp({"car": 1}, 1, strict, low)
    b2 = compute_erp({"car": 1}, 1, lenient, low)
    assert close(b1.total_charge, 2.0) and b1.unread == 1
    assert close(b2.total_charge, 3.0)


def test_plates_that_do_not_match_the_vehicles_are_ignored_with_a_warning():
    b = compute_erp({"car": 3}, 3, ErpSettings(plate_date=EVEN_DAY), [])
    assert b.use_plates is False and b.plate_warning


def test_no_vehicles():
    b = compute_erp({}, 0, ErpSettings(), [])
    assert b.total_vehicles == 0 and b.total_charge == 0


def test_unknown_class_is_reported_but_not_charged():
    b = compute_erp({"car": 1, "truck": 2}, 3, ErpSettings(), None)
    assert b.other_classes == {"truck": 2} and b.total_vehicles == 1


def test_currency_formatting():
    assert fmt_money(2, ErpSettings()) == "$2.00"
    assert fmt_money(2, ErpSettings(currency="IDR", fx_rate=16000)) == "Rp 32,000"
    assert fmt_money(2, ErpSettings(currency="CNY", fx_rate=7.2)) == "CN¥ 14.40"


def test_density_levels():
    s = ErpSettings()
    assert density_level(4, s)[0] == "Free-flowing"
    assert density_level(5, s)[0] == "Busy"
    assert density_level(10, s)[0] == "Congested"
    assert density_level(99, ErpSettings(density_on=False))[0] == "Disabled"
