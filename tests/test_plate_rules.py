"""Plate parsing, masking and the odd-even rule (pure logic, no heavy imports)."""
from datetime import date

from purwa_yolo.core import plate_rules as pr

EVEN_DAY, ODD_DAY = date(2026, 11, 14), date(2026, 11, 15)


def test_parity_uses_last_digit_of_the_number():
    assert pr.plate_parity("B 1234 XYZ") == "even"
    assert pr.plate_parity("d-5671-ab") == "odd"
    assert pr.plate_parity("L777AB") == "odd"
    assert pr.plate_parity("B1230A") == "even"          # 0 is even
    assert pr.plate_parity("B1234") == "even"           # no suffix is fine


def test_invalid_plates_have_no_parity():
    assert pr.plate_parity("HELLO") is None
    assert pr.plate_parity("B12345XYZ") is None         # too many digits
    assert pr.plate_parity("") is None
    assert pr.plate_parity(None) is None


def test_ocr_noise_is_stripped_before_parsing():
    assert pr.clean_plate("b 12?4-xyz") == "B124XYZ"
    assert pr.plate_parity("B 12?4 XYZ") == "even"


def test_date_parity():
    assert pr.date_parity(EVEN_DAY) == "even"
    assert pr.date_parity(ODD_DAY) == "odd"
    assert pr.date_parity(date(2026, 11, 30)) == "even"
    assert pr.date_parity(date(2026, 12, 1)) == "odd"


def test_odd_even_rule_on_an_even_date():
    assert pr.plate_multiplier("odd", EVEN_DAY, 1.5) == 1.5
    assert pr.plate_multiplier("even", EVEN_DAY, 1.5) == 1.0


def test_odd_even_rule_is_reversed_on_an_odd_date():
    assert pr.plate_multiplier("even", ODD_DAY, 1.5) == 1.5
    assert pr.plate_multiplier("odd", ODD_DAY, 1.5) == 1.0


def test_unreadable_plate_is_never_surcharged():
    assert pr.plate_multiplier(None, EVEN_DAY, 1.5) == 1.0
    assert pr.plate_multiplier(None, ODD_DAY, 1.5) == 1.0


def test_custom_surcharge_is_respected():
    assert pr.plate_multiplier("odd", EVEN_DAY, 2.0) == 2.0


def test_masking_keeps_area_code_and_last_digit_only():
    assert pr.mask_plate("B1234XYZ") == "B ***4 ***"
    assert pr.mask_plate("B1234") == "B ***4"
    masked = pr.mask_plate("B1234XYZ")
    assert "123" not in masked and "XYZ" not in masked


def test_full_plate_only_on_request():
    assert pr.mask_plate("B1234XYZ", show_full=True) == "B 1234 XYZ"


def test_masking_invalid_text_hides_everything():
    assert pr.mask_plate("HELLO WORLD!") == "**********"
    assert pr.mask_plate(None) == "—"


def test_best_plate_picks_best_supported_reading():
    votes = {"B1234XYZ": [1.8, 2], "B1284XYZ": [0.6, 1]}
    text, conf = pr.best_plate(votes)
    assert text == "B1234XYZ" and abs(conf - 0.9) < 1e-9


def test_best_plate_without_votes():
    assert pr.best_plate({}) is None
    assert pr.best_plate(None) is None
