"""Pure helpers for plates and the odd-even rule (no Streamlit, no OpenCV)."""
import re
from datetime import datetime, timedelta, timezone

from purwa_yolo.app_config import UTC_OFFSET_HOURS

# format Plate: Izinkan 1-6 huruf di depan, 1-4 angka, dan 0-6 huruf di belakang
PLATE_RE = re.compile(r"^([A-Z]{1,6})(\d{1,4})([A-Z]{0,6})$")


def today_wib():
    """Today's date in Indonesia Western Time (UTC+7)."""
    return datetime.now(timezone(timedelta(hours=UTC_OFFSET_HOURS))).date()


def clean_plate(text):
    """Uppercase and keep letters/digits only ('b 1234-xyz' -> 'B1234XYZ')."""
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def plate_parity(text):
    """'odd' or 'even' from the last digit of the plate NUMBER
    (B 1234 XYZ -> 4 -> even). None if the text is not a valid plate."""
    m = PLATE_RE.match(clean_plate(text)) if text else None
    if not m:
        return None
    return "odd" if int(m.group(2)[-1]) % 2 else "even"


def plate_shape(text):
    """Rough layout label from a cleaned plate text.

    Because clean_plate() strips spaces, we classify from the pattern:
      - "single"  : matches the standard 1-line pattern (1-2 letters, 1-4
                    digits, 0-3 letters) -> typical single-row plate
      - "double"  : longer than a standard plate (likely a 2-row plate that
                    got concatenated, e.g. "B1234XYZ" from "B 1234\nXYZ")
      - "unknown" : does not look like an Indonesian plate at all
    """
    if not text:
        return "unknown"
    clean = clean_plate(text)
    if PLATE_RE.match(clean):
        return "single"
    if len(clean) > 9:
        return "double"
    return "unknown"


def date_parity(date_obj):
    """'odd' or 'even' from the day of the month."""
    return "odd" if date_obj.day % 2 else "even"


def plate_multiplier(parity, date_obj, surcharge):
    """Odd-even rule: the surcharge applies when the plate parity differs from
    the date parity (even date -> odd plates pay more; odd date -> even plates
    pay more). Plates that match the date, and unreadable plates, pay x1.0."""
    if parity is None:
        return 1.0
    return surcharge if parity != date_parity(date_obj) else 1.0


def mask_plate(text, show_full=False):
    """Privacy: keep the area code and the last digit only (B ***4 ***)."""
    if not text:
        return "—"
    clean = clean_plate(text)
    m = PLATE_RE.match(clean)
    if m:
        area, digits, suffix = m.groups()
        if not show_full:
            digits = "*" * (len(digits) - 1) + digits[-1]
            suffix = "*" * len(suffix)
        return " ".join(p for p in (area, digits, suffix) if p)
    return clean if show_full else "*" * len(clean)


def best_plate(votes):
    """votes: {text: [sum_conf, n]} -> (text, mean_conf) of the best-supported
    reading, or None."""
    if not votes:
        return None
    text, (total, n) = max(votes.items(), key=lambda kv: kv[1][0])
    return text, total / n
