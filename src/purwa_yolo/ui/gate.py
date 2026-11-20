"""Simple front gate: a one-time math question that basic bots cannot pass."""
import random
import time

import streamlit as st

from purwa_yolo.ui import theme


def _header(msg):
    theme.hero("Quick human check", msg)


def human_check():
    """Nothing below the call to this function runs until the check is passed."""
    ss = st.session_state
    if ss.get("human_ok"):
        return

    _, mid, _ = st.columns([1, 3, 1])

    # temporary lock after too many wrong answers
    if time.time() < ss.get("hc_lock_until", 0):
        with mid:
            _header("Too many wrong answers.")
            st.error("Please wait a few seconds and reload the page.")
        st.stop()

    # create a fresh question (kept on the server, never sent to the browser)
    if "hc" not in ss:
        a, b = random.randint(2, 9), random.randint(2, 9)
        op = random.choice(["+", "×"])
        ss["hc"] = {"q": f"{a} {op} {b}",
                    "ans": str(a + b if op == "+" else a * b),
                    "t": time.time()}

    with mid:
        _header("Solve one small sum to open the ERP system.")
        msg = ss.pop("hc_msg", None)
        if msg:
            st.error(msg)

        with st.form("human_check_form", border=True):
            st.markdown(f"### {ss['hc']['q']} = ?")
            reply = st.text_input("Your answer", placeholder="Type a number")
            submitted = st.form_submit_button("Continue", width="stretch")

    if submitted:
        hc = ss.pop("hc", None)
        too_fast = hc is None or time.time(
        ) - hc["t"] < 2   # bots answer instantly
        if hc is not None and reply.strip() == hc["ans"] and not too_fast:
            ss["human_ok"] = True
            st.rerun()

        ss["hc_fails"] = ss.get("hc_fails", 0) + 1
        if ss["hc_fails"] >= 3:                  # 3 wrong answers -> 30 s lock
            ss["hc_lock_until"] = time.time() + 30
            ss["hc_fails"] = 0
        else:
            ss["hc_msg"] = "That was not right. Try the new question."
        st.rerun()
    st.stop()
