import hashlib
from dataclasses import asdict

import streamlit as st

from utils.config import Execution, Strategy, export_settings, import_settings


def settings():
    with st.sidebar:
        st.markdown("### Research settings")
        uploaded = st.file_uploader(
            "Import settings",
            type=["json"],
            help="Version 2 exports only; credentials are never included.",
        )
        if uploaded is not None:
            raw = uploaded.getvalue()
            digest = hashlib.sha256(raw).hexdigest()
            if st.session_state.get("imported_digest") != digest:
                try:
                    s, e = import_settings(raw.decode())
                    for prefix, obj in [("s_", s), ("e_", e)]:
                        for name, value in asdict(obj).items():
                            st.session_state[prefix + name] = value
                    st.session_state["imported_digest"] = digest
                except (ValueError, TypeError, UnicodeError) as exc:
                    st.error(str(exc))
        for prefix, obj in [("s_", Strategy()), ("e_", Execution())]:
            for name, value in asdict(obj).items():
                st.session_state.setdefault(prefix + name, value)
        with st.expander("Strategy rules"):
            st.selectbox("Entry style", ["breakout", "pullback"], key="s_mode")
            st.caption("All setups require price > MA50 > MA200.")
            for name, label, low, high in [
                ("rsi_min", "RSI minimum", 0.0, 100.0),
                ("rsi_max", "RSI maximum", 0.0, 100.0),
                ("rvol_min", "Minimum relative volume", 0.0, 10.0),
                ("vol_max", "Maximum annual volatility (1 = 100%)", 0.01, 5.0),
                ("adx_min", "Minimum ADX", 0.0, 100.0),
                ("atr_entry", "Entry distance (ATR)", 0.0, 5.0),
                ("atr_stop", "Stop distance (ATR)", 0.1, 20.0),
                ("atr_target", "Target distance (ATR)", 0.1, 50.0),
            ]:
                st.number_input(label, min_value=low, max_value=high, key="s_" + name)
            st.number_input("Maximum holding sessions", min_value=1, max_value=200, key="s_horizon")
        with st.expander("Account & execution"):
            st.caption("USD account and USD instruments. No GBP conversion. Whole shares only.")
            for name, label, low, high in [
                ("capital", "Starting capital (USD)", 1.0, 1e8),
                ("risk_pct", "Planned risk fraction (0.01 = 1%)", 0.001, 0.10),
                ("allocation_pct", "Maximum allocation fraction", 0.01, 1.0),
                ("fixed_amount", "Fixed investment cap (USD; 0 = off)", 0.0, 1e8),
                ("slippage_bps", "Slippage (bps, each side)", 0.0, 500.0),
                ("spread_bps", "Full spread (bps)", 0.0, 1000.0),
                ("commission", "Commission per filled order (USD)", 0.0, 1000.0),
            ]:
                st.number_input(label, min_value=low, max_value=high, key="e_" + name)
            st.caption(
                "Sizing includes estimated fees and stop costs. Gaps can exceed the planned loss."
            )
        try:
            strategy = Strategy(**{k: st.session_state["s_" + k] for k in asdict(Strategy())})
            execution = Execution(**{k: st.session_state["e_" + k] for k in asdict(Execution())})
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        st.download_button(
            "Export settings",
            export_settings(strategy, execution),
            "research-settings.json",
            "application/json",
            width="stretch",
        )
        st.caption("Rules are exploratory defaults, not optimised recommendations.")
    return strategy, execution
