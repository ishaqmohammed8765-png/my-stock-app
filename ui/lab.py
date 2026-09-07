"""Portfolio study workflow with an explicit final-test reveal."""

import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from utils.data_loader import MarketData, demo_market, normalise_bars, parse_symbols
from utils.portfolio import PortfolioConfig, make_panel, portfolio_metrics
from utils.services import market
from utils.strategy_lab import paper_targets, prepare_study, test_frozen


def uploaded_panel(upload):
    raw = pd.read_csv(upload)
    if set(["symbol", "date", "open", "high", "low", "close", "volume", "currency"]) - set(raw):
        raise ValueError("CSV requires symbol,date,open,high,low,close,volume,currency columns.")
    datasets = []
    for symbol, group in raw.groupby("symbol"):
        if not group.currency.eq("USD").all():
            raise ValueError("CSV quote currencies must all be USD.")
        parse_symbols(str(symbol))
        if group.date.duplicated().any():
            raise ValueError("Duplicate dates in uploaded instrument.")
        bars, warnings = normalise_bars(group.rename(columns={"date": "timestamp"}))
        datasets.append(
            MarketData(
                str(symbol),
                bars,
                "User CSV (adjustment asserted by uploader)",
                "USD",
                datetime.now(timezone.utc).isoformat(),
                warnings=warnings,
            )
        )
    return make_panel(datasets)


def strategy_lab(demo, provider, key, secret):
    st.subheader("Strategy Lab")
    st.write(
        "Compare three fixed monthly trend/momentum rules, then freeze one before revealing the final 20% of history. Unallocated capital stays in cash."
    )
    st.caption(
        "Candidates: 200-session trend; 126- and 252-session momentum with the latest 21 sessions skipped and a 200-session trend filter. USD, whole shares, no leverage."
    )
    symbols = st.text_input("Portfolio universe", "SPY, EFA, IEF, GLD, VNQ", key="lab_symbols")
    capital = st.number_input(
        "Portfolio capital (USD)", 100.0, 1000000000.0, 10000.0, key="lab_capital"
    )
    commission = st.number_input("Commission per order (USD)", 0.0, 1000.0, 1.0, key="lab_fee")
    slip = st.number_input("Slippage per side (basis points)", 0.0, 500.0, 10.0, key="lab_slip")
    spread = st.number_input("Full spread (basis points)", 0.0, 1000.0, 10.0, key="lab_spread")
    cfg = PortfolioConfig(
        capital=capital, commission=commission, slippage_bps=slip, spread_bps=spread
    )
    st.caption(
        "50% maximum target weight; a 20% closing drawdown triggers liquidation at the next open. Gaps can exceed this threshold. Cash earns 0%."
    )
    upload = st.file_uploader("Optional historical portfolio CSV", type=["csv"])
    asserted = st.checkbox("Uploaded OHLC prices use a consistent split/dividend adjustment basis")
    identity = (symbols, demo, provider, upload.getvalue() if upload else None, asserted, cfg)
    if st.session_state.get("lab_identity") != identity:
        for name in ["lab_panel", "lab_study", "lab_result"]:
            st.session_state.pop(name, None)
        st.session_state.lab_identity = identity
    if st.button("Prepare and freeze study", type="primary"):
        try:
            with st.spinner("Validating history and comparing training periods…"):
                if upload:
                    if not asserted:
                        raise ValueError("Confirm the uploaded price adjustment basis first.")
                    panel = uploaded_panel(upload)
                else:
                    tickers = parse_symbols(symbols)
                    panel = make_panel(
                        [
                            demo_market(s) if demo else market(s, 10, provider, key, secret)
                            for s in tickers
                        ]
                    )
                study = prepare_study(panel, cfg)
                st.session_state.lab_panel = panel
                st.session_state.lab_study = study
                st.session_state.pop("lab_result", None)
        except Exception as exc:
            st.error("Study could not be prepared: " + str(exc))
    if "lab_study" not in st.session_state:
        st.info("Prepare a study using 10 years of daily history, a CSV, or the synthetic demo.")
        return
    study, panel = st.session_state.lab_study, st.session_state.lab_panel
    if panel.synthetic:
        st.warning(
            "SYNTHETIC DEMO: generated prices provide no evidence of profitability. Demo instruments may share identical paths."
        )
    st.write("Frozen candidate: **" + study["manifest"]["candidate"] + "**")
    st.caption(
        "Development folds restart with the same capital and are independent experiments, not a compounded portfolio. Selection uses only the preceding 756 sessions; each forward fold lasts up to 126 sessions."
    )
    st.dataframe(study["folds"], hide_index=True)
    with st.expander("Final selection: training results only"):
        st.dataframe(study["ranking"], hide_index=True)
    st.download_button(
        "Download frozen research plan",
        json.dumps(study["manifest"], indent=2),
        "research-plan.json",
        "application/json",
    )
    st.write(
        "Reserved final period: "
        + study["manifest"]["final_start"]
        + " through "
        + study["manifest"]["final_end"]
    )
    st.warning(
        "Reusing this final period after changing rules or the universe makes it development data. Reveal history is tracked only in this browser session; retain the downloaded plan externally."
    )
    if st.button("Reveal final test once"):
        st.session_state.lab_result = test_frozen(panel, cfg, study["manifest"])
        st.session_state.lab_reveals = st.session_state.get("lab_reveals", 0) + 1
    if "lab_result" in st.session_state:
        results = st.session_state.lab_result
        st.caption(f"Final tests revealed in this session: {st.session_state.lab_reveals}")
        metrics = pd.DataFrame({name: portfolio_metrics(run) for name, run in results.items()}).T
        st.dataframe(metrics)
        st.line_chart(pd.DataFrame({name: run.equity for name, run in results.items()}))
        selected, benchmark, stressed = [metrics.loc[k, "net_return"] for k in results]
        if not panel.synthetic and selected > 0 and selected > benchmark and stressed > 0:
            st.info(
                "Passed this historical screen. Forward paper observation is still needed; this is not proof of a repeatable edge."
            )
        else:
            st.warning(
                "Profitability is not established. Synthetic data, a loss, benchmark underperformance or failed cost stress prevents passing this screen."
            )
        st.download_button(
            "Download final orders",
            results["Selected"].orders.to_csv(index=False),
            "final-orders.csv",
            "text/csv",
        )
        st.download_button(
            "Download final equity",
            pd.DataFrame({name: run.equity for name, run in results.items()}).to_csv(),
            "final-equity.csv",
            "text/csv",
        )
        st.download_button(
            "Download paper observation targets",
            paper_targets(panel, cfg, study["manifest"]).to_csv(index=False),
            "paper-targets.csv",
            "text/csv",
        )
        st.caption(
            "Paper targets are research weights, not executable instructions. Confirm current prices, instrument eligibility and portfolio halt state before any forward observation."
        )
