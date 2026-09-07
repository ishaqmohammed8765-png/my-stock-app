import json
import math
from dataclasses import asdict
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from utils.backtester import position_size
from utils.data_loader import is_stale, parse_symbols
from utils.evaluation import evaluate
from utils.screener import screen_row
from utils.services import compliance, market, news
from utils.state import result_key
from utils.strategy import assess

from .charts import line_chart, price_chart


def number(value, suffix="", decimals=2):
    try:
        return f"{value:,.{decimals}f}{suffix}" if math.isfinite(value) else "Unknown"
    except (TypeError, ValueError):
        return "Unknown"


def overview(data, bars, strategy, execution, zoya_key, refresh):
    setup = assess(bars.iloc[-1], strategy)
    blocked = is_stale(data) or data.currency != "USD"
    left, right = st.columns([1.8, 1])
    with left:
        st.subheader("Current setup")
        if blocked:
            st.warning("Setup unavailable: data is stale or USD quote currency is unverified.")
        elif setup.eligible:
            st.success("Rules qualify · conditional entry")
        else:
            st.info("Wait · conditions not met")
        for reason in setup.reasons:
            st.write(reason)
        st.plotly_chart(price_chart(bars), width="stretch", key="overview_price")
    with right:
        st.subheader("Trade planning")
        st.caption(
            "Indicative levels from the last completed session. A gap changes the simulated fill and protective levels."
        )
        for label, value in [
            ("Entry trigger", setup.entry),
            ("Indicative stop", setup.stop),
            ("Indicative target", setup.target),
        ]:
            st.metric(label + " (USD)", number(value) if not blocked else "Unavailable")
        if setup.eligible and not blocked:
            qty = position_size(execution.capital, setup.entry, setup.stop, execution)
            st.caption(
                f"Indicative size: {qty} whole shares, subject to the actual fill, fees and risk cap."
            )
            if qty == 0:
                st.warning(
                    "The capital or risk budget cannot support one share with these settings."
                )
            plan = {
                "type": "synthetic_demo" if data.demo else "paper_trade_plan",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "symbol": data.symbol,
                "currency": data.currency,
                "data_through": str(bars.index[-1].date()),
                "source": data.source,
                "identity": result_key(data, strategy, execution),
                "setup": asdict(setup),
                "indicative_quantity": qty,
                "strategy": asdict(strategy),
                "execution": asdict(execution),
                "note": "Conditional plan only. No order is placed. Recheck after each completed session; record actual paper fills and costs separately.",
            }
            st.download_button(
                "Export paper-trade plan",
                json.dumps(plan, indent=2),
                f"{'DEMO-' if data.demo else ''}{data.symbol}-paper-plan.json",
                "application/json",
            )
    st.markdown("#### Company context")
    meta = data.metadata
    cols = st.columns(3)
    cap, shares, short = (
        meta.get("marketCap"),
        meta.get("floatShares"),
        meta.get("shortPercentOfFloat"),
    )
    cols[0].metric(
        "Market cap (USD B)",
        number(cap / 1e9 if data.currency == "USD" and isinstance(cap, (float, int)) else None),
    )
    cols[1].metric(
        "Float (million shares)", number(shares / 1e6 if isinstance(shares, (float, int)) else None)
    )
    cols[2].metric(
        "Short interest", number(short * 100 if isinstance(short, (float, int)) else None, "%")
    )
    st.caption(
        "Current provider metadata, not historical point-in-time data. Missing values are never replaced with scoring assumptions."
    )
    with st.expander("Shariah compliance · Zoya"):
        st.caption(
            "Optional current stock report. This does not establish historical compliance during a backtest."
        )
        if data.demo:
            st.info("Compliance checks are disabled for synthetic data.")
        elif not zoya_key.startswith("live-"):
            st.info(
                "Configure ZOYA_API_KEY with a live key. Sandbox results cannot verify a stock."
            )
        elif st.button("Check current compliance"):
            report = compliance(data.symbol, zoya_key, refresh)
            st.write(f"{report.symbol}: {report.status}")
            st.caption(f"Report date: {report.report_date or 'Unknown'}")
            if report.error:
                st.warning(report.error)


def charts(bars):
    st.subheader("Price & participation")
    st.caption("Last 252 completed sessions. Historical prices are adjusted for corporate actions.")
    st.plotly_chart(price_chart(bars, detailed=True), width="stretch")
    d = bars.tail(252)
    a, b = st.columns(2)
    with a:
        st.markdown("#### Momentum · RSI")
        st.plotly_chart(line_chart({"RSI (14)": d.rsi14}, 260), width="stretch")
    with b:
        st.markdown("#### Trend strength · ADX")
        st.plotly_chart(line_chart({"ADX (14)": d.adx14}, 260), width="stretch")
    st.markdown("#### MACD")
    st.plotly_chart(
        line_chart({"MACD": d.macd, "Signal": d.macd_signal, "Histogram": d.macd_hist}, 280),
        width="stretch",
    )


def research(data, bars, strategy, execution):
    st.subheader("Test the evidence")
    st.write(
        "The first 200 sessions initialise indicators. Of the remaining history, the first 70% is the development period and the final 30% is the holdout."
    )
    st.caption(
        "Choose rules before inspecting the holdout. Repeated tuning against it makes it another development sample. This app does not fit or optimise a model."
    )
    if data.currency != "USD":
        st.warning(
            "Evaluation is blocked until USD quote currency is verified. No implicit currency conversion is performed."
        )
        return
    identity = result_key(data, strategy, execution)
    if st.button("Run historical evaluation", type="primary"):
        st.session_state.pop("evaluation", None)
        try:
            with st.spinner("Checking the holdout, costs and separate time windows…"):
                result = evaluate(bars, strategy, execution)
            st.session_state["evaluation"] = (identity, result)
        except ValueError as exc:
            st.error(str(exc))
    stored = st.session_state.get("evaluation")
    if not stored or stored[0] != identity:
        st.info(
            "Run an evaluation for this ticker, dataset and settings. Previous results are not displayed or exported under new inputs."
        )
        return
    result = stored[1]
    st.markdown(f"### {'Demo result · ' if data.demo else ''}{result.label}")
    for reason in result.reasons:
        st.write(f"• {reason}")
    st.caption(
        f"Holdout starts {result.split_date}. Figures below refer only to this later period; open positions are valued at each daily close."
    )
    m = result.holdout.metrics
    for col, label, value in zip(
        st.columns(4),
        ["Holdout return", "Worst drawdown", "Closed trades", "Net expectancy / trade (USD)"],
        [
            f"{m['total_return']:.1%}",
            f"{m['max_drawdown']:.1%}",
            str(m["trades"]),
            number(m["expectancy"]),
        ],
    ):
        col.metric(label, value)
    st.plotly_chart(
        line_chart(
            {
                "Strategy": result.holdout.equity,
                "Buy & hold": result.benchmark,
                "Cost stress": result.stress.equity,
            }
        ),
        width="stretch",
        key="equity_comparison",
    )
    st.caption(
        "USD account equity. Buy & hold invests the available account in the same stock; the strategy retains cash according to its risk and allocation caps. Different exposure is intentional."
    )
    a, b = st.columns(2)
    with a:
        st.metric("Cost-stress holdout return", f"{result.stress.metrics['total_return']:.1%}")
        st.caption(
            "Spread, slippage and commission increase up to 2×, capped at supported limits; zero stays zero. This is a sensitivity check, not a worst-case scenario."
        )
    with b:
        lo, hi = result.interval
        st.metric(
            "Exploratory expectancy interval (USD)",
            f"{number(lo)} to {number(hi)}" if math.isfinite(lo) else "Too few trades",
        )
        st.caption(
            "95% moving-block bootstrap interval of observed net trade P&L. It does not account for every dependency, regime change or strategy-selection bias."
        )
    st.markdown("#### Stability across separate holdout windows")
    st.dataframe(
        result.windows,
        hide_index=True,
        width="stretch",
        column_config={
            "Strategy return": st.column_config.NumberColumn(format="percent"),
            "Buy & hold return": st.column_config.NumberColumn(format="percent"),
        },
    )
    st.caption(
        "Each window starts flat with the same capital and closes at its end. Window returns do not sum to the continuous holdout return."
    )
    with st.expander("Development comparison, drawdown & trade details"):
        st.dataframe(
            pd.DataFrame(
                [result.development.metrics, result.holdout.metrics],
                index=["Development", "Holdout"],
            ),
            width="stretch",
        )
        dd = result.holdout.equity / result.holdout.equity.cummax() - 1
        st.plotly_chart(line_chart({"Holdout drawdown": dd}, 250), width="stretch")
        st.dataframe(result.holdout.trades, hide_index=True, width="stretch")
        st.json(result.holdout.diagnostics)
    if m["trades"] == 0:
        st.warning(
            "No holdout trades. Rules, whole-share affordability or the risk cap may prevent entries; do not loosen them merely to obtain a good-looking backtest."
        )
    trades = result.holdout.trades.assign(
        symbol=data.symbol, currency="USD", dataset_id=identity, synthetic=data.demo
    )
    st.download_button(
        "Download holdout trades",
        trades.to_csv(index=False),
        f"{'DEMO-' if data.demo else ''}{data.symbol}-holdout-trades.csv",
        "text/csv",
    )
    record = {
        "identity": identity,
        "symbol": data.symbol,
        "source": data.source,
        "currency": data.currency,
        "synthetic": data.demo,
        "fetched_at": data.fetched_at,
        "strategy": asdict(strategy),
        "execution": asdict(execution),
        "holdout_start": result.split_date,
        "evidence": result.label,
        "reasons": result.reasons,
    }
    st.download_button(
        "Download research record",
        json.dumps(record, indent=2),
        f"{'DEMO-' if data.demo else ''}{data.symbol}-research-record.json",
        "application/json",
    )
    with st.expander("Simulation assumptions & limits"):
        st.write(
            "One long position at a time; no leverage, shorting or fractional shares. Prior-close signals create next-session orders. Stops are checked on the entry day. Opening gaps precede intraday ranges, and scheduled time exits occur at the open. Ambiguous daily paths use a conservative stop-first policy; an intraday entry does not receive an unproven target fill. Commissions apply only to filled orders and are included in every trade statistic."
        )
        st.write(
            "Adjusted prices provide a corporate-action-adjusted research proxy, not a reconstruction of a broker's split/dividend ledger. No taxes, FX conversion, idle-cash interest or partial-fill model are included. Daily data cannot reconstruct the intraday path. Sharpe uses a zero cash-rate assumption. Gaps and changing spreads can create larger losses than the planned risk cap."
        )


def screener(strategy, zoya_key, refresh, demo):
    st.subheader("Find setups to investigate")
    st.caption(
        "Uses the same entry rules as the simulator. Qualifying is not a probability of profit. Current universe selection can introduce survivorship bias."
    )
    if demo:
        st.info("The live screener is disabled in synthetic demo mode.")
        return
    raw = st.text_input("Symbols (up to 20)", "AAPL, MSFT, AMZN, NVDA, META", key="scan_symbols")
    limit_cap = st.checkbox("Apply a market-cap ceiling")
    cap = (
        st.number_input("Maximum market cap (USD B)", min_value=0.1, value=5.0)
        if limit_cap
        else None
    )
    halal = st.checkbox("Only verified Zoya-compliant stocks")
    signature = (raw, str(strategy), cap, halal, refresh)
    if st.button("Scan completed sessions", type="primary"):
        st.session_state.pop("scan", None)
        if halal and not zoya_key.startswith("live-"):
            st.error("A live Zoya key is required. Unknown reports are excluded.")
            return
        try:
            symbols = parse_symbols(raw)
        except ValueError as exc:
            st.error(str(exc))
            return
        rows, excluded = [], []
        progress = st.progress(0)
        for i, symbol in enumerate(symbols):
            try:
                row = screen_row(market(symbol, 2, "Yahoo", "", "", refresh), strategy, cap)
                if halal:
                    report = compliance(symbol, zoya_key, refresh)
                    if report.status != "COMPLIANT":
                        excluded.append({"Symbol": symbol, "Reason": report.error or report.status})
                        continue
                    row["Compliance report date"] = report.report_date
                rows.append(row)
            except Exception:
                excluded.append(
                    {"Symbol": symbol, "Reason": "Data unavailable or failed validation."}
                )
            finally:
                progress.progress((i + 1) / len(symbols))
        progress.empty()
        st.session_state["scan"] = (signature, rows, excluded)
    saved = st.session_state.get("scan")
    if saved and saved[0] == signature:
        rows, excluded = saved[1:]
        if rows:
            frame = pd.DataFrame(rows).sort_values(["Setup", "Symbol"])
            st.dataframe(frame, hide_index=True, width="stretch")
            st.download_button(
                "Download screen", frame.to_csv(index=False), "stock-screen.csv", "text/csv"
            )
        else:
            st.info("No results passed the requested filters.")
        if excluded:
            st.caption("Excluded or unavailable symbols")
            st.dataframe(pd.DataFrame(excluded), hide_index=True, width="stretch")


def news_view(data, refresh):
    st.subheader("Headlines for context")
    st.caption(
        "Read the source and date. Headlines are not converted into a sentiment score or profit forecast."
    )
    if data.demo:
        st.info("Live news is disabled for synthetic data.")
        return
    try:
        with st.spinner("Loading headlines…"):
            items = news(data.symbol, refresh)
    except Exception:
        st.warning("News is unavailable. Price analysis can still be used.")
        return
    if not items:
        st.info("No headlines were returned.")
    for item in items:
        with st.container(border=True):
            st.link_button(item["title"], item["url"])
            st.caption(f"{item['publisher']} · {item['date']}")
