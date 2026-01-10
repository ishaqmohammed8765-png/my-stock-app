with tab_live:
    st.subheader("Live Quotes")
    st.caption("Uses Alpaca quotes via your RealtimeStream. (Quotes update only while this tab reruns.)")

    if not LIVE_AVAILABLE:
        st.info("Live module not available (or import failed).")
        st.stop()

    if not has_keys(api_key, sec_key):
        st.info("Live is disabled because Alpaca keys are missing.")
        st.stop()

    # ---------------------------
    # Stop stream if ticker changed
    # ---------------------------
    last_live_symbol = st.session_state.get("live_last_symbol")
    stream = st.session_state.get("live_stream")

    def _is_running(obj) -> bool:
        try:
            return bool(obj is not None and getattr(obj, "is_running", lambda: False)())
        except Exception:
            return False

    live_running = _is_running(stream)

    if live_running and last_live_symbol and last_live_symbol != symbol:
        # best effort stop on symbol switch
        try:
            stream.stop()
        except Exception:
            pass
        st.session_state["live_stream"] = None
        stream = None
        live_running = False

    # ---------------------------
    # Controls
    # ---------------------------
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2], vertical_alignment="center")

    start_clicked = c1.button("▶ Start", use_container_width=True, disabled=live_running)
    stop_clicked = c2.button("⏹ Stop", use_container_width=True, disabled=not live_running)
    clear_clicked = c3.button("🧹 Clear", use_container_width=True)

    st.session_state["live_autorefresh"] = c4.toggle(
        "Auto refresh",
        value=bool(st.session_state.get("live_autorefresh", True)),
    )

    if clear_clicked:
        st.session_state["live_rows"] = []

    # ---------------------------
    # Start stream
    # ---------------------------
    if start_clicked and not live_running:
        try:
            stream = RealtimeStream(api_key, sec_key, symbol)
            stream.start()
            st.session_state["live_stream"] = stream
            st.session_state["live_last_symbol"] = symbol
            live_running = True
        except Exception as e:
            st.session_state["live_stream"] = None
            st.error("Failed to start live stream.")
            st.caption(f"{type(e).__name__}: {e}")
            live_running = False

    # ---------------------------
    # Stop stream (best effort)
    # ---------------------------
    if stop_clicked and live_running:
        try:
            if stream is not None:
                stream.stop()
        except Exception:
            pass
        st.session_state["live_stream"] = None
        live_running = False

    # ---------------------------
    # Pull latest messages into buffer
    # ---------------------------
    stream = st.session_state.get("live_stream")
    if stream is not None:
        try:
            new_msgs = stream.get_latest(max_items=250)
        except Exception:
            new_msgs = []

        if new_msgs:
            st.session_state.setdefault("live_rows", [])
            st.session_state["live_rows"].extend(new_msgs)
            st.session_state["live_rows"] = st.session_state["live_rows"][-600:]  # cap memory

    # ---------------------------
    # Status
    # ---------------------------
    st.caption("Status: ✅ running (quotes)" if live_running else "Status: ⏸ stopped")

    rows = st.session_state.get("live_rows", [])
    if not rows:
        st.info("No quote updates received yet.")
    else:
        # Convert alpaca quote objects/dicts into a clean dataframe
        def to_dict(x: Any) -> dict:
            if isinstance(x, dict):
                return x
            for attr in ("model_dump", "dict"):
                m = getattr(x, attr, None)
                if callable(m):
                    try:
                        return m()
                    except Exception:
                        pass

            d: Dict[str, Any] = {}
            for k in ("symbol", "timestamp", "bid_price", "ask_price", "bid_size", "ask_size"):
                if hasattr(x, k):
                    d[k] = getattr(x, k)
            if not d:
                d["message"] = str(x)
            return d

        df_live = pd.DataFrame([to_dict(x) for x in rows])
        if df_live.empty:
            st.warning("Received live messages but could not parse into a table.")
        else:
            # Normalize common Alpaca short keys
            rename_map = {
                "bp": "bid_price",
                "ap": "ask_price",
                "bs": "bid_size",
                "as": "ask_size",
                "t": "timestamp",
                "S": "symbol",
            }
            for k, v in rename_map.items():
                if k in df_live.columns and v not in df_live.columns:
                    df_live[v] = df_live[k]

            # Timestamp parse
            if "timestamp" in df_live.columns:
                df_live["timestamp"] = pd.to_datetime(df_live["timestamp"], errors="coerce", utc=True)

            # Compute mid + spread bps
            if "bid_price" in df_live.columns and "ask_price" in df_live.columns:
                bid = pd.to_numeric(df_live["bid_price"], errors="coerce")
                ask = pd.to_numeric(df_live["ask_price"], errors="coerce")
                mid = (bid + ask) / 2.0
                spread = (ask - bid)

                df_live["mid"] = mid
                df_live["spread_bps"] = np.where(mid > 0, (spread / mid) * 10000.0, np.nan)

            # Latest snapshot metrics
            latest = df_live.dropna(subset=["bid_price", "ask_price"], how="any").tail(1) if {"bid_price","ask_price"} <= set(df_live.columns) else df_live.tail(1)
            if not latest.empty and {"bid_price", "ask_price"} <= set(latest.columns):
                lbid = float(pd.to_numeric(latest["bid_price"], errors="coerce").iloc[0])
                lask = float(pd.to_numeric(latest["ask_price"], errors="coerce").iloc[0])
                lmid = (lbid + lask) / 2.0 if np.isfinite([lbid, lask]).all() else np.nan
                lsp_bps = ((lask - lbid) / lmid) * 10000.0 if np.isfinite([lbid, lask, lmid]).all() and lmid > 0 else np.nan

                m1, m2, m3 = st.columns(3)
                m1.metric("Bid", f"{lbid:.4f}" if np.isfinite(lbid) else "—")
                m2.metric("Ask", f"{lask:.4f}" if np.isfinite(lask) else "—")
                m3.metric("Spread (bps)", f"{lsp_bps:.1f}" if np.isfinite(lsp_bps) else "—")

            # Display tidy table
            show_cols = [c for c in ["timestamp", "symbol", "bid_price", "ask_price", "mid", "spread_bps", "bid_size", "ask_size", "message"] if c in df_live.columns]
            view = df_live[show_cols] if show_cols else df_live

            if "timestamp" in view.columns:
                view = view.sort_values("timestamp").tail(160)
            else:
                view = view.tail(160)

            st.dataframe(view, use_container_width=True, height=520)

    # ---------------------------
    # Auto refresh (correct way)
    # ---------------------------
    if live_running and st.session_state.get("live_autorefresh", True):
        if st_autorefresh is not None:
            st_autorefresh(interval=750, key=f"live_refresh_{symbol}")
        else:
            st.info("Tip: install `streamlit-autorefresh` to enable auto-refresh without clicking.")
