results, trades = backtest_strategy(
    df,
    market_df=None,
    horizon=20,
    mode="pullback",  # <- FIXED (or "breakout")
    atr_entry=1.0,
    atr_stop=2.0,
    atr_target=3.0,
    require_risk_on=False,  # <- since market_df=None, this should be False
    rsi_min=30,
    rsi_max=70,
    rvol_min=1.2,
    vol_max=1.0,
    cooldown_bars=5,
    include_spread_penalty=True,
    assumed_spread_bps=5.0,
    start_equity=100000,
)
df_chart = df.copy()
add_indicators_inplace(df_chart)
st.line_chart(df_chart[['close', 'ma50', 'ma200']])
