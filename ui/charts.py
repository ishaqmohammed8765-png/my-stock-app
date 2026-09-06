import plotly.graph_objects as go
from plotly.subplots import make_subplots

TEAL, BLUE, RED = "#55d6be", "#8caaff", "#f18e9c"


def style(fig, height=430):
    return fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=12, r=12, t=25, b=15),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08),
        font=dict(family="sans-serif", color="#dce5ef"),
        xaxis_rangeslider_visible=False,
    )


def price_chart(df, *, detailed=False):
    d = df.tail(252)
    fig = make_subplots(
        rows=2 if detailed else 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.78, 0.22] if detailed else [1],
    )
    fig.add_trace(
        go.Candlestick(
            x=d.index,
            open=d.open,
            high=d.high,
            low=d.low,
            close=d.close,
            name="Adjusted price",
            increasing_line_color=TEAL,
            decreasing_line_color=RED,
        ),
        row=1,
        col=1,
    )
    for col, name, color in [("ma50", "MA50", BLUE), ("ma200", "MA200", "#d8bf7d")]:
        fig.add_trace(
            go.Scatter(x=d.index, y=d[col], name=name, line=dict(color=color, width=1.5)),
            row=1,
            col=1,
        )
    if detailed:
        for col in ["bb_upper", "bb_lower"]:
            fig.add_trace(
                go.Scatter(
                    x=d.index,
                    y=d[col],
                    name=col.replace("_", " "),
                    line=dict(color="#6e839c", width=1, dash="dot"),
                ),
                row=1,
                col=1,
            )
        fig.add_trace(
            go.Bar(
                x=d.index,
                y=d.volume,
                name="Volume",
                marker_color=[TEAL if c >= o else RED for c, o in zip(d.close, d.open)],
            ),
            row=2,
            col=1,
        )
    return style(fig, 560 if detailed else 410)


def line_chart(series, height=350):
    fig = go.Figure()
    for (name, values), color in zip(series.items(), [TEAL, BLUE, RED, "#d8bf7d"]):
        fig.add_trace(
            go.Scatter(x=values.index, y=values.values, name=name, line=dict(color=color, width=2))
        )
    return style(fig, height)
