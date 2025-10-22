import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np

df = pd.read_csv("data/C9999.XDCE.10m.20251018.csv", parse_dates=["timestamps"])
df = df.sort_values("timestamps")
df['date'] = df['timestamps'].dt.date


# ========= 聚合成日线 =========
daily = df.groupby("date").agg({
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum"
}).reset_index()

# ========= Dash app =========
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H3("多级K线联动图", style={"textAlign": "center"}),
    html.Div([
        html.H4("日线K线图（点击查看详情）"),
        dcc.Graph(id="daily-chart", clear_on_unhover=True)
    ]),
    html.Div([
        html.H4(id="sub-title"),
        dcc.Graph(id="intraday-chart")
    ])
])

# --- 初始绘制日线图 ---
@app.callback(
    Output("daily-chart", "figure"),
    Input("daily-chart", "id")  # 仅用于初始化
)
def render_daily(_):
    fig = go.Figure(data=[go.Candlestick(
        x=daily["date"],
        open=daily["open"],
        high=daily["high"],
        low=daily["low"],
        close=daily["close"],
        name="日线"
    )])
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        height=400
    )
    return fig

# --- 点击日线后更新下方分钟图 ---
@app.callback(
    Output("intraday-chart", "figure"),
    Output("sub-title", "children"),
    Input("daily-chart", "clickData")
)
def update_intraday(clickData):
    if not clickData:
        # 默认显示最近一天
        selected_date = daily["date"].iloc[-1]
    else:
        selected_date = clickData["points"][0]["x"]

    sub = df[df["date"] == pd.to_datetime(selected_date).date()]

    fig = go.Figure(data=[go.Candlestick(
        x=sub["timestamps"],
        open=sub["open"],
        high=sub["high"],
        low=sub["low"],
        close=sub["close"],
        name="分钟K线"
    )])
    fig.update_layout(
        title=f"{selected_date} 分钟级K线",
        xaxis_rangeslider_visible=False,
        height=400
    )
    return fig, f"📅 {selected_date} 当日分钟K线"

if __name__ == "__main__":
    app.run(debug=True)
