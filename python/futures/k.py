import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import pandas as pd

# ====== 数据加载与聚合 ======
df = pd.read_csv("data/C9999.XDCE.10m.20251018.csv", parse_dates=["timestamps"])
df = df.sort_values("timestamps")
df['date'] = df['timestamps'].dt.date

daily = df.groupby("date").agg({
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum"
}).reset_index()

# ====== Dash app ======
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H3("多级K线联动图", style={"textAlign": "center"}),

    # --- 日期控制区 ---
    html.Div([
        html.Button("⬅ 上一天", id="prev-day", n_clicks=0),
        html.Span(id="current-date-label", style={"margin": "0 15px", "fontWeight": "bold"}),
        html.Button("下一天 ➡", id="next-day", n_clicks=0),
    ], style={"textAlign": "center", "marginBottom": "10px"}),

    # --- 日线图 ---
    html.Div([
        dcc.Graph(id="daily-chart", clear_on_unhover=True)
    ]),

    # --- 分钟图 ---
    html.Div([
        html.H4(id="sub-title"),
        dcc.Graph(id="intraday-chart")
    ]),

    # --- 当前选中日期（隐藏状态）---
    dcc.Store(id="selected-date", data=str(daily["date"].iloc[0]))
])

# ====== 初始化日线图 ======
@app.callback(
    Output("daily-chart", "figure"),
    Input("selected-date", "data")
)
def render_daily(selected_date):
    fig = go.Figure(data=[go.Candlestick(
        x=daily["date"],
        open=daily["open"],
        high=daily["high"],
        low=daily["low"],
        close=daily["close"],
        name="日线"
    )])

    # 高亮当前选中日期
    selected_date = pd.to_datetime(selected_date).date()
    if selected_date in daily["date"].values:
        selected_row = daily[daily["date"] == selected_date].iloc[0]
        fig.add_trace(go.Scatter(
            x=[selected_date],
            y=[selected_row["close"]],
            mode="markers+text",
            text=["⬆"],
            textposition="top center",
            marker=dict(size=12, color="red"),
            name="选中日期"
        ))

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        height=400,
        title=None
    )
    return fig

# ====== 点击图表或按钮切换日期 ======
@app.callback(
    Output("selected-date", "data"),
    Input("daily-chart", "clickData"),
    Input("prev-day", "n_clicks"),
    Input("next-day", "n_clicks"),
    State("selected-date", "data")
)
def update_selected_date(clickData, prev_clicks, next_clicks, current_date):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_date

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    dates = list(daily["date"])
    current_date = pd.to_datetime(current_date).date()

    if triggered_id == "daily-chart" and clickData:
        return clickData["points"][0]["x"]

    idx = dates.index(current_date)
    if triggered_id == "prev-day" and idx > 0:
        return str(dates[idx - 1])
    elif triggered_id == "next-day" and idx < len(dates) - 1:
        return str(dates[idx + 1])

    return str(current_date)

# ====== 下方分钟图随选中日期更新 ======
@app.callback(
    Output("intraday-chart", "figure"),
    Output("sub-title", "children"),
    Output("current-date-label", "children"),
    Input("selected-date", "data")
)
def update_intraday(selected_date):
    selected_date = pd.to_datetime(selected_date).date()
    sub = df[df["date"] == selected_date]

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
        height=400,
    )
    return fig, f"📅 {selected_date} 当日分钟K线", f"当前日期：{selected_date}"

# ====== 运行 ======
if __name__ == "__main__":
    app.run(debug=True)
