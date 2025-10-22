import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd

# 示例数据

df = pd.read_csv("data/C9999.XDCE.10m.20251018.csv", parse_dates=["timestamps"])
df = df.sort_values("timestamps")
df['date'] = df['timestamps'].dt.date
unique_dates = sorted(df['date'].unique())
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H3(id='title', style={'textAlign': 'center'}),
    dcc.Graph(id='kline-graph'),
    html.Div([
        html.Button('上一天', id='prev-day', n_clicks=0),
        html.Button('下一天', id='next-day', n_clicks=0),
    ], style={'textAlign': 'center', 'marginTop': 10}),
    dcc.Store(id='day-index', data=0)
])

@app.callback(
    Output('kline-graph', 'figure'),
    Output('title', 'children'),
    Output('day-index', 'data'),
    Input('prev-day', 'n_clicks'),
    Input('next-day', 'n_clicks'),
    Input('day-index', 'data'),
)
def update_chart(prev_clicks, next_clicks, day_idx):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'none'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'next-day' and day_idx < len(unique_dates) - 1:
        day_idx += 1
    elif button_id == 'prev-day' and day_idx > 0:
        day_idx -= 1

    current_date = unique_dates[day_idx]
    sub = df[df['date'] == current_date]

    fig = go.Figure(data=[go.Candlestick(
        x=sub['timestamps'],
        open=sub['open'],
        high=sub['high'],
        low=sub['low'],
        close=sub['close']
    )])
    fig.update_layout(title=f"{current_date} K线图", xaxis_rangeslider_visible=False)

    return fig, f"{current_date}", day_idx

if __name__ == '__main__':
    app.run(debug=True)   # ✅ 新写法
