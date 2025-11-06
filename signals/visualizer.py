from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go


def create_signal_tab():
    return html.Div([
        html.H3("Сигналы в реальном времени", style={'color': 'white'}),

        # Контролы
        html.Div([
            html.Label("Тикер:", style={'color': 'white'}),
            dcc.Input(id='signal-ticker', value='SPX', type='text'),

            html.Label("Чувствительность:", style={'color': 'white', 'margin-top': '10px'}),
            dcc.Slider(id='sensitivity', min=1, max=10, value=5, marks={i: str(i) for i in range(1, 11)})
        ], style={'padding': '20px', 'background': '#2a2a2a', 'border-radius': '10px'}),

        # График
        dcc.Graph(id='signals-graph'),

        # Таблица сигналов
        html.Div(id='signals-table', style={'margin-top': '20px'})
    ])


def update_signal_graph(ticker, sensitivity):
    # Получаем данные
    from detector import detect_levels
    options_data = ...  # Ваш код загрузки данных

    # Создаем базовый график
    fig = go.Figure()

    # Добавляем свечи (ваш существующий код)

    # Добавляем уровни
    levels = detect_levels(options_data, options_data['Spot'].iloc[-1])

    colors = {
        'gamma_flip': '#FF6B6B',
        'call_wall': '#4ECDC4',
        'put_wall': '#FFA07A'
    }

    for level in levels:
        fig.add_shape(
            type="line",
            x0=0, x1=1, xref="paper",
            y0=level['strike'], y1=level['strike'],
            line=dict(
                color=colors.get(level['type'], 'gray'),  # Добавлен цвет по умолчанию
                width=2
            ),  # <-- Закрываем dict и добавляем запятую
            opacity=0.7
        )

    return fig