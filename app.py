import yfinance as yf
import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objs as go
import pandas as pd
from datetime import timedelta
from flask_caching import Cache
from urllib.parse import parse_qs
import os
from datetime import datetime

# Инициализация Dash приложения
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Настройка кэширования
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory',
    'CACHE_THRESHOLD': 100
})
cache.clear()

def load_allowed_users():
    try:
        with open('users.txt', 'r') as file:
            users = [line.strip() for line in file if line.strip()]
            return users
    except FileNotFoundError:
        print("Файл users.txt не найден. Создайте файл и добавьте пользователей.")
        return []
    except Exception as e:
        print(f"Ошибка при чтении файла users.txt: {e}")
        return []


def read_max_power_spx():
    try:
        with open('Max Power SPX.txt', 'r') as file:
            content = file.read().strip()
            if content:
                return float(content)
    except (FileNotFoundError, ValueError):
        pass
    return None


def load_expirations_from_file(ticker):
    file_map = {
        "^SPX": "spx_oi_data.txt",
        "SPY": "spy_oi_data.txt",
        "QQQ": "qqq_oi_data.txt",
        "^NDX": "ndx_oi_data.txt",
        "NVDA": "nvda_oi_data.txt",
        "VIX": "nvda_oi_data.txt"
    }

    file_name = file_map.get(ticker)
    if not file_name:
        return None

    try:
        if os.path.exists(file_name):
            with open(file_name, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                if not lines:
                    return None

                # Парсим даты из файла
                file_dates = set()
                for line in lines:
                    if line.strip():
                        parts = line.strip().split('\t')
                        if parts:  # Первая часть - дата
                            date_str = parts[0]
                            try:
                                dt = datetime.strptime(date_str, '%a %b %d %Y')
                                file_dates.add(dt.strftime('%Y-%m-%d'))
                            except:
                                continue
                return sorted(file_dates) if file_dates else None
    except Exception as e:
        print(f"Ошибка при чтении файла {file_name}: {e}")
    return None


def get_yfinance_expirations(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.options
    except Exception as e:
        print(f"Ошибка загрузки данных {ticker} из yfinance: {e}")
        return []


# Функция для загрузки данных SPX OI из файла
def load_spx_oi_data():
    try:
        file_path = 'spx_oi_data.txt'
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.readlines()
        print(f"Файл {file_path} не найден")
    except Exception as e:
        print(f"Ошибка при чтении файла spx_oi_data.txt: {e}")
    return None


# Функция для загрузки данных SPY OI из файла
def load_spy_oi_data():
    try:
        file_path = 'spy_oi_data.txt'
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.readlines()
        print(f"Файл {file_path} не найден")
    except Exception as e:
        print(f"Ошибка при чтении файла spy_oi_data.txt: {e}")
    return None


# Функция для загрузки данных QQQ OI из файла
def load_qqq_oi_data():
    try:
        file_path = 'qqq_oi_data.txt'
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.readlines()
        print(f"Файл {file_path} не найден")
    except Exception as e:
        print(f"Ошибка при чтении файла qqq_oi_data.txt: {e}")
    return None


# Функция для загрузки данных NDX OI из файла
def load_ndx_oi_data():
    try:
        file_path = 'ndx_oi_data.txt'
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.readlines()
        print(f"Файл {file_path} не найден")
    except Exception as e:
        print(f"Ошибка при чтении файла ndx_oi_data.txt: {e}")
    return None


# Функция для загрузки данных NDX OI из файла
def load_vix_oi_data():
    try:
        file_path = 'vix_oi_data.txt'
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.readlines()
        print(f"Файл {file_path} не найден")
    except Exception as e:
        print(f"Ошибка при чтении файла vix_oi_data.txt: {e}")
    return None


# Функция для загрузки данных NVDA OI из файла
def load_nvda_oi_data():
    try:
        file_path = 'nvda_oi_data.txt'
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.readlines()
        print(f"Файл {file_path} не найден")
    except Exception as e:
        print(f"Ошибка при чтении файла nvda_oi_data.txt: {e}")
    return None


def convert_expiration_date(expiration_date):
    """Конвертирует дату экспирации из формата 'YYYY-MM-DD' в 'Day Month DD YYYY'"""
    try:
        dt = datetime.strptime(expiration_date, '%Y-%m-%d')
        return dt.strftime('%a %b %d %Y')
    except:
        return None


# Функция для преобразования тикеров
def normalize_ticker(ticker):
    index_map = {
        "SPX": "^SPX", "NDX": "^NDX", "RUT": "^RUT", "DIA": "^DIA",
        "SPY": "SPY", "QQQ": "QQQ", "DIA": "DIA", "XSP": "^XSP", "IWM": "IWM", "VIX": "^VIX"
    }
    return index_map.get(ticker.upper(), ticker.upper())


# Функция получения данных по опционам с кэшированием
@cache.memoize(timeout=60)
def get_yfinance_options(ticker, expiration):
    """Кешированная функция для получения данных опционов из yfinance"""
    try:
        stock = yf.Ticker(ticker)
        option_chain = stock.option_chain(expiration)
        calls = option_chain.calls[['strike', 'openInterest', 'volume']].rename(
            columns={'openInterest': 'Call OI', 'volume': 'Call Volume'})
        puts = option_chain.puts[['strike', 'openInterest', 'volume']].rename(
            columns={'openInterest': 'Put OI', 'volume': 'Put Volume'})
        return calls.merge(puts, on='strike', how='outer').sort_values(by='strike')
    except Exception as e:
        print(f"Ошибка загрузки данных из yfinance для {expiration}: {e}")
        return None


@cache.memoize(timeout=60)
def get_yfinance_spot_price(ticker):
    """Кешированная функция для получения текущей цены из yfinance"""
    try:
        stock = yf.Ticker(ticker)
        if ticker == "^SPX":
            xsp_ticker = yf.Ticker("^XSP")
            if xsp_ticker.history(period="1d").shape[0] > 0:
                xsp_price = xsp_ticker.history(period="1d")['Close'].iloc[-1]
                return xsp_price * 10  # Умножаем цену XSP на 10

            # Стандартная логика для других тикеров
        stock = yf.Ticker(ticker)
        if stock.history(period="1d").shape[0] > 0:
            return stock.history(period="1d")['Close'].iloc[-1]
    except Exception as e:
        print(f"Ошибка загрузки текущей цены для {ticker}: {e}")
    return None


def get_option_data(ticker, expirations):
    ticker = normalize_ticker(ticker)

    try:
        stock = yf.Ticker(ticker)
        available_dates = stock.options
        print(f"Доступные даты экспирации для {ticker}: {available_dates}")
    except Exception as e:
        print(f"Ошибка загрузки данных {ticker}: {e}")
        return None, [], None, None

    if not available_dates:
        print(f"Нет доступных дат экспирации для {ticker}")
        return None, [], None, None

    if not expirations:
        expirations = [available_dates[0]]

    all_options_data = []

    # Загрузка ручных данных для соответствующих тикеров
    manual_data_dict = {}
    if ticker == "^SPX":
        manual_data = load_spx_oi_data()
    elif ticker == "SPY":
        manual_data = load_spy_oi_data()
    elif ticker == "QQQ":
        manual_data = load_qqq_oi_data()
    elif ticker == "^NDX":
        manual_data = load_ndx_oi_data()
    elif ticker == "^VIX":
        manual_data = load_vix_oi_data()
    elif ticker == "NVDA":
        manual_data = load_nvda_oi_data()
    else:
        manual_data = None

    if manual_data:
        def parse_date(date_str):
            try:
                return datetime.strptime(date_str, '%a %b %d %Y').strftime('%Y-%m-%d')
            except:
                return None

        # Helper function to safely convert to int
        def safe_int(value):
            try:
                return int(value.replace(',', '')) if value.strip() else 0
            except:
                return 0

        # Парсим данные из файла
        for line in manual_data:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) >= 6:  # Новый формат с volume
                    date_str = parts[0]
                    call_oi = parts[1]
                    strike = parts[2]
                    put_oi = parts[3]
                    call_volume = parts[4]
                    put_volume = parts[5]

                    parsed_date = parse_date(date_str)
                    if parsed_date:
                        if parsed_date not in manual_data_dict:
                            manual_data_dict[parsed_date] = []
                        manual_data_dict[parsed_date].append({
                            'Call OI': safe_int(call_oi),
                            'strike': float(strike),
                            'Put OI': safe_int(put_oi),
                            'Call Volume': safe_int(call_volume),
                            'Put Volume': safe_int(put_volume)
                        })
                elif len(parts) >= 4:  # Старый формат без volume
                    date_str = parts[0]
                    call_oi = parts[1]
                    strike = parts[2]
                    put_oi = parts[3]

                    parsed_date = parse_date(date_str)
                    if parsed_date:
                        if parsed_date not in manual_data_dict:
                            manual_data_dict[parsed_date] = []
                        manual_data_dict[parsed_date].append({
                            'Call OI': safe_int(call_oi),
                            'strike': float(strike),
                            'Put OI': safe_int(put_oi),
                            'Call Volume': 0,  # По умолчанию 0
                            'Put Volume': 0  # По умолчанию 0
                        })

    # Обрабатываем запрошенные даты экспирации
    for expiration in expirations:
        if expiration in manual_data_dict:
            print(f"Используем ручные данные для {expiration} (тикер: {ticker})")
            df = pd.DataFrame(manual_data_dict[expiration])

            # Если volume не было в файле (все нули), пробуем загрузить из yfinance
            if df['Call Volume'].sum() == 0 and df['Put Volume'].sum() == 0:
                try:
                    # Используем кешированную функцию для получения volume
                    volume_data = get_yfinance_options(ticker, expiration)
                    if volume_data is not None:
                        volume_data = volume_data[['strike', 'Call Volume', 'Put Volume']]
                        df = df.merge(volume_data, on='strike', how='left')
                        df['Call Volume'] = df['Call Volume_y'].fillna(0)
                        df['Put Volume'] = df['Put Volume_y'].fillna(0)
                        df.drop(['Call Volume_x', 'Put Volume_x', 'Call Volume_y', 'Put Volume_y'],
                                axis=1, inplace=True, errors='ignore')
                except Exception as e:
                    print(f"Ошибка загрузки volume из yfinance: {e}")

            all_options_data.append(df)
        else:
            # Полная загрузка из yfinance с использованием кеширования
            print(f"Загружаем данные для {expiration} из yfinance (тикер: {ticker})")
            options_data = get_yfinance_options(ticker, expiration)
            if options_data is not None:
                all_options_data.append(options_data)

    if not all_options_data:
        print("Нет данных по опционам")
        return None, available_dates, None, None

    combined_data = pd.concat(all_options_data).groupby("strike", as_index=False).sum()

    # Получаем текущую цену с использованием кеширования
    spot_price = get_yfinance_spot_price(ticker)

    if spot_price:
        # Для SPX используем специальный расчет с ценой XSP*10
        if ticker == "^SPX":
            combined_data['Net GEX'] = (
                (combined_data['Call OI'] * spot_price / 100 * spot_price * 0.001) -
                (combined_data['Put OI'] * spot_price / 100 * spot_price * 0.001)
            ).round(1)

            combined_data['AG'] = (
                (combined_data['Call OI'] * spot_price / 100 * spot_price * 0.005) * 8 +
                (combined_data['Put OI'] * spot_price / 100 * spot_price * 0.005) * 8
            ).round(1)
        else:
            # Стандартный расчет для других тикеров
            combined_data['Net GEX'] = (
                (combined_data['Call OI'] * spot_price / 100 * spot_price * 0.001) -
                (combined_data['Put OI'] * spot_price / 100 * spot_price * 0.001)
            ).round(1)

            combined_data['AG'] = (
                (combined_data['Call OI'] * spot_price / 100 * spot_price * 0.005) * 8 +
                (combined_data['Put OI'] * spot_price / 100 * spot_price * 0.005) * 8
            ).round(1)

    return combined_data, available_dates, spot_price, stock


# Функция для расчета статических уровней (без изменений)
def calculate_static_levels(options_data, spot_price):
    # Уровни сопротивления
    resistance_levels = []

    # Максимальные значения AG выше текущей цены
    ag_above_spot = options_data[options_data['strike'] > spot_price]
    if not ag_above_spot.empty:
        max_ag_strike = ag_above_spot.loc[ag_above_spot['AG'].idxmax(), 'strike']
        resistance_levels.append(('AG', max_ag_strike))

    # Максимальные положительные значения Net GEX выше текущей цены
    net_gex_above_spot = options_data[(options_data['strike'] > spot_price) & (options_data['Net GEX'] > 0)]
    if not net_gex_above_spot.empty:
        max_net_gex_strike = net_gex_above_spot.loc[net_gex_above_spot['Net GEX'].idxmax(), 'strike']
        resistance_levels.append(('Net GEX', max_net_gex_strike))

    # Уровни поддержки
    support_levels = []

    # Максимальные значения AG ниже текущей цены
    ag_below_spot = options_data[options_data['strike'] < spot_price]
    if not ag_below_spot.empty:
        max_ag_strike = ag_below_spot.loc[ag_below_spot['AG'].idxmax(), 'strike']
        support_levels.append(('AG', max_ag_strike))

    # Максимальные отрицательные значения Net GEX ниже текущей цены
    net_gex_below_spot = options_data[(options_data['strike'] < spot_price) & (options_data['Net GEX'] < 0)]
    if not net_gex_below_spot.empty:
        max_net_gex_strike = net_gex_below_spot.loc[net_gex_below_spot['Net GEX'].idxmin(), 'strike']
        support_levels.append(('Net GEX', max_net_gex_strike))

    # Объединение уровней, если они находятся близко друг к другу (в пределах 20 пунктов)
    def merge_levels(levels):
        merged = []
        for level in sorted(levels, key=lambda x: x[1]):
            if merged and abs(level[1] - merged[-1][1]) <= 20:
                merged[-1] = ('Merged', min(merged[-1][1], level[1]), max(merged[-1][1], level[1]))
            else:
                merged.append(level)
        return merged

    resistance_levels = merge_levels(resistance_levels)
    support_levels = merge_levels(support_levels)

    return resistance_levels, support_levels


# Функция для добавления статических уровней на график (без изменений)
def add_static_levels_to_chart(fig, resistance_levels, support_levels, market_open_time, market_close_time):
    # Параметры зон
    resistance_zone_lower_percent = -0.00045
    resistance_zone_upper_percent = 0.0002
    support_zone_lower_percent = -0.0002
    support_zone_upper_percent = 0.00045

    # Добавление зон сопротивления
    for level in resistance_levels:
        if isinstance(level[1], tuple):  # Если уровень объединен
            lower, upper = level[1]
        else:  # Если уровень одиночный
            lower = level[1] * (1 + resistance_zone_lower_percent)
            upper = level[1] * (1 + resistance_zone_upper_percent)

        fig.add_trace(go.Scatter(
            x=[market_open_time, market_close_time, market_close_time, market_open_time],
            y=[lower, lower, upper, upper],
            fill="toself",
            fillcolor="rgba(0, 255, 216, 0.2)",  # Полупрозрачный синий
            line=dict(color="rgba(0, 255, 216, 0.2)"),
            mode="lines",
            name=f'Resistance Zone',
            hoverinfo="none",
        ))

    # Добавление зон поддержки
    for level in support_levels:
        if isinstance(level[1], tuple):  # Если уровень объединен
            lower, upper = level[1]
        else:  # Если уровень одиночный
            lower = level[1] * (1 + support_zone_lower_percent)
            upper = level[1] * (1 + support_zone_upper_percent)

        fig.add_trace(go.Scatter(
            x=[market_open_time, market_close_time, market_close_time, market_open_time],
            y=[lower, lower, upper, upper],
            fill="toself",
            fillcolor="rgba(153, 50, 50, 0.2)",  # Полупрозрачный оранжевый
            line=dict(color="rgba(153, 50, 50, 0.5)"),
            mode="lines",
            name=f'Support Zone',
            hoverinfo="none",
        ))

    return fig


oi_volume_page = html.Div([
    html.H1("Open Interest / Volume", style={'textAlign': 'center'}),

    html.Div([
        html.Label(""),
        dcc.Input(id='ticker-input-oi-volume', type='text', value='SPX', className='dash-input'),
        html.Button('Search', id='search-button-oi-volume', n_clicks=0, className='dash-button', style={'margin-left': '10px'}),
    ], className='dash-container', style={'display': 'flex', 'align-items': 'center'}),

    html.Div([
        html.Label("Expiration:"),
        dcc.Dropdown(id='date-dropdown-oi-volume', multi=True, className='dash-dropdown'),
    ], className='dash-container'),

    html.Div([
        html.Label("Parameters:"),
        html.Div([
            html.Button("Volume spread", id="btn-volume-spread", className="parameter-button active"),
            html.Button("Call OI", id="btn-call-oi-oi-volume", className="parameter-button"),
            html.Button("Put OI", id="btn-put-oi-oi-volume", className="parameter-button"),
            html.Button("Call Volume", id="btn-call-vol-oi-volume", className="parameter-button"),
            html.Button("Put Volume", id="btn-put-vol-oi-volume", className="parameter-button"),
        ], className="button-container", style={'display': 'flex', 'flex-wrap': 'wrap', 'gap': '10px'}),
    ], className='dash-container'),

    dcc.Store(id='selected-params-oi-volume', data=['Volume Spread']),
    dcc.Store(id='options-data-store-oi-volume'),

    dcc.Graph(
        id='oi-volume-chart',
        style={'height': '900px', 'border-radius': '12px',
               'box-shadow': '0 4px 10px rgba(0, 0, 0, 0.3)',
               'overflow': 'hidden',
               'background-color': '#1e1e1e',
               'padding': '10px',
               'margin-bottom': '20px'},
        config={
            'displayModeBar': False,
            'scrollZoom': False,
            'dragmode': False
        }
    ),

    dcc.Graph(
        id='oi-volume-price-chart',
        style={'height': '950px', 'border-radius': '12px',
               'box-shadow': '0 4px 10px rgba(0, 0, 0, 0.3)',
               'overflow': 'hidden',
               'background-color': '#1e1e1e',
               'padding': '10px',
               'margin-bottom': '20px'},
        config={
            'displayModeBar': False,
            'scrollZoom': False,
            'dragmode': False
        }
    )
])

# Лейаут для страницы "Options Summary"
options_summary_page = html.Div(
    className='options-summary-page',
    children=[
        html.H1("P/C Ratio", style={'textAlign': 'center', 'color': 'white'}),

        html.Div(
            dash_table.DataTable(
                id='options-summary-table',
                columns=[
                    {'name': 'Ticker', 'id': 'Ticker'},
                    {'name': 'Price', 'id': 'Price'},
                    {'name': 'Resistance', 'id': 'Resistance'},
                    {'name': 'Support', 'id': 'Support'},
                    {'name': 'Call OI Amount', 'id': 'Call OI Amount'},
                    {'name': 'Put OI Amount', 'id': 'Put OI Amount'},
                    {'name': 'P/C Ratio', 'id': 'P/C Ratio'}
                ],
                # Основные настройки
                editable=False,
                row_selectable='none',
                cell_selectable=False,
                style_as_list_view=True,  # Убирает полосы между строками

                # Стилизация
                style_table={
                    'overflowX': 'auto',
                    'borderRadius': '12px',
                    'boxShadow': '0 4px 10px rgba(0, 0, 0, 0.3)',
                    'backgroundColor': '#1e1e1e',
                    'pointerEvents': 'none'  # Полное отключение взаимодействия
                },
                style_header={
                    'backgroundColor': '#1e1e1e',
                    'color': 'white',
                    'fontWeight': 'bold',
                    'border': 'none'
                },
                style_cell={
                    'backgroundColor': '#2d2d2d',
                    'color': 'white',
                    'padding': '10px',
                    'textAlign': 'center',
                    'border': 'none',
                    'cursor': 'default',
                    'pointerEvents': 'none'  # Отключает события мыши для ячеек
                },
                style_data={
                    'border': 'none',
                    'pointerEvents': 'none'  # Отключает события мыши для данных
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#252525'
                    },
                    {
                        'if': {'column_id': 'P/C Ratio'},
                        'fontWeight': 'bold'
                    }
                ],

                # Дополнительные параметры для полного отключения взаимодействия
                active_cell=None,
                selected_cells=None,

            ),
            style={'margin-bottom': '20px'}
        ),
        dcc.Location(id='summary-url', refresh=False)
    ],
    style={
        'margin-left': '10%',
        'padding': '20px',
        'color': 'white'
    }
)

# Лейаут для страницы "How to use GEX"
how_to_use_gex_page = html.Div(
    className='how-to-use-gex-page',
    children=[
        html.H1("How To Use GEX", style={'textAlign': 'center', 'color': 'white'}),

        # Video links section with buttons
        html.Div([
            html.H2("Video Tutorials", style={'color': '#00ffcc', 'textAlign': 'center', 'margin-bottom': '30px'}),

            # Button container
            html.Div([
                # Long/Short Signals button
                html.A(
                    html.Div([
                        html.Img(
                            src="https://i.postimg.cc/FFw0nQVC/c710e3f4fb226fa0e20d67de72a9a55f.png",
                            style={
                                'height': '20px',
                                'margin-right': '10px',
                                'vertical-align': 'middle'
                            }
                        ),
                        html.Span("Long/Short Signals")
                    ],
                        style={
                            'display': 'flex',
                            'align-items': 'center',
                            'justify-content': 'center'
                        }),
                    href="https://youtu.be/WsKWDGZDT3Q",
                    target="_blank",
                    style={
                        'display': 'block',
                        'color': '#0088cc',
                        'text-decoration': 'none',
                        'font-weight': 'bold',
                        'font-size': '14px',
                        'background-color': 'rgba(0,136,204,0.1)',
                        'padding': '12px',
                        'border-radius': '8px',
                        'border': '1px solid rgba(0,136,204,0.3)',
                        'text-transform': 'uppercase',
                        'letter-spacing': '1px',
                        'transition': 'all 0.3s ease',
                        'margin': '10px 0',
                        'cursor': 'pointer'
                    }
                ),

                # Breakout Signals button
                html.A(
                    html.Div([
                        html.Img(
                            src="https://i.postimg.cc/FFw0nQVC/c710e3f4fb226fa0e20d67de72a9a55f.png",
                            style={
                                'height': '20px',
                                'margin-right': '10px',
                                'vertical-align': 'middle'
                            }
                        ),
                        html.Span("Breakout Signals")
                    ],
                        style={
                            'display': 'flex',
                            'align-items': 'center',
                            'justify-content': 'center'
                        }),
                    href="https://youtu.be/GNpU7PbjE1A",
                    target="_blank",
                    style={
                        'display': 'block',
                        'color': '#0088cc',
                        'text-decoration': 'none',
                        'font-weight': 'bold',
                        'font-size': '14px',
                        'background-color': 'rgba(0,136,204,0.1)',
                        'padding': '12px',
                        'border-radius': '8px',
                        'border': '1px solid rgba(0,136,204,0.3)',
                        'text-transform': 'uppercase',
                        'letter-spacing': '1px',
                        'transition': 'all 0.3s ease',
                        'margin': '10px 0',
                        'cursor': 'pointer'
                    }
                ),

                # Support/Resistance Levels button
                html.A(
                    html.Div([
                        html.Img(
                            src="https://i.postimg.cc/FFw0nQVC/c710e3f4fb226fa0e20d67de72a9a55f.png",
                            style={
                                'height': '20px',
                                'margin-right': '10px',
                                'vertical-align': 'middle'
                            }
                        ),
                        html.Span("Support/Resistance Levels")
                    ],
                        style={
                            'display': 'flex',
                            'align-items': 'center',
                            'justify-content': 'center'
                        }),
                    href="https://youtu.be/uThgJ_QMiNU",
                    target="_blank",
                    style={
                        'display': 'block',
                        'color': '#0088cc',
                        'text-decoration': 'none',
                        'font-weight': 'bold',
                        'font-size': '14px',
                        'background-color': 'rgba(0,136,204,0.1)',
                        'padding': '12px',
                        'border-radius': '8px',
                        'border': '1px solid rgba(0,136,204,0.3)',
                        'text-transform': 'uppercase',
                        'letter-spacing': '1px',
                        'transition': 'all 0.3s ease',
                        'margin': '10px 0',
                        'cursor': 'pointer'
                    }
                ),

                # Iron Condor Timing button
                html.A(
                    html.Div([
                        html.Img(
                            src="https://i.postimg.cc/FFw0nQVC/c710e3f4fb226fa0e20d67de72a9a55f.png",
                            style={
                                'height': '20px',
                                'margin-right': '10px',
                                'vertical-align': 'middle'
                            }
                        ),
                        html.Span("neutral strategies")
                    ],
                        style={
                            'display': 'flex',
                            'align-items': 'center',
                            'justify-content': 'center'
                        }),
                    href="https://youtu.be/lmvKwZ6NlmE",
                    target="_blank",
                    style={
                        'display': 'block',
                        'color': '#0088cc',
                        'text-decoration': 'none',
                        'font-weight': 'bold',
                        'font-size': '14px',
                        'background-color': 'rgba(0,136,204,0.1)',
                        'padding': '12px',
                        'border-radius': '8px',
                        'border': '1px solid rgba(0,136,204,0.3)',
                        'text-transform': 'uppercase',
                        'letter-spacing': '1px',
                        'transition': 'all 0.3s ease',
                        'margin': '10px 0',
                        'cursor': 'pointer'
                    }
                )
            ], style={
                'max-width': '400px',
                'margin': '0 auto',
                'padding': '40px',
                'background-color': 'rgba(30,30,30,0.8)',
                'border-radius': '12px',
                'box-shadow': '0 8px 30px rgba(0,0,0,0.5)',
                'backdrop-filter': 'blur(10px)',
                'border': '1px solid rgba(255,255,255,0.1)',
                'display': 'flex',
                'flex-direction': 'column',
                'align-items': 'center'
            })
        ], style={'margin-bottom': '50px'}),

        html.Div([
            html.H2("Gamma Exposure (GEX)", style={'color': '#00ffcc'}),
            html.P(
                "Gamma Exposure (GEX) measures how much market makers need to hedge their options positions. Positive GEX means market makers are long gamma and act as market stabilizers (buying dips and selling rallies). Negative GEX means they're short gamma and may amplify market moves."),

            html.H3("Key Concepts:", style={'color': '#00ffcc'}),
            html.Ul([
                html.Li(html.Strong("Positive GEX:"), " Market makers are stabilizing forces (buy low, sell high)"),
                html.Li(html.Strong("Negative GEX:"), " Market makers amplify moves (buy high, sell low)"),
                html.Li(html.Strong("GEX Flip Zones:"), " Where gamma changes from positive to negative or vice versa"),
                html.Li(html.Strong("OI, Volume:"),
                        " Price where most options expire worthless (often acts as magnet)"),
                html.Li(html.Strong("AG (Absolute Gamma):"), " Total gamma regardless of direction (shows key levels)")
            ], style={'color': 'white'}),

            html.H2("Basic Guidelines", style={'color': '#00ffcc'}),

            html.H3("1. Positive GEX", style={'color': '#ab47bc'}),
            html.P("When GEX is strongly positive:"),
            html.Ul([
                html.Li("Expect dip buying and selling at resistance levels"),
                html.Li("Look for support at AG, High Put Volume, High Put OI strikes"),
                html.Li(
                    "Resistance often forms at Max Positive GEX, High Call Volume strikes, High Call OI strikes"),
                html.Li("VWAP tends to act as strong support/resistance")
            ], style={'color': 'white'}),

            html.H3("2. Negative GEX", style={'color': '#ab47bc'}),
            html.P("When GEX is strongly negative:"),
            html.Ul([
                html.Li("Expect trend-following momentum moves"),
                html.Li("Breakdowns are more likely to continue"),
                html.Li("Watch for accelerating dealer hedging activity"),
                html.Li("Monitor VIX. A spike above 30 may indicate panic")
            ], style={'color': 'white'}),

            html.H3("3. GEX Flip Zones", style={'color': '#ab47bc'}),
            html.P("These are critical levels where gamma flips:"),
            html.Ul([
                html.Li("These levels always act as strong support/resistance"),
                html.Li(
                    "Breakouts through these levels tend to be real and may accelerate in the breakout direction"),
            ], style={'color': 'white'}),

            html.H2("Practical Trading Tips", style={'color': '#00ffcc'}),
            html.Ol([
                html.Li("Combine GEX with VWAP - longs from VWAP in positive GEX environments have high probability"),
                html.Li(
                    "Watch for confluence - when multiple indicators point to the same level (GEX + OI + volume + AG), it strengthens its role as either a magnet or strong support/resistance"),
                html.Li(
                    "In positive GEX, sell rallies at resistance and buy dips at support"),
                html.Li(
                    "In negative GEX, trade with the downside momentum but be ready to exit quickly"),
                html.Li("Monitor GEX changes intraday, especially around key technical levels")
            ], style={'color': 'white'}),

            html.H2("Common Mistakes", style={'color': '#00ffcc'}),
            html.Ul([
                html.Li(
                    "Fading gamma (e.g., buying during negative GEX). For longs, wait for selling to exhaust and price to reclaim VWAP. Remember: in negative GEX, market makers sell into weakness and even strong supports (High Put Vol, High AG etc.) may fail during panic"),
                html.Li("Ignoring G-Flip zones when they align with technical levels"),
                html.Li(
                    "Ignoring macro/fundamental context (Even if price is in positive GEX early/mid-day, negative fundamental triggers can still emerge)"),
                html.Li(
                    "Ignoring other technical analysis. For example, after multi-day declines with extreme negative GEX, avoid blind shorting as oversold conditions (RSI, S5FI etc.) may halt further downside")
            ], style={'color': 'white'}),

            html.Div([
                html.H3("Example Trade Setup", style={'color': '#00ffcc'}),
                html.P(
                    "Scenario: SPX in strong positive GEX environment: Long from VWAP or AG support level:"),
                html.Ul([
                    html.Li("Entry: Buy at VWAP or AG support level"),
                    html.Li("Stop: Below nearest Put OI cluster or Put Vol cluster"),
                    html.Li("Target: Next resistance level"),
                    html.Li(
                        "Management: Scale out as price approaches resistance")
                ], style={'color': 'white'})
            ], style={'margin-top': '20px', 'padding': '15px', 'background-color': '#252525', 'border-radius': '10px'}),

            html.Div([
                html.H3("Key Metrics to Monitor", style={'color': '#00ffcc'}),
                html.Table([
                    html.Tr([
                        html.Th("Indicator", style={'text-align': 'left'}),
                        html.Th("Bullish Signal", style={'text-align': 'left'}),
                        html.Th("Bearish Signal", style={'text-align': 'left'})
                    ]),
                    html.Tr([
                        html.Td("Net GEX"),
                        html.Td("Predominantly positive values"),
                        html.Td("Predominantly negative values")
                    ]),
                    html.Tr([
                        html.Td("AG"),
                        html.Td("Mostly above price"),
                        html.Td("Mostly below price")
                    ]),
                    html.Tr([
                        html.Td("P/C Ratio"),
                        html.Td("Below 0.8"),
                        html.Td("Above 1.2")
                    ]),
                    html.Tr([
                        html.Td("Call Volume"),
                        html.Td("Call Vol > Put Vol"),
                        html.Td("Call Vol < Put Vol")
                    ]),
                    html.Tr([
                        html.Td("Put Volume"),
                        html.Td("Put Vol < Call Vol"),
                        html.Td("Put Vol > Call Vol")
                    ])
                ], style={'width': '100%', 'border-collapse': 'collapse', 'margin-top': '15px'})
            ], style={'margin-top': '30px'}),

            html.Div([
                html.H3("Remember:", style={'color': '#00ffcc'}),
                html.P("GEX is just one tool in your arsenal. Always combine it with:"),
                html.Ul([
                    html.Li("Price action analysis"),
                    html.Li("Volume profile"),
                    html.Li("Market context"),
                    html.Li("Risk management")
                ], style={'color': 'white'})
            ], style={'margin-top': '30px', 'padding': '15px', 'background-color': '#252525', 'border-radius': '10px'})
        ], style={
            'max-width': '900px',
            'margin': '0 auto',
            'padding': '20px',
            'color': 'white',
            'line-height': '1.6'
        }),
    ],
    style={
        'margin-left': '10%',
        'padding': '20px',
        'color': 'white'
    }
)

# Лейаут для страницы "Disclaimer"
disclaimer_page = html.Div(
    className='disclaimer-page',
    children=[
        html.H1("Disclaimer", style={'textAlign': 'center', 'color': 'white'}),

        html.Div([
            dcc.Markdown('''
            #### Information on Quant Power, contained on this and/or related web products, does not constitute individual investment advice. It is provided solely for informational purposes and should not be considered as an offer or recommendation to invest, buy, or sell any asset or financial instrument.
            #### The Project Administration reserves the right to modify and update the content of this information resource and other documents without notifying users.


            ### 1. No Investment Recommendations
            Content on this platform is not intended to be and does not constitute financial advice, investment advice, trading advice, or any other advice. The provided information should not be used as the sole basis for making investment decisions.

            ### 2. Risk Disclosure
            Trading and investing involve substantial risk of loss and are not suitable for every investor. You should carefully consider your investment objectives, level of experience, and risk appetite before making any investment decisions.

            ### 3. No Guarantees
            We do not guarantee the effectiveness or applicability of any strategies or information provided. Past performance is not indicative of future results.

            ### 4. Third-Party Content
            Our platform may contain links to third-party websites or content. We do not endorse and are not responsible for the accuracy of such third-party materials.

            ### 5. Limitation of Liability
            Quant Power shall not be liable for any direct, indirect, incidental, or consequential damages arising from or related to your use of this platform.

            ### 6. Data Accuracy
            While we strive to provide accurate market data, we cannot guarantee the precision of information obtained from third-party sources such as Yahoo Finance, etc.

            ### 7. For Informational Purposes Only
            This platform is intended solely for informational purposes and should not be construed as a recommendation to buy or sell any financial instrument. By using this platform, you acknowledge that you have read, understood, and agree to comply with this disclaimer.
            ''',
                         style={
                             'color': 'white',
                             'line-height': '1.6',
                             'padding': '20px',
                             'background-color': '#252525',
                             'boxShadow': '0 4px 10px rgba(0, 0, 0, 0.3)',
                             'border-radius': '10px',
                             'margin-top': '20px'
                         })
        ], style={
            'max-width': '800px',
            'margin': '0 auto',
            'padding': '20px'
        })
    ],
    style={
        'margin-left': '10%',
        'padding': '20px',
        'color': 'white'
    }
)
# Функция для получения исторических данных для ценовых графиков
def get_historical_data_for_chart(ticker):
    """Получает исторические данные с учетом замены SPX на XSP*10"""
    if ticker == "^SPX":
        # Для SPX используем данные XSP и умножаем на 10
        xsp_ticker = yf.Ticker("^XSP")
        data = xsp_ticker.history(period='1d', interval='1m')
        if not data.empty:
            # Умножаем все ценовые колонки на 10
            for col in ['Open', 'High', 'Low', 'Close']:
                data[col] = data[col] * 10
            # Volume оставляем как есть (не умножаем)
        return data
    else:
        # Стандартная логика для других тикеров
        stock = yf.Ticker(ticker)
        return stock.history(period='1d', interval='1m')


def calculate_vwap(data, ticker):
    """Рассчитывает VWAP с учетом особенностей тикера"""
    if ticker == "^SPX":
        # Для SPX используем оригинальные данные SPX для расчета VWAP
        spx_ticker = yf.Ticker("^SPX")
        spx_data = spx_ticker.history(period='1d', interval='1m')
        if not spx_data.empty:
            spx_data['CumulativeVolume'] = spx_data['Volume'].cumsum()
            spx_data['CumulativePV'] = (
                        spx_data['Volume'] * (spx_data['High'] + spx_data['Low'] + spx_data['Close']) / 3).cumsum()
            spx_data['VWAP'] = spx_data['CumulativePV'] / spx_data['CumulativeVolume']
            return spx_data['VWAP']

    # Стандартный расчет VWAP для других тикеров
    data['CumulativeVolume'] = data['Volume'].cumsum()
    data['CumulativePV'] = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum()
    data['VWAP'] = data['CumulativePV'] / data['CumulativeVolume']
    return data['VWAP']

# Лейаут для объединенной страницы
app.layout = html.Div([
    dcc.Store(id='username-store', storage_type='local'),
    dcc.Store(id='auth-status', storage_type='local', data=False),

    # Блок для ввода имени пользователя (отображается только до авторизации)
    html.Div(id='login-container', children=[
        # Градиентный фон вместо изображения
        html.Div(style={
            'position': 'absolute',
            'top': 0,
            'left': 0,
            'width': '100%',
            'height': '100%',
            'background': 'linear-gradient(0deg, #223558 0%, #00b5e2 100%)',
            'opacity': 0.5,
            'object-fit': 'cover',
            'z-index': -1
        }),

        # Главный контейнер
        html.Div([
            # Логотип и заголовок
            html.Div([

                html.H1("QUANT POWER", style={
                    'color': 'transparent',
                        'background': 'linear-gradient(90deg, #00ffcc, #008cff)',
                        '-webkit-background-clip': 'text',
                        'text-align': 'center',
                        'margin-bottom': '10px',
                        'font-weight': 'bold',
                        'font-size': '2.5rem',
                        'text-decoration': 'none',
                        'display': 'block',
                        'text-shadow': '0 2px 10px rgba(0, 255, 204, 0.3)'
                }),
                html.P("Exclusive information for exceptional people", style={
                    'color': 'rgba(255,255,255,0.8)',
                    'text-align': 'center',
                    'margin-bottom': '40px',
                    'font-size': '18px'
                })
            ], style={'text-align': 'center'}),

            # Форма входа
            html.Div([
                html.Label("Enter your login:", style={
                    'color': 'white',
                    'font-size': '16px',
                    'margin-bottom': '10px',
                    'display': 'block'
                }),
                dcc.Input(
                    id='username-input',
                    type='text',
                    placeholder='login',
                    className='dash-input',
                    style={
                        'width': '92.5%',
                        'padding': '15px',
                        'border-radius': '8px',
                        'border': 'none',
                        'background-color': 'rgba(45,45,45,0.8)',
                        'color': 'white',
                        'font-size': '16px',
                        'margin-bottom': '20px',
                        'box-shadow': '0 4px 15px rgba(0,0,0,0.2)'
                    }
                ),
                html.Button(
                    'CHECK ACCESS',
                    id='submit-button',
                    n_clicks=0,
                    className='dash-button',
                    style={
                        'width': '100%',
                        'padding': '15px',
                        'border-radius': '8px',
                        'background': 'linear-gradient(135deg, #00ffcc 0%, #008cff 100%)',
                        'color': '#1e1e1e',
                        'font-weight': 'bold',
                        'border': 'none',
                        'cursor': 'pointer',
                        'font-size': '16px',
                        'transition': 'all 0.3s ease',
                        'box-shadow': '0 4px 15px rgba(0,255,204,0.3)',
                        'margin-bottom': '20px'
                    }
                ),
                html.Div(id='access-message', style={
                    'margin-top': '15px',
                    'text-align': 'center',
                    'font-size': '16px',
                    'min-height': '24px',
                    'color': '#00ffcc'
                }),

                # Кнопка Telegram
                html.Div(
                    dcc.Link(
                        html.Div([
                            html.Img(
                                src='https://upload.wikimedia.org/wikipedia/commons/8/82/Telegram_logo.svg',
                                style={'height': '24px', 'margin-right': '10px'}
                            ),
                            "Get access"
                        ], style={
                            'display': 'flex',
                            'align-items': 'center',
                            'justify-content': 'center'
                        }),
                        href="https://t.me/+ccPkiwklte01MDNi",
                        target="_blank",
                        style={
                            'display': 'block',
                            'color': '#0088cc',
                            'text-decoration': 'none',
                            'font-weight': 'bold',
                            'font-size': '14px',
                            'background-color': 'rgba(0,136,204,0.1)',
                            'padding': '12px',
                            'border-radius': '8px',
                            'border': '1px solid rgba(0,136,204,0.3)',
                            'text-transform': 'uppercase',
                            'letter-spacing': '1px',
                            'transition': 'all 0.3s ease',
                            'margin-top': '20px'
                        },
                        className='telegram-link'
                    ),
                    style={'width': '100%'}
                )
            ], style={
                'max-width': '400px',
                'margin': '0 auto',
                'padding': '40px',
                'background-color': 'rgba(30,30,30,0.8)',
                'border-radius': '12px',
                'box-shadow': '0 8px 30px rgba(0,0,0,0.5)',
                'backdrop-filter': 'blur(10px)',
                'border': '1px solid rgba(255,255,255,0.1)'
            }),

            # Преимущества платформы
            html.Div([
                html.H2("", style={
                    'color': '#00ffcc',
                    'text-align': 'center',
                    'margin': '40px 0 30px'
                }),
                html.Div([
                    html.Div([
                        html.Div([
                            html.Img(
                                src='https://i.postimg.cc/3JpcHcJC/stock-market.png',
                                style={'height': '60px', 'margin-bottom': '15px'}
                            ),
                            html.H3("Key Levels", style={'color': 'transparent',
                        'background': 'linear-gradient(90deg, #00ffcc, #008cff)',
                        '-webkit-background-clip': 'text',
                        'text-align': 'center',
                        'margin-bottom': '10px',
                        'font-weight': 'bold',
                        'font-size': '1.5rem',
                        'text-decoration': 'none',
                        'display': 'block',
                        'text-shadow': '0 2px 10px rgba(0, 255, 204, 0.3)'}),
                            html.P("Identification of key support and resistance levels based on GEX profile",
                                   style={'color': 'rgba(255,255,255,0.7)', 'font-size': '14px'})
                        ], style={
                            'padding': '20px',
                            'text-align': 'center'
                        })
                    ], style={
                        'background': 'rgba(45,45,45,0.6)',
                        'border-radius': '10px',
                        'border': '1px solid rgba(0,255,204,0.1)', 'box-shadow': '0 4px 15px rgba(0,255,204,0.2)',
                        'transition': 'all 0.3s ease',
                        ':hover': {
                            'transform': 'translateY(-5px)',
                            'box-shadow': '0 10px 20px rgba(0,255,204,0.1)'
                        }
                    }),

                    html.Div([
                        html.Div([
                            html.Img(
                                src='https://i.postimg.cc/j2sNDCYh/pie-chart.png',
                                style={'height': '60px', 'margin-bottom': '15px'}
                            ),
                            html.H3("Gamma Analysis", style={'color': 'transparent',
                        'background': 'linear-gradient(90deg, #008cff, #00ffcc)',
                        '-webkit-background-clip': 'text',
                        'text-align': 'center',
                        'margin-bottom': '10px',
                        'font-weight': 'bold',
                        'font-size': '1.5rem',
                        'text-decoration': 'none',
                        'display': 'block',
                        'text-shadow': '0 2px 10px rgba(0, 255, 204, 0.3)'}),
                            html.P("Detailed analysis of gamma exposure to understand market makers' actions",
                                   style={'color': 'rgba(255,255,255,0.7)', 'font-size': '14px'})
                        ], style={
                            'padding': '20px',
                            'text-align': 'center'
                        })
                    ], style={
                        'background': 'rgba(45,45,45,0.6)',
                        'border-radius': '10px',
                        'border': '1px solid rgba(0,255,204,0.1)','box-shadow': '0 4px 15px rgba(0,255,204,0.2)',
                        'transition': 'all 0.3s ease'
                    }),

                    html.Div([
                        html.Div([
                            html.Img(
                                src='https://i.postimg.cc/Xvw5j9BP/1.png',
                                style={'height': '60px', 'margin-bottom': '15px'}
                            ),
                            html.H3("Volume & OI", style={'color': 'transparent',
                        'background': 'linear-gradient(90deg, #00ffcc, #008cff)',
                        '-webkit-background-clip': 'text',
                        'text-align': 'center',
                        'margin-bottom': '10px',
                        'font-weight': 'bold',
                        'font-size': '1.5rem',
                        'text-decoration': 'none',
                        'display': 'block',
                        'text-shadow': '0 2px 10px rgba(0, 255, 204, 0.3)'}),
                            html.P("Analysis of open interest and volume to identify key levels",
                                   style={'color': 'rgba(255,255,255,0.7)', 'font-size': '14px'})
                        ], style={
                            'padding': '20px',
                            'text-align': 'center'
                        })
                    ], style={
                        'background': 'rgba(45,45,45,0.6)',
                        'border-radius': '10px',
                        'border': '1px solid rgba(0,255,204,0.1)', 'box-shadow': '0 4px 15px rgba(0,255,204,0.2)',
                        'transition': 'all 0.3s ease'
                    }),

                    html.Div([
                        html.Div([
                            html.Img(
                                src='https://i.postimg.cc/tCNWG3Pm/magnet.png',
                                style={'height': '60px', 'margin-bottom': '15px'}
                            ),
                            html.H3("Quant Power", style={'color': 'transparent',
                        'background': 'linear-gradient(90deg, #008cff, #00ffcc)',
                        '-webkit-background-clip': 'text',
                        'text-align': 'center',
                        'margin-bottom': '10px',
                        'font-weight': 'bold',
                        'font-size': '1.5rem',
                        'text-decoration': 'none',
                        'display': 'block',
                        'text-shadow': '0 2px 10px rgba(0, 255, 204, 0.3)'}),
                            html.P("Unique development that identifies the magnet level that attracts price",
                                   style={'color': 'rgba(255,255,255,0.7)', 'font-size': '14px'})
                        ], style={
                            'padding': '20px',
                            'text-align': 'center'
                        })
                    ], style={
                        'background': 'rgba(45,45,45,0.6)',
                        'border-radius': '10px',
                        'border': '1px solid rgba(0,255,204,0.1)', 'box-shadow': '0 4px 15px rgba(0,255,204,0.2)',
                        'transition': 'all 0.3s ease'
                    }),
                    html.Div([
                        html.Div([
                            html.Img(
                                src='https://i.postimg.cc/3xXCLMrV/classification.png',
                                style={'height': '60px', 'margin-bottom': '15px'}
                            ),
                            html.H3("Power Zone", style={'color': 'transparent',
                        'background': 'linear-gradient(90deg, #008cff, #00ffcc)',
                        '-webkit-background-clip': 'text',
                        'text-align': 'center',
                        'margin-bottom': '10px',
                        'font-weight': 'bold',
                        'font-size': '1.5rem',
                        'text-decoration': 'none',
                        'display': 'block',
                        'text-shadow': '0 2px 10px rgba(0, 255, 204, 0.3)'}),
                            html.P("Unique development that identifies the range within which price will move",
                                   style={'color': 'rgba(255,255,255,0.7)', 'font-size': '14px'})
                        ], style={
                            'padding': '20px',
                            'text-align': 'center'
                        })
                    ], style={
                        'background': 'rgba(45,45,45,0.6)',
                        'border-radius': '10px',
                        'border': '1px solid rgba(0,255,204,0.1)', 'box-shadow': '0 4px 15px rgba(0,255,204,0.2)',
                        'transition': 'all 0.3s ease'
                    }), html.Div([
                        html.Div([
                            html.Img(
                                src='https://i.postimg.cc/KcR9Pr4T/cctv-camera.png',
                                style={'height': '60px', 'margin-bottom': '15px'}
                            ),
                            html.H3("Monitoring", style={'color': 'transparent',
                        'background': 'linear-gradient(90deg, #00ffcc, #008cff)',
                        '-webkit-background-clip': 'text',
                        'text-align': 'center',
                        'margin-bottom': '10px',
                        'font-weight': 'bold',
                        'font-size': '1.5rem',
                        'text-decoration': 'none',
                        'display': 'block',
                        'text-shadow': '0 2px 10px rgba(0, 255, 204, 0.3)'}),
                            html.P("Real-time tracking of changes",
                                   style={'color': 'rgba(255,255,255,0.7)', 'font-size': '14px'})
                        ], style={
                            'padding': '20px',
                            'text-align': 'center'
                        })
                    ], style={
                        'background': 'rgba(45,45,45,0.6)',
                        'border-radius': '10px',
                        'border': '1px solid rgba(0,255,204,0.1)', 'box-shadow': '0 4px 15px rgba(0,255,204,0.2)',
                        'transition': 'all 0.3s ease'
                    }),
                    html.Div([
                        html.Div([
                            html.Img(
                                src='https://i.postimg.cc/Px0ppkHX/sand-clock.png',
                                style={'height': '60px', 'margin-bottom': '15px'}
                            ),
                            html.H3("Time is Money", style={'color': 'transparent',
                        'background': 'linear-gradient(90deg, #008cff, #00ffcc)',
                        '-webkit-background-clip': 'text',
                        'text-align': 'center',
                        'margin-bottom': '10px',
                        'font-weight': 'bold',
                        'font-size': '1.5rem',
                        'text-decoration': 'none',
                        'display': 'block',
                        'text-shadow': '0 2px 10px rgba(0, 255, 204, 0.3)'}),
                            html.P("Everything you need in one place. Spend time trading, not searching for information",
                                   style={'color': 'rgba(255,255,255,0.7)', 'font-size': '14px'})
                        ], style={
                            'padding': '20px',
                            'text-align': 'center'
                        })
                    ], style={
                        'background': 'rgba(45,45,45,0.6)',
                        'border-radius': '10px',
                        'border': '1px solid rgba(0,255,204,0.1)', 'box-shadow': '0 4px 15px rgba(0,255,204,0.2)',
                        'transition': 'all 0.3s ease'
                    }),
                    html.Div([
                        html.Div([
                            html.Img(
                                src='https://i.postimg.cc/zGMNKwvj/transparency.png',
                                style={'height': '60px', 'margin-bottom': '15px'}
                            ),
                            html.H3("Simplicity", style={'color': 'transparent',
                        'background': 'linear-gradient(90deg, #00ffcc, #008cff)',
                        '-webkit-background-clip': 'text',
                        'text-align': 'center',
                        'margin-bottom': '10px',
                        'font-weight': 'bold',
                        'font-size': '1.5rem',
                        'text-decoration': 'none',
                        'display': 'block',
                        'text-shadow': '0 2px 10px rgba(0, 255, 204, 0.3)'}),
                            html.P("Thoughtful visualization allows you to assess the market in moments",
                                   style={'color': 'rgba(255,255,255,0.7)', 'font-size': '14px'})
                        ], style={
                            'padding': '20px',
                            'text-align': 'center'
                        })
                    ], style={
                        'background': 'rgba(45,45,45,0.6)',
                        'border-radius': '10px',
                        'border': '1px solid rgba(0,255,204,0.1)', 'box-shadow': '0 4px 15px rgba(0,255,204,0.2)',
                        'transition': 'all 0.3s ease'
                    })
                ], style={
                    'display': 'grid',
                    'grid-template-columns': 'repeat(auto-fit, minmax(250px, 1fr))',
                    'gap': '20px',
                    'max-width': '1200px',
                    'margin': '0 auto'
                })
            ], style={'margin-top': '60px', 'padding': '0 20px'}),

            # Футер
            html.Div([
                html.P("© 2020 Quant Power. All rights reserved.", style={
                    'color': 'rgba(255,255,255,0.5)',
                    'text-align': 'center',
                    'margin-top': '60px',
                    'font-size': '14px'
                }),
                html.P("Disclaimer: Information is provided for educational purposes only.", style={
                    'color': 'rgba(255,255,255,0.3)',
                    'text-align': 'center',
                    'margin-top': '10px',
                    'font-size': '12px'
                })
            ])
        ], style={
            'max-width': '1400px',
            'margin': '0 auto',
            'padding': '40px 20px',
            'position': 'relative',
            'z-index': 1
        })
    ], style={
        'position': 'relative',
        'min-height': '100vh',
        'background': 'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)',
        'overflow': 'hidden'
    }),

    # Основной контент (отображается только после авторизации)
    html.Div(id='main-content', style={'display': 'none'}, children=[
        # Левая панель навигации
        html.Div([
            html.Div([
                dcc.Link(
                    html.H2("Quant Power", style={'color': 'white', 'cursor': 'pointer'}),
                    href="/",
                    style={'text-decoration': 'none'}
                ),
                html.Hr(),
                html.Ul([
                    html.Li(dcc.Link("OI Volume", href="/oi-volume",
                                     style={'color': 'white', 'text-decoration': 'none'})),
                    html.Li(style={'height': '20px'}),  # Добавляем пустой элемент для отступа
                    html.Li(dcc.Link("Key Levels", href="/key-levels",
                                     style={'color': 'white', 'text-decoration': 'none'})),
                    html.Li(style={'height': '20px'}),  # Добавляем пустой элемент для отступа
                    html.Li(dcc.Link("P/C Ratio", href="/options-summary",
                                     style={'color': 'white', 'text-decoration': 'none'})),
                    html.Li(style={'height': '20px'}),  # Добавляем пустой элемент для отступа
                    html.Li(dcc.Link("How to use GEX", href="/how-to-use-gex",
                                     style={'color': 'white', 'text-decoration': 'none'})),
                    html.Li(style={'height': '20px'}),  # Добавляем пустой элемент для отступа
                ], style={'list-style-type': 'none', 'padding': '0'}),

                # Добавляем Disclaimer внизу с дополнительным отступом
                html.Div([
                    html.Hr(),
                    html.Ul([
                        html.Li(dcc.Link("Disclaimer", href="/disclaimer",
                                         style={'color': 'gray', 'text-decoration': 'none', 'font-size': '20px'}))
                    ], style={'list-style-type': 'none', 'padding': '0', 'margin-top': '20px'})
                ], style={'position': 'absolute', 'bottom': '30px', 'width': '80%'})
            ], style={'padding': '20px', 'height': '100%', 'position': 'relative'})
        ], style={'width': '10%', 'height': '100%',
                  'position': 'fixed', 'left': '0', 'top': '0',
                'background': '#191919',
                'box-shadow': '5px 0 15px rgba(0,0,0,0.2)'}),

        # Основной контент страницы
        html.Div([
            dcc.Location(id='url', refresh=False),
            html.Div(id='page-content')
        ], style={'margin-left': '10%', 'padding': '20px',
                'width': 'calc(100% - 220px)',
                'min-height': '100vh',
                'background': '#191919'})
    ])
])

# Лейаут для главной страницы
index_page = html.Div([
    html.H1("Quant Power", style={'textAlign': 'center'}),

    html.Div([
        html.Label(""),
        dcc.Input(id='ticker-input', type='text', value='SPX', className='dash-input'),
        html.Button('Search', id='search-button', n_clicks=0, className='dash-button', style={'margin-left': '10px'}),
    ], className='dash-container', style={'display': 'flex', 'align-items': 'center'}),

    html.Div([
        html.Label("Expiration:"),
        dcc.Dropdown(id='date-dropdown', multi=True, className='dash-dropdown'),
    ], className='dash-container'),

    html.Div([
        html.Label("Parameters:"),
        html.Div([
            html.Button("Net GEX", id="btn-net-gex", className="parameter-button"),
            html.Button("AG", id="btn-ag", className="parameter-button"),
            html.Button("Call OI", id="btn-call-oi", className="parameter-button"),
            html.Button("Put OI", id="btn-put-oi", className="parameter-button"),
            html.Button("Call Volume", id="btn-call-vol", className="parameter-button"),
            html.Button("Put Volume", id="btn-put-vol", className="parameter-button"),
            html.Button("Power Zone", id="btn-power-zone", className="parameter-button"),  # New button
        ], className="button-container"),
    ], className='dash-container'),

    dcc.Store(id='selected-params', data=['Net GEX']),
    dcc.Store(id='options-data-store'),

    dcc.Graph(
        id='options-chart',
        style={'height': '900px', 'border-radius': '12px',
               'box-shadow': '0 4px 10px rgba(0, 0, 0, 0.3)',
               'overflow': 'hidden',
               'background-color': '#1e1e1e',
               'padding': '10px',
               'margin-bottom': '20px'},
        config={
            'displayModeBar': False,
            'scrollZoom': False,
            'dragmode': False
        }
    ),

    dcc.Graph(
        id='price-chart',
        style={'height': '950px', 'border-radius': '12px',
               'box-shadow': '0 4px 10px rgba(0, 0, 0, 0.3)',
               'overflow': 'hidden',
               'background-color': '#1e1e1e',
               'padding': '10px',
               'margin-bottom': '20px'},
        config={
            'displayModeBar': False,
            'scrollZoom': False,
            'dragmode': False
        }
    ),
    dcc.Graph(
        id='price-chart-simplified',
        style={'height': '950px', 'border-radius': '12px',
               'box-shadow': '0 4px 10px rgba(0, 0, 0, 0.3)',
               'overflow': 'hidden',
               'background-color': '#1e1e1e',
               'padding': '10px',
               'margin-bottom': '20px'},
        config={
            'displayModeBar': False,
            'scrollZoom': False,
            'dragmode': False
        }
    )
])

# Лейаут для страницы "Key Levels"
key_levels_page = html.Div(
    className='key-levels-page',
    children=[
        html.H1("Key Levels", style={'textAlign': 'center', 'color': 'white'}),

        html.Div([
            html.Label("", style={'color': 'white'}),
            dcc.Input(id='ticker-input-key-levels', type='text', value='SPX', className='dash-input'),
            html.Button('Search', id='search-button-key-levels', n_clicks=0, className='dash-button',
                        style={'margin-left': '10px'}),
        ], className='dash-container', style={'display': 'flex', 'align-items': 'center'}),

        html.Div(
            dcc.Graph(
                id='key-levels-chart',
                style={'height': '900px'},
                config={
                    'displayModeBar': False,
                    'scrollZoom': False,
                    'dragmode': False
                }
            ),
            className='graph-container'
        ),

        html.Div(
            id='market-forecast',
            style={
                'box-shadow': '0 4px 10px rgba(0, 0, 0, 0.3)',
                'background-color': '#1e1e1e',
                'padding': '20px',
                'border-radius': '12px',
                'margin-top': '20px',
                'color': 'white'
            },
            children=[
                html.H3("", style={'color': 'white'}),
                html.Div(id='forecast-text', style={'margin-top': '10px'})
            ]
        )
    ]
)


# Добавляем новый callback для прогноза
# Добавляем новый callback для прогноза
@app.callback(
    Output('forecast-text', 'children'),
    [Input('search-button-key-levels', 'n_clicks'),
     Input('ticker-input-key-levels', 'n_submit')],
    [State('ticker-input-key-levels', 'value')],
    prevent_initial_call=False  # Разрешаем первоначальный вызов
)
def update_forecast(n_clicks, n_submit, ticker):
    ctx = dash.callback_context

    # Если это первоначальный запуск - используем SPX
    if not ctx.triggered:
        ticker = 'SPX'
    elif not ticker:  # Если поле пустое
        ticker = 'SPX'

    ticker = normalize_ticker(ticker)
    stock = yf.Ticker(ticker)

    # Получаем данные по ценам
    try:
        hist = stock.history(period='3mo', interval='1d')
        intraday_hist = get_historical_data_for_chart(ticker)
        if hist.empty or intraday_hist.empty:
            return html.Div("Нет данных для анализа", style={'color': 'white'})

        current_price = intraday_hist['Close'].iloc[-1]
        vwap = (intraday_hist['Volume'] * (
                intraday_hist['High'] + intraday_hist['Low'] + intraday_hist['Close']) / 3).sum() / intraday_hist[
                   'Volume'].sum()
        current_volume = intraday_hist['Volume'].iloc[-1]
        avg_volume = intraday_hist['Volume'].mean()
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        return html.Div("Ошибка загрузки ценовых данных", style={'color': 'white'})

    # Технические индикаторы
    hist['MA20'] = hist['Close'].rolling(window=20).mean()
    hist['MA50'] = hist['Close'].rolling(window=50).mean()
    hist['MA200'] = hist['Close'].rolling(window=200).mean()

    # RSI расчет
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]

    # Получаем данные по опционам
    options_data, _, spot_price, _ = get_option_data(ticker, [])
    if options_data is None or options_data.empty:
        return html.Div("Нет данных по опционам для анализа", style={'color': 'white'})

    # Рассчитываем технические уровни
    price_range = 0.02 if ticker in ["^SPX", "^NDX", "^RUT", "^DJI"] else 0.05
    lower_limit = current_price * (1 - price_range)
    upper_limit = current_price * (1 + price_range)
    filtered_data = options_data[(options_data['strike'] >= lower_limit) & (options_data['strike'] <= upper_limit)]

    if filtered_data.empty:
        return html.Div("Недостаточно данных в ценовом диапазоне", style={'color': 'white'})

    # Ключевые уровни
    max_call_vol_strike = filtered_data.loc[filtered_data['Call Volume'].idxmax(), 'strike']
    max_put_vol_strike = filtered_data.loc[filtered_data['Put Volume'].idxmax(), 'strike']
    max_neg_gex_strike = filtered_data.loc[filtered_data['Net GEX'].idxmin(), 'strike']
    max_pos_gex_strike = filtered_data.loc[filtered_data['Net GEX'].idxmax(), 'strike']
    max_ag_strike = filtered_data.loc[filtered_data['AG'].idxmax(), 'strike']
    max_call_oi_strike = filtered_data.loc[filtered_data['Call OI'].idxmax(), 'strike']
    max_put_oi_strike = filtered_data.loc[filtered_data['Put OI'].idxmax(), 'strike']

    # Находим G-Flip зону
    g_flip_zone = None
    gex_values = filtered_data['Net GEX'].values
    for i in range(len(gex_values) - 6):
        if gex_values[i] > 0 and all(gex_values[i + j] < 0 for j in range(1, 7)):
            g_flip_zone = filtered_data.iloc[i]['strike']
            break

    # Рассчитываем P/C Ratio и объемы
    total_call_oi = options_data['Call OI'].sum()
    total_put_oi = options_data['Put OI'].sum()
    pc_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else float('inf')

    call_volume = filtered_data['Call Volume'].sum()
    put_volume = filtered_data['Put Volume'].sum()
    volume_ratio = put_volume / call_volume if call_volume > 0 else float('inf')

    # Статус Net GEX
    net_gex_status = "положительный" if filtered_data['Net GEX'].sum() > 0 else "отрицательный"

    # Проверяем рыночный фон
    bullish_background = bearish_background = False
    if current_price:
        strikes_below = filtered_data[filtered_data['strike'] < current_price].sort_values('strike', ascending=False)
        strikes_above = filtered_data[filtered_data['strike'] > current_price].sort_values('strike')

        # Бычий фон (положительный GEX вокруг цены)
        pos_gex_below = sum(strikes_below['Net GEX'].head(3) > 0)
        pos_gex_above = sum(strikes_above['Net GEX'].head(3) > 0)
        bullish_background = pos_gex_below >= 3 and pos_gex_above >= 3

        # Медвежий фон (отрицательный GEX вокруг цены)
        neg_gex_below = sum(strikes_below['Net GEX'].head(3) < 0)
        neg_gex_above = sum(strikes_above['Net GEX'].head(3) < 0)
        bearish_background = neg_gex_below >= 3 and neg_gex_above >= 3

    # Функция для расчета вероятности отскока/пробоя
    def calculate_probability(strike, direction):
        factors = []
        weights = {
            'max_ag': 30,
            'max_gex': 30,
            'max_vol': 20,
            'g_flip': 20,
            'max_oi': 15,
            'vwap': 10,
            'ma': 10,
            'rsi': 5
        }

        # Проверяем технические факторы
        if abs(strike - max_ag_strike) <= 0.01 * strike:
            factors.append(('max_ag', weights['max_ag']))
        if direction == "support" and abs(strike - max_neg_gex_strike) <= 0.01 * strike:
            factors.append(('max_gex', weights['max_gex']))
        if direction == "resistance" and abs(strike - max_pos_gex_strike) <= 0.01 * strike:
            factors.append(('max_gex', weights['max_gex']))
        if direction == "support" and abs(strike - max_put_vol_strike) <= 0.01 * strike:
            factors.append(('max_vol', weights['max_vol']))
        if direction == "resistance" and abs(strike - max_call_vol_strike) <= 0.01 * strike:
            factors.append(('max_vol', weights['max_vol']))
        if g_flip_zone and abs(strike - g_flip_zone) <= 0.01 * strike:
            factors.append(('g_flip', weights['g_flip']))
        if direction == "support" and abs(strike - max_put_oi_strike) <= 0.01 * strike:
            factors.append(('max_oi', weights['max_oi']))
        if direction == "resistance" and abs(strike - max_call_oi_strike) <= 0.01 * strike:
            factors.append(('max_oi', weights['max_oi']))

        # Добавляем технические индикаторы
        if abs(strike - vwap) <= 0.005 * strike:
            factors.append(('vwap', weights['vwap']))
        for ma in [hist['MA20'].iloc[-1], hist['MA50'].iloc[-1], hist['MA200'].iloc[-1]]:
            if abs(strike - ma) <= 0.005 * strike:
                factors.append(('ma', weights['ma']))
                break

        # Учитываем RSI
        if (direction == "support" and current_rsi < 30) or (direction == "resistance" and current_rsi > 70):
            factors.append(('rsi', weights['rsi']))

        # Ограничиваем максимальную вероятность 95%
        probability = min(95, sum(f[1] for f in factors))
        return probability, factors

    # Формируем прогноз
    forecast = []

    # 1. Основные данные (обновленный заголовок)
    forecast.append(html.H4(f"📊 Advanced Options Market Analysis: {ticker}",
                            style={'color': '#00ffcc', 'text-align': 'left', 'margin-bottom': '15px'}))

    # Информационная панель
    info_panel = [
        html.Div([
            html.Div([
                html.P("Price:", style={'color': 'white'}),
                html.P(f"{current_price:.2f}", style={'color': 'white', 'font-weight': 'bold'})
            ], style={'display': 'flex', 'justify-content': 'space-between'}),

            html.Div([
                html.P("VWAP:", style={'color': 'white'}),
                html.P(f"{vwap:.2f}", style={'color': 'white', 'font-weight': 'bold'})
            ], style={'display': 'flex', 'justify-content': 'space-between'}),


            html.Div([
                html.P("RSI (14):", style={'color': 'white'}),
                html.P(f"{current_rsi:.1f}",
                       style={'color': 'red' if current_rsi > 70 else 'green' if current_rsi < 30 else 'white',
                              'font-weight': 'bold'})
            ], style={'display': 'flex', 'justify-content': 'space-between'}),

            html.Div([
                html.P("P/C Ratio:", style={'color': 'white'}),
                html.P(f"{pc_ratio:.2f}",
                       style={'color': 'red' if pc_ratio > 1.3 else 'green' if pc_ratio < 0.7 else 'white',
                              'font-weight': 'bold'})
            ], style={'display': 'flex', 'justify-content': 'space-between'}),

            html.Div([
                html.P("Net GEX:", style={'color': 'white'}),
                html.P(f"{filtered_data['Net GEX'].sum():,.0f}",
                       style={'color': 'green' if filtered_data['Net GEX'].sum() > 0 else 'red',
                              'font-weight': 'bold'})
            ], style={'display': 'flex', 'justify-content': 'space-between'}),
        ], style={
            'background-color': '#252525',
            'padding': '15px',
            'border-radius': '10px',
            'margin-bottom': '20px'
        })
    ]

    forecast.extend(info_panel)

    # 2. Определение рыночного сценария
    market_context = []
    if bullish_background:
        market_context.append(html.Div([
            html.H4("📈 STRONG BULLISH",
                    style={'color': 'lightgreen', 'text-align': 'center', 'margin-bottom': '10px'}),
            html.P("🔹 The price is in positive GEX", style={'color': 'lightgreen'}),
            html.P("🔹 Market makers act as market stabilizers", style={'color': 'lightgreen'}),
            html.P("🔹 Corrections are likely to be limited", style={'color': 'lightgreen'})
        ], style={
            'background-color': 'rgba(0, 255, 0, 0.1)',
            'padding': '15px',
            'border-radius': '10px',
            'border-left': '5px solid lightgreen',
            'margin-bottom': '20px'
        }))
    elif bearish_background:
        market_context.append(html.Div([
            html.H4("📉 STRONG BEARISH",
                    style={'color': 'salmon', 'text-align': 'center', 'margin-bottom': '10px'}),
            html.P("🔹 The price is in negative GEX", style={'color': 'salmon'}),
            html.P("🔹 Market makers increase volatility", style={'color': 'salmon'}),
            html.P("🔹 Sudden movements are likely", style={'color': 'salmon'})
        ], style={
            'background-color': 'rgba(255, 0, 0, 0.1)',
            'padding': '15px',
            'border-radius': '10px',
            'border-left': '5px solid salmon',
            'margin-bottom': '20px'
        }))
    else:
        market_context.append(html.Div([
            html.H4("🔄 NEUTRAL/CONSOLIDATION SCENARIO",
                    style={'color': 'yellow', 'text-align': 'center', 'margin-bottom': '10px'}),
            html.P("🔹 GEX doesn't show a clear direction", style={'color': 'yellow'}),
            html.P("🔹 Trading in the range is likely", style={'color': 'yellow'}),
            html.P("🔹 Look for a breakdown with volume confirmation", style={'color': 'yellow'})
        ], style={
            'background-color': 'rgba(255, 255, 0, 0.1)',
            'padding': '15px',
            'border-radius': '10px',
            'border-left': '5px solid yellow',
            'margin-bottom': '20px'
        }))

    forecast.extend(market_context)

    # 3. Анализ ключевых уровней
    levels_analysis = []

    # Определяем ключевые уровни
    support_levels = []
    resistance_levels = []

    # Уровни поддержки
    if max_put_vol_strike < current_price:
        support_levels.append(('Put Volume', max_put_vol_strike))
    if max_neg_gex_strike < current_price:
        support_levels.append(('negative GEX', max_neg_gex_strike))
    if max_ag_strike < current_price:
        support_levels.append(('AG', max_ag_strike))
    if max_put_oi_strike < current_price:
        support_levels.append(('Put OI', max_put_oi_strike))

    # Уровни сопротивления
    if max_call_vol_strike > current_price:
        resistance_levels.append(('Call Volume', max_call_vol_strike))
    if max_pos_gex_strike > current_price:
        resistance_levels.append(('positive GEX', max_pos_gex_strike))
    if max_ag_strike > current_price:
        resistance_levels.append(('AG', max_ag_strike))
    if max_call_oi_strike > current_price:
        resistance_levels.append(('Call OI', max_call_oi_strike))

    # G-Flip зона
    if g_flip_zone:
        if g_flip_zone > current_price:
            resistance_levels.append(('G-Flip', g_flip_zone))
        else:
            support_levels.append(('G-Flip', g_flip_zone))

    # Группируем уровни поддержки по цене
    support_groups = {}
    for level in support_levels:
        price = level[1]
        if price not in support_groups:
            support_groups[price] = []
        support_groups[price].append(level)

    # Группируем уровни сопротивления по цене
    resistance_groups = {}
    for level in resistance_levels:
        price = level[1]
        if price not in resistance_groups:
            resistance_groups[price] = []
        resistance_groups[price].append(level)

    # Функция для форматирования уровней
    # В функции update_forecast замените блок format_level_groups на следующий:
    def format_level_groups(level_groups, level_type):
        formatted = []
        for price in sorted(level_groups.keys(), key=lambda x: abs(current_price - x)):
            levels = level_groups[price]
            prob, _ = calculate_probability(price, level_type)

            # Определяем цвет в зависимости от типа уровня
            if level_type == "support":
                color = '#02d432'  # Зеленый для поддержек
                prob_text = "rebound"
                level_name = "support"
            else:
                color = '#f32d35'  # Красный для сопротивлений
                prob_text = "stand down"
                level_name = "resistance"

            # Собираем названия параметров
            param_names = [level[0] for level in levels]

            # Определяем силу уровня
            if prob > 70:
                strength = "💪 Strong"
                strength_desc = "High probability"
            elif prob > 40:
                strength = "🆗 Average"
                strength_desc = "Moderate probability"
            else:
                strength = "⚠️ Weak"
                strength_desc = "Low probability"

            # Формируем основной текст (убрали факторы)
            main_text = f"{strength} {level_name} на {price:.2f}: {strength_desc} {prob_text} ({prob}%)"

            # Формируем список параметров (оставили только подтверждающие параметры)
            params_text = "Confirmed: " + ", ".join(param_names)

            formatted.append(html.Div([
                html.Div([
                    html.Span(main_text, style={'color': color, 'font-weight': 'bold'}),
                ], style={'margin-bottom': '5px'}),

                html.Div(params_text, style={'color': 'white', 'margin-left': '20px', 'margin-bottom': '15px'})
            ], style={
                'background-color': '#252525',
                'padding': '10px',
                'border-radius': '5px',
                'margin-bottom': '10px',
                'border-left': f'3px solid {color}'
            }))

        return formatted

    # Добавляем поддержки
    if support_groups:
        levels_analysis.append(html.H5("📉 KEY SUPPORTS:",
                                       style={'color': 'white', 'margin-top': '20px'}))
        levels_analysis.extend(format_level_groups(support_groups, "support"))

    # Добавляем сопротивления
    if resistance_groups:
        levels_analysis.append(html.H5("📈 KEY RESISTANCES:",
                                       style={'color': 'white', 'margin-top': '20px'}))
        levels_analysis.extend(format_level_groups(resistance_groups, "resistance"))

    forecast.extend(levels_analysis)

    # 4. Торговые идеи (обновленная логика)
    trading_ideas = []
    trading_ideas.append(html.H5("💡 variants:", style={'color': 'white', 'margin-top': '30px'}))

    def generate_trading_idea(price, level_type, prob, confirmations):
        idea = []
        color = 'lightgreen' if level_type == "support" else 'salmon'
        emoji = "🟢" if level_type == "support" else "🔴"

        if prob > 70:  # Сильный уровень - торгуем отскок/отбой
            if level_type == "support":
                idea.append(html.P(f"{emoji} Long positions on the rebound from support:",
                                   style={'color': color, 'font-weight': 'bold'}))
                idea.append(html.P(f"• Buy on the rebound from {price:.2f} with confirmation",
                                   style={'color': 'white'}))
            else:
                idea.append(html.P(f"{emoji} Short positions on the rebound from resistance:",
                                   style={'color': color, 'font-weight': 'bold'}))
                idea.append(
                    html.P(f"• sell when you break away from the resistance level {price:.2f} with confirmation",
                           style={'color': 'white'}))

            idea.append(html.P(f"• Probability of {'rebound' if level_type == 'support' else 'rebound'}: {prob}%",
                               style={'color': color}))
            idea.append(html.P(f"• Stop loss: {'below' if level_type == 'support' else 'higher'} level",
                               style={'color': 'white'}))
            idea.append(html.P(f"• Targets: the nearest {'resistances' if level_type == 'support' else 'supports'}",
                               style={'color': 'white'}))
        else:  # Слабый уровень - торгуем пробой
            if level_type == "support":
                idea.append(html.P(f"🔴 Short positions at the breakdown of support:",
                                   style={'color': 'salmon', 'font-weight': 'bold'}))
                idea.append(html.P(f"•Sell at the breakdown {price:.2f} with volumes",
                                   style={'color': 'white'}))
            else:
                idea.append(html.P(f"🟢 Long positions at the breakdown of resistance:",
                                   style={'color': 'lightgreen', 'font-weight': 'bold'}))
                idea.append(html.P(f"• Buy at the breakdown {price:.2f} with volumes",
                                   style={'color': 'white'}))

            idea.append(html.P(f"• The probability of continuation: {100 - prob}%",
                               style={'color': 'lightgreen' if level_type == 'resistance' else 'salmon'}))
            idea.append(html.P("• Look for confirmation on smaller timeframes",
                               style={'color': 'white'}))
            idea.append(
                html.P(f"•Targets: next {'support levels' if level_type == 'support' else 'resistance levels'}",
                       style={'color': 'white'}))

        return html.Div(idea, style={
            'background-color': f'rgba({0 if color == "lightgreen" else 255}, {255 if color == "lightgreen" else 0}, 0, 0.1)',
            'padding': '15px',
            'border-radius': '10px',
            'margin-bottom': '15px'
        })

    # Добавляем идеи для поддержек
    if support_groups:
        for price, levels in support_groups.items():
            prob, _ = calculate_probability(price, "support")
            confirmations = [level[0] for level in levels]
            trading_ideas.append(generate_trading_idea(price, "support", prob, confirmations))

    # Добавляем идеи для сопротивлений
    if resistance_groups:
        for price, levels in resistance_groups.items():
            prob, _ = calculate_probability(price, "resistance")
            confirmations = [level[0] for level in levels]
            trading_ideas.append(generate_trading_idea(price, "resistance", prob, confirmations))

    # Общие рекомендации для нейтрального рынка
    if not bullish_background and not bearish_background:
        trading_ideas.append(html.Div([
            html.P("🟡 Range Trading:", style={'color': 'yellow', 'font-weight': 'bold'}),
            html.P("• Buy near confirmed support levels, sell near confirmed resistance levels",
                   style={'color': 'white'}),
            html.P("• Use limit orders to enter near key levels", style={'color': 'white'}),
            html.P("• Reduce position size by 30-50% due to uncertainty", style={'color': 'white'}),
            html.P("• Look for false breakouts for better entries", style={'color': 'white'})
        ], style={
            'background-color': 'rgba(255, 255, 0, 0.1)',
            'padding': '15px',
            'border-radius': '10px',
            'margin-bottom': '15px'
        }))

    forecast.extend(trading_ideas)

    # 5. Управление рисками
    risk_management = []
    risk_management.append(html.H5("⚠️ RISK MANAGEMENT:", style={'color': 'white', 'margin-top': '30px'}))

    risk_management.append(html.Div([
        html.P("🔹 Position Sizing:", style={'color': 'white', 'font-weight': 'bold'}),
        html.P("• Risk no more than 1-2% of capital per trade", style={'color': 'white'}),
        html.P("• Reduce position size in high volatility conditions", style={'color': 'white'}),

        html.P("🔹 Stop Loss:", style={'color': 'white', 'font-weight': 'bold', 'margin-top': '10px'}),
        html.P(f"• For long positions: below nearest support ({min(support_groups.keys()):.2f} if present)"
               if support_groups else "• For long positions: 1-2% below entry point", style={'color': 'white'}),
        html.P(
            f"• For short positions: above nearest resistance ({min(resistance_groups.keys()):.2f} if present)"
            if resistance_groups else "• For short positions: 1-2% above entry point", style={'color': 'white'}),

        html.P("🔹 Take Profit:", style={'color': 'white', 'font-weight': 'bold', 'margin-top': '10px'}),
        html.P("• Secure partial profits at key levels", style={'color': 'white'}),
        html.P("• Use trailing stop after reaching first target", style={'color': 'white'}),

        html.P("🔹 Psychology:", style={'color': 'white', 'font-weight': 'bold', 'margin-top': '10px'}),
        html.P("• Avoid emotionally-driven trades", style={'color': 'white'}),
        html.P("• Stick to your trading plan", style={'color': 'white'}),
        html.P("• Review every trade", style={'color': 'white'})
    ], style={
        'background-color': '#252525',
        'padding': '15px',
        'border-radius': '10px',
        'margin-bottom': '20px'
    }))

    forecast.extend(risk_management)

    # 6. Дополнительные инсайты
    insights = []
    insights.append(html.H5("🔍 ADDITIONAL INSIGHTS:", style={'color': 'white', 'margin-top': '30px'}))

    # RSI Analysis
    rsi_analysis = ""
    if current_rsi > 70:
        rsi_analysis = "🔹 RSI indicates overbought conditions - potential correction"
    elif current_rsi < 30:
        rsi_analysis = "🔹 RSI indicates oversold conditions - potential bounce"
    else:
        rsi_analysis = "🔹 RSI in neutral zone - look for other confirmations"

    # P/C Ratio Analysis
    pc_analysis = ""
    if pc_ratio > 1.3:
        pc_analysis = "🔹 High P/C Ratio: market expects downside"
    elif pc_ratio < 0.7:
        pc_analysis = "🔹 Low P/C Ratio: market expects upside"
    else:
        pc_analysis = "🔹 Neutral P/C Ratio: no clear signal"

    insights.append(html.Div([
        html.P(rsi_analysis, style={'color': 'white'}),
        html.P(pc_analysis, style={'color': 'white'})
    ], style={
        'background-color': '#252525',
        'padding': '15px',
        'border-radius': '10px',
        'margin-bottom': '20px'
    }))

    forecast.extend(insights)

    return html.Div(forecast,
                    style={'color': 'white', 'padding': '20px', 'background-color': '#1e1e1e', 'border-radius': '10px'})


# Callback для проверки имени пользователя и управления видимостью элементов
@app.callback(
    [Output('access-message', 'children'),
     Output('main-content', 'style'),
     Output('login-container', 'style'),
     Output('username-store', 'data'),
     Output('auth-status', 'data')],
    [Input('submit-button', 'n_clicks')],
    [State('username-input', 'value'),
     State('username-store', 'data'),
     State('auth-status', 'data')]
)
def check_username(n_clicks, username, stored_username, auth_status):
    if n_clicks > 0:
        # Загружаем список разрешенных пользователей из файла
        allowed_users = load_allowed_users()

        if username and username in allowed_users:
            return (
                "Access is opened",
                {'display': 'block'},
                {'display': 'none'},
                username,
                True
            )
        else:
            return (
                "Access is closed",
                {'display': 'none'},
                {'display': 'block'},
                None,
                False
            )
    elif stored_username:
        # Проверяем сохраненного пользователя при загрузке страницы
        allowed_users = load_allowed_users()
        if stored_username in allowed_users and auth_status:
            return (
                "",
                {'display': 'block'},
                {'display': 'none'},
                stored_username,
                True
            )

    return (
        "",
        {'display': 'none'},
        {'display': 'block'},
        stored_username,
        auth_status
    )


# Callback для обновления списка дат
# Замените существующий callback на этот:
@app.callback(
    [Output('date-dropdown', 'options'), Output('date-dropdown', 'value')],
    [Input('search-button', 'n_clicks'),
     Input('ticker-input', 'n_submit')],
    [State('ticker-input', 'value')],
    prevent_initial_call=False
)
def update_dates(n_clicks, n_submit, ticker):
    ctx = dash.callback_context

    if not ctx.triggered:
        ticker = 'SPX'
    elif not ticker:
        ticker = 'SPX'

    ticker = normalize_ticker(ticker)

    # Получаем даты из файла
    file_dates = load_expirations_from_file(ticker)

    # Получаем даты из yfinance
    yfinance_dates = get_yfinance_expirations(ticker)

    # Объединяем даты, сохраняя порядок и уникальность
    if file_dates:
        # Если есть даты из файла, добавляем только те даты из yfinance, которые новее последней даты из файла
        last_file_date = datetime.strptime(file_dates[-1], '%Y-%m-%d')
        additional_dates = [
            date for date in yfinance_dates
            if datetime.strptime(date, '%Y-%m-%d') > last_file_date
        ]
        available_dates = file_dates + additional_dates
    else:
        # Если файла нет или он пуст, используем только даты из yfinance
        available_dates = yfinance_dates

    if not available_dates:
        print(f"Нет доступных дат экспирации для {ticker}")
        return [], []

    options = [{'label': date, 'value': date} for date in available_dates]
    return options, [available_dates[0]] if available_dates else []  # По умолчанию выбираем ближайшую дату


# Callback для обновления нажатых кнопок
@app.callback(
    [Output('selected-params', 'data'),
     Output('btn-net-gex', 'className'),
     Output('btn-ag', 'className'),
     Output('btn-call-oi', 'className'),
     Output('btn-put-oi', 'className'),
     Output('btn-call-vol', 'className'),
     Output('btn-put-vol', 'className'),
     Output('btn-power-zone', 'className')],  # Add output for new button
    [Input('btn-net-gex', 'n_clicks'),
     Input('btn-ag', 'n_clicks'),
     Input('btn-call-oi', 'n_clicks'),
     Input('btn-put-oi', 'n_clicks'),
     Input('btn-call-vol', 'n_clicks'),
     Input('btn-put-vol', 'n_clicks'),
     Input('btn-power-zone', 'n_clicks')],  # Add input for new button
    State('selected-params', 'data')
)
def update_selected_params(btn_net, btn_ag, btn_call_oi, btn_put_oi, btn_call_vol, btn_put_vol, btn_power_zone,
                           selected_params):
    ctx = dash.callback_context
    if not ctx.triggered:
        return (selected_params,
                "parameter-button", "parameter-button", "parameter-button",
                "parameter-button", "parameter-button", "parameter-button",
                "parameter-button")  # Add default for new button

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    button_map = {
        "btn-net-gex": "Net GEX",
        "btn-ag": "AG",
        "btn-call-oi": "Call OI",
        "btn-put-oi": "Put OI",
        "btn-call-vol": "Call Volume",
        "btn-put-vol": "Put Volume",
        "btn-power-zone": "Power Zone"  # New button mapping
    }

    param = button_map.get(button_id)

    if param:
        if param in selected_params:
            selected_params.remove(param)
        else:
            selected_params.append(param)

    button_classes = {
        "btn-net-gex": "active" if "Net GEX" in selected_params else "parameter-button",
        "btn-ag": "active" if "AG" in selected_params else "parameter-button",
        "btn-call-oi": "active" if "Call OI" in selected_params else "parameter-button",
        "btn-put-oi": "active" if "Put OI" in selected_params else "parameter-button",
        "btn-call-vol": "active" if "Call Volume" in selected_params else "parameter-button",
        "btn-put-vol": "active" if "Put Volume" in selected_params else "parameter-button",
        "btn-power-zone": "active" if "Power Zone" in selected_params else "parameter-button"  # New button class
    }

    return (selected_params,
            button_classes["btn-net-gex"],
            button_classes["btn-ag"],
            button_classes["btn-call-oi"],
            button_classes["btn-put-oi"],
            button_classes["btn-call-vol"],
            button_classes["btn-put-vol"],
            button_classes["btn-power-zone"])  # Add return for new button


# Callback для обновления графика опционов (возвращаем оригинальную версию)
@app.callback(
    Output('options-chart', 'figure'),
    [Input('search-button', 'n_clicks'),
     Input('ticker-input', 'n_submit'),
     Input('date-dropdown', 'value'),
     Input('selected-params', 'data')],
    [State('ticker-input', 'value')]
)
def update_options_chart(n_clicks, n_submit, dates, selected_params, ticker):
    ctx = dash.callback_context
    if not ctx.triggered or not dates or not selected_params:
        return go.Figure()

    ticker = normalize_ticker(ticker)

    # Получаем текущее значение VIX
    current_vix = get_current_vix()

    # Определяем диапазон в зависимости от VIX (только для SPX)
    if ticker == "^SPX":
        if current_vix < 20:
            price_range = 0.012  # 0.012%
        elif 20 <= current_vix < 25:
            price_range = 0.016  # 0.016%
        elif 25 <= current_vix < 30:
            price_range = 0.026  # 0.026%
        else:
            price_range = 0.023  # 2.3%
    elif ticker in ["^NDX", "^RUT", "^Dia"]:
        price_range = 0.017
    elif ticker in ["SPY", "QQQ", "DIA", "^XSP", "^IWM"]:
        price_range = 0.03
    elif ticker in ["^VIX"]:
        price_range = 0.5
    else:
        # Для акций - проверяем цену
        try:
            stock_info = yf.Ticker(ticker).info
            current_price = stock_info.get('regularMarketPrice', 0) or stock_info.get('currentPrice', 0)

            if current_price < 20:
                price_range = 2  # 200% для малых дешевых акций
            elif 20 <= current_price < 40:
                price_range = 0.30  # 30% для акций средней цены
            else:
                price_range = 0.12  # 12% для дорогих акций
        except Exception as e:
            print(f"Ошибка получения данных для {ticker}: {e}")
            price_range = 0.12  # Значение по умолчанию при ошибке

    options_data, _, spot_price, max_ag_strike = get_option_data(ticker, dates)
    if options_data is None or options_data.empty:
        return go.Figure()

    if spot_price:
        left_limit = spot_price - (spot_price * price_range)
        right_limit = spot_price + (spot_price * price_range)
        options_data = options_data[
            (options_data['strike'] >= left_limit) & (options_data['strike'] <= right_limit)
            ]
    if "Power Zone" in selected_params and spot_price:
        options_data['Power Zone'] = (
                (options_data['Call OI'] * spot_price / 100 * spot_price * 0.005) +
                (options_data['Put OI'] * spot_price / 100 * spot_price * 0.005) +
                (options_data['Call Volume'] * spot_price / 100 * spot_price * 0.005) +
                (options_data['Put Volume'] * spot_price / 100 * spot_price * 0.005)
        ).round(1)
    else:
        left_limit = right_limit = 0

        # Создаем фигуру перед использованием
    fig = go.Figure()

    # Оригинальная логика отображения параметров
    for parameter in selected_params:
        hover_texts = [
            f"Strike: {strike}<br>Call OI: {coi}<br>Put OI: {poi}<br>Call Volume: {cvol}<br>Put Volume: {pvol}<br>{parameter}: {val}"
            for strike, coi, poi, cvol, pvol, val in zip(
                options_data['strike'],
                options_data['Call OI'],
                options_data['Put OI'],
                options_data['Call Volume'],
                options_data['Put Volume'],
                options_data[parameter]
            )
        ]

        if parameter == "Net GEX":
            fig.add_trace(go.Bar(
                x=options_data['strike'],
                y=options_data['Net GEX'],
                marker_color=['#22b5ff' if v >= 0 else 'red' for v in options_data['Net GEX']],
                name="Net GEX",
                hovertext=hover_texts,
                hoverinfo="text",
                marker=dict(line=dict(width=0))
            ))

        elif parameter == "AG":
            fig.add_trace(go.Scatter(
                x=options_data['strike'],
                y=options_data['AG'],
                mode='lines+markers',
                line=dict(shape='spline', smoothing=0.7),
                marker=dict(size=8, color='#915bf8'),
                fill='tozeroy',
                name="AG",
                hovertext=hover_texts,
                hoverinfo="text",
                yaxis='y2'
            ))

        elif parameter == "Power Zone":
            fig.add_trace(go.Scatter(
                x=options_data['strike'],
                y=options_data['Power Zone'],
                mode='lines+markers',
                line=dict(shape='spline', smoothing=0.7),
                marker=dict(size=8, color='yellow'),
                fill='tozeroy',
                name="Power Zone",
                hovertext=hover_texts,
                hoverinfo="text",
                yaxis='y2'
            ))

        elif parameter == "Call OI":
            fig.add_trace(go.Scatter(
                x=options_data['strike'],
                y=options_data['Call OI'],
                mode='lines+markers',
                line=dict(shape='spline', smoothing=0.7),
                marker=dict(size=8, color='#02d432'),
                fill='tozeroy',
                name="Call OI",
                hovertext=hover_texts,
                hoverinfo="text",
                yaxis='y2'
            ))

        elif parameter == "Put OI":
            fig.add_trace(go.Scatter(
                x=options_data['strike'],
                y=options_data['Put OI'],
                mode='lines+markers',
                line=dict(shape='spline', smoothing=0.7),
                marker=dict(size=8, color='#f32d35'),
                fill='tozeroy',
                name="Put OI",
                hovertext=hover_texts,
                hoverinfo="text",
                yaxis='y2'
            ))

        elif parameter == "Call Volume":
            fig.add_trace(go.Scatter(
                x=options_data['strike'],
                y=options_data['Call Volume'],
                mode='lines+markers',
                line=dict(shape='spline', smoothing=0.7),
                marker=dict(size=8, color='#003cfe'),
                fill='tozeroy',
                name="Call Volume",
                hovertext=hover_texts,
                hoverinfo="text",
                yaxis='y2'
            ))

        elif parameter == "Put Volume":
            fig.add_trace(go.Scatter(
                x=options_data['strike'],
                y=options_data['Put Volume'],
                mode='lines+markers',
                line=dict(shape='spline', smoothing=0.7),
                marker=dict(size=8, color='#e55f04'),
                fill='tozeroy',
                name="Put Volume",
                hovertext=hover_texts,
                hoverinfo="text",
                yaxis='y2'
            ))

    if spot_price:
        fig.add_vline(
            x=spot_price,
            line_dash="solid",
            line_color="orange",
            annotation_text=f"Price: {spot_price:.2f}",
            annotation_position="top",
            annotation_font=dict(color="orange"),
        )

    # Оригинальное оформление графика
    fig.update_layout(
        xaxis=dict(
            title="Strike",
            showgrid=False,
            zeroline=False,
            tickmode='array',
            tickvals=options_data['strike'].tolist(),
            tickformat='1',
            fixedrange=True
        ),
        yaxis=dict(
            title="Net GEX",
            side="left",
            showgrid=False,
            zeroline=False,
            fixedrange=True
        ),
        yaxis2=dict(
            title="",
            side="right",
            overlaying="y",
            showgrid=False,
            zeroline=False,
            fixedrange=True
        ),
        title="" + ticker,
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font=dict(color='white'),
        dragmode=False,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Добавляем водяной знак
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        text="Quant Power",
        showarrow=False,
        font=dict(size=80, color="rgba(255, 255, 255, 0.1)"),
        textangle=0,
    )

    return fig


# Добавим функцию для получения текущего значения VIX
def get_current_vix():
    vix_ticker = yf.Ticker("^VIX")
    try:
        vix_data = vix_ticker.history(period='1d', interval='1m')
        if not vix_data.empty:
            return vix_data['Close'].iloc[-1]
    except Exception as e:
        print(f"Ошибка получения VIX: {e}")
    return 20  # Значение по умолчанию, если не удалось получить данные


# Callback для обновления графика price-chart
@app.callback(
    Output('price-chart', 'figure'),
    [Input('search-button', 'n_clicks'),
     Input('ticker-input', 'n_submit')],
    [State('ticker-input', 'value')],
    prevent_initial_call=False
)
def update_price_chart(n_clicks, n_submit, ticker):
    ctx = dash.callback_context

    if not ctx.triggered or ticker is None:
        ticker = 'SPX'

    ticker = normalize_ticker(ticker)

    # Check for Max Power value from file if SPX
    max_power_from_file = None
    if ticker == "^SPX":
        max_power_from_file = read_max_power_spx()

    interval = '1m'
    data = get_historical_data_for_chart(ticker)

    if data.empty:
        return go.Figure()

    options_data, _, spot_price, _ = get_option_data(ticker, [])

    # Добавляем проверку на None
    if options_data is None:
        options_data = pd.DataFrame()

    # Получаем текущее значение VIX
    current_vix = get_current_vix()

    # Determine price range based on VIX and ticker type
    if ticker == "^SPX":
        if current_vix < 20:
            price_range = 0.01
        elif 20 <= current_vix < 25:
            price_range = 0.018
        elif 25 <= current_vix < 30:
            price_range = 0.026
        else:
            price_range = 0.04
    elif ticker in ["^NDX", "^RUT", "^Dia"]:
        price_range = 0.017
    elif ticker in ["SPY", "QQQ", "DIA", "XSP", "IWM"]:
        price_range = 0.02
    elif ticker in ["^VIX"]:
        price_range = 0.5
    else:
        price_range = 0.12

    # Calculate VWAP
    data['CumulativeVolume'] = data['Volume'].cumsum()
    data['CumulativePV'] = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum()
    data['VWAP'] = data['CumulativePV'] / data['CumulativeVolume']

    # Get options data
    options_data, _, spot_price, _ = get_option_data(ticker, [])

    if spot_price:
        left_limit = spot_price - (spot_price * price_range)
        right_limit = spot_price + (spot_price * price_range)
        options_data = options_data[
            (options_data['strike'] >= left_limit) & (options_data['strike'] <= right_limit)
            ]
    else:
        left_limit = right_limit = 0

    if options_data is not None and not options_data.empty:
        visible_options_data = options_data[
            (options_data['strike'] >= left_limit) & (options_data['strike'] <= right_limit)
            ]
    else:
        visible_options_data = pd.DataFrame()

    if not visible_options_data.empty:
        max_ag_strike = visible_options_data.loc[visible_options_data['AG'].idxmax(), 'strike']

        # Проверяем наличие положительных значений Net GEX
        has_positive_gex = (visible_options_data['Net GEX'] > 0).any()
        if has_positive_gex:
            max_p1_strike = visible_options_data.loc[visible_options_data['Net GEX'].idxmax(), 'strike']
        else:
            max_p1_strike = None

        # Проверяем наличие отрицательных значений Net GEX перед отображением N1
        has_negative_gex = (visible_options_data['Net GEX'] < 0).any()
        if has_negative_gex:
            max_n1_strike = visible_options_data.loc[visible_options_data['Net GEX'].idxmin(), 'strike']
        else:
            max_n1_strike = None

        max_call_vol_strike = visible_options_data.loc[visible_options_data['Call Volume'].idxmax(), 'strike']
        max_put_vol_strike = visible_options_data.loc[visible_options_data['Put Volume'].idxmax(), 'strike']
        # Подготовка данных для зоны Call OI
        call_oi_data = visible_options_data[['strike', 'Call OI']].copy()
        call_oi_data = call_oi_data.sort_values('strike')
        max_call_oi = call_oi_data['Call OI'].max()
        call_oi_data['Call OI Normalized'] = (call_oi_data[
                                                  'Call OI'] / max_call_oi) * 0.8  # Нормализация для ширины зоны
    else:
        max_ag_strike = None
        max_p1_strike = None
        max_n1_strike = None
        max_call_vol_strike = None
        max_put_vol_strike = None

    # Prepare Power Zone data (only for SPX)
    if ticker == "^SPX" and not visible_options_data.empty:
        # Calculate Power Zone (combination of OI and Volume)
        visible_options_data['Power Zone'] = (
                (visible_options_data['Call OI'] * spot_price / 100 * spot_price * 0.005) +
                (visible_options_data['Put OI'] * spot_price / 100 * spot_price * 0.005) +
                (visible_options_data['Call Volume'] * spot_price / 100 * spot_price * 0.005) +
                (visible_options_data['Put Volume'] * spot_price / 100 * spot_price * 0.005)
        ).round(1)

        power_zone_data = visible_options_data[['strike', 'Power Zone']].copy()
        power_zone_data = power_zone_data.sort_values('strike')

        # Add mid-points for smoother curve
        if len(power_zone_data) > 1:
            new_points = []
            for i in range(len(power_zone_data) - 1):
                mid_strike = (power_zone_data.iloc[i]['strike'] + power_zone_data.iloc[i + 1]['strike']) / 2
                mid_value = (power_zone_data.iloc[i]['Power Zone'] + power_zone_data.iloc[i + 1]['Power Zone']) / 2
                new_points.append({'strike': mid_strike, 'Power Zone': mid_value})

            power_zone_data = pd.concat([power_zone_data, pd.DataFrame(new_points)]).sort_values('strike')

        max_power = power_zone_data['Power Zone'].max()
        if max_power > 0:
            power_zone_data['Power Zone Normalized'] = (power_zone_data['Power Zone'] / max_power) * 0.3
        else:
            power_zone_data['Power Zone Normalized'] = 0
    else:
        power_zone_data = pd.DataFrame()

    # Market time references
    market_open_time = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
    market_close_time = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)
    current_time = datetime.now()

    # Line widths configuration
    line_widths = {
        'AG': 7,
        'P1': 5,
        'N1': 5,
        'Call Vol': 4,
        'Put Vol': 4,
        'Max Power': 3,
        'Power Zone': 2
    }

    # Create figure
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=""
    ))

    # Power Zone вертикально (только для SPX)
    if ticker == "^SPX" and not power_zone_data.empty and max_power > 0:
        fig.add_trace(go.Scatter(
            x=market_open_time + (market_close_time - market_open_time) * power_zone_data['Power Zone Normalized'],
            y=power_zone_data['strike'],
            mode='lines+markers',
            line=dict(shape='spline', smoothing=0.8, color='yellow', width=2),
            marker=dict(size=1, color='yellow'),
            name='Power Zone',
            fill='tozerox',
            fillcolor='rgba(255, 255, 0, 0.2)',
            hoverinfo='text',
            hovertext=[f'Strike: {strike:.2f}<br>Power: {power:,.0f}'
                      for strike, power in zip(power_zone_data['strike'], power_zone_data['Power Zone'])]
        ))

    # Добавляем линию P1 только если есть положительные значения Net GEX
    if max_p1_strike is not None:
        fig.add_trace(go.Scatter(
            x=[market_open_time, market_close_time],
            y=[max_p1_strike, max_p1_strike],
            mode='lines',
            line=dict(color='#00ff00', width=line_widths['P1']),
            name=f'P1 Strike: {max_p1_strike:.2f}',
            yaxis='y'
        ))

    # Добавляем линию N1 только если есть отрицательные значения Net GEX
    if max_n1_strike is not None:
        fig.add_trace(go.Scatter(
            x=[market_open_time, market_close_time],
            y=[max_n1_strike, max_n1_strike],
            mode='lines',
            line=dict(color='#ff0000', width=line_widths['N1']),
            name=f'N1 Strike: {max_n1_strike:.2f}',
            yaxis='y'
        ))

    if max_call_vol_strike is not None:
        fig.add_trace(go.Scatter(
            x=[market_open_time, market_close_time],
            y=[max_call_vol_strike, max_call_vol_strike],
            mode='lines',
            line=dict(color='#00a0ff', width=line_widths['Call Vol']),
            name=f'Call Vol Strike: {max_call_vol_strike:.2f}',
            yaxis='y'
        ))

    if max_put_vol_strike is not None:
        fig.add_trace(go.Scatter(
            x=[market_open_time, market_close_time],
            y=[max_put_vol_strike, max_put_vol_strike],
            mode='lines',
            line=dict(color='#ac5631', width=line_widths['Put Vol']),
            name=f'Put Vol Strike: {max_put_vol_strike:.2f}',
            yaxis='y'
        ))

    # Определяем Max Power Strike
    if max_power_from_file is not None and ticker == "^SPX":
        # Use value from file if available for SPX
        max_power_strike = max_power_from_file
    else:
        if current_time - market_open_time <= timedelta(minutes=45):
            max_power_strike = max_call_vol_strike if max_call_vol_strike is not None else max_put_vol_strike
        else:
            # Новая логика после первых 45 минут
            if not options_data.empty and spot_price:
                # Вычисляем комбинированный показатель
                combined_power = (
                        (options_data['Call OI'] * spot_price / 100 * spot_price * 0.005) +
                        (options_data['Put OI'] * spot_price / 100 * spot_price * 0.005) +
                        (options_data['Call Volume'] * spot_price / 100 * spot_price * 0.005) +
                        (options_data['Put Volume'] * spot_price / 100 * spot_price * 0.005)
                ).round(1)
                max_power_strike = options_data.loc[combined_power.idxmax(), 'strike']
            else:
                max_power_strike = max_call_vol_strike if max_call_vol_strike is not None else max_put_vol_strike

    if max_power_strike is not None:
        fig.add_trace(go.Scatter(
            x=[market_open_time, market_close_time],
            y=[max_power_strike, max_power_strike],
            mode='lines',
            line=dict(color='#ffdf00', width=line_widths['Max Power']),
            name=f'Quant Power: {max_power_strike:.2f}',
            yaxis='y'
        ))

    if max_ag_strike is not None:
        fig.add_trace(go.Scatter(
            x=[market_open_time, market_close_time],
            y=[max_ag_strike, max_ag_strike],
            mode='lines',
            line=dict(color='#ab47bc', dash='dash', width=line_widths['AG']),
            name=f'AG Strike: {max_ag_strike:.2f}',
            yaxis='y'
        ))

    fig.update_layout(
        title=f"{ticker}",
        xaxis=dict(
            title="Time",
            type='date',
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            rangeslider=dict(visible=False),
            autorange=False,
            range=[market_open_time, market_close_time],
            fixedrange=True
        ),
        yaxis=dict(
            title="price",
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            fixedrange=True
        ),
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font=dict(color='white'),
        hovermode='x unified',
        margin=dict(l=50, r=50, b=50, t=50),
        dragmode=False
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        text="Quant Power",
        showarrow=False,
        font=dict(size=80, color="rgba(255, 255, 255, 0.1)"),
        textangle=0,
    )

    return fig


# Callback для обновления нового графика цены
@app.callback(
    Output('price-chart-simplified', 'figure'),
    [Input('search-button', 'n_clicks'),
     Input('ticker-input', 'n_submit')],
    [State('ticker-input', 'value')],
    prevent_initial_call=False
)
def update_price_chart_simplified(n_clicks, n_submit, ticker):
    ctx = dash.callback_context

    if not ctx.triggered:
        ticker = 'SPX'
    elif not ticker:
        ticker = 'SPX'

    ticker = normalize_ticker(ticker)
    interval = '1m'
    data = get_historical_data_for_chart(ticker)

    if data.empty:
        return go.Figure()

    # Используем новую функцию для расчета VWAP
    vwap_data = calculate_vwap(data, ticker)

    # Остальной код остается без изменений...
    options_data, _, spot_price, _ = get_option_data(ticker, [])

    if options_data is None or options_data.empty:
        return go.Figure()

    data['CumulativeVolume'] = data['Volume'].cumsum()
    data['CumulativePV'] = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum()
    data['VWAP'] = data['CumulativePV'] / data['CumulativeVolume']

    options_data, _, spot_price, _ = get_option_data(ticker, [])

    if options_data is None or options_data.empty:
        return go.Figure()

    if ticker in ["^SPX", "^NDX", "^RUT", "^DJI"]:
        price_range = 0.01
        resistance_zone_lower_percent = -0.0005
        resistance_zone_upper_percent = 0.0015
        support_zone_lower_percent = -0.0015
        support_zone_upper_percent = 0.0005
    elif ticker in ["SPY", "QQQ", "DIA", "XSP", "IWM"]:
        price_range = 0.022
        resistance_zone_lower_percent = -0.0005
        resistance_zone_upper_percent = 0.0015
        support_zone_lower_percent = -0.0015
        support_zone_upper_percent = 0.0005
    else:
        price_range = 0.05
        resistance_zone_lower_percent = -0.002
        resistance_zone_upper_percent = 0.0035
        support_zone_lower_percent = -0.0035
        support_zone_upper_percent = 0.002

    if spot_price:
        left_limit = spot_price - (spot_price * price_range)
        right_limit = spot_price + (spot_price * price_range)
        options_data = options_data[
            (options_data['strike'] >= left_limit) & (options_data['strike'] <= right_limit)
            ]
    else:
        left_limit = right_limit = 0

    max_call_vol_strike = options_data.loc[options_data['Call Volume'].idxmax(), 'strike']
    max_put_vol_strike = options_data.loc[options_data['Put Volume'].idxmax(), 'strike']
    max_negative_net_gex_strike = options_data.loc[options_data['Net GEX'].idxmin(), 'strike']

    resistance_zone_lower = max_call_vol_strike * (1 + resistance_zone_lower_percent)
    resistance_zone_upper = max_call_vol_strike * (1 + resistance_zone_upper_percent)

    if max_put_vol_strike < max_negative_net_gex_strike:
        support_zone_lower = max_put_vol_strike * (1 + support_zone_lower_percent)
        support_zone_upper = max_put_vol_strike * (1 + support_zone_upper_percent)
    else:
        support_zone_lower = max_negative_net_gex_strike * (1 + support_zone_lower_percent)
        support_zone_upper = max_negative_net_gex_strike * (1 + support_zone_upper_percent)

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=""
    ))

    fig.add_trace(go.Scatter(
        x=data.index,
        y=vwap_data,
        mode='lines',
        line=dict(color='#00ffcc', width=2),
        name='VWAP'
    ))

    market_open_time = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
    market_close_time = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)

    fig.add_trace(go.Scatter(
        x=[market_open_time, market_close_time, market_close_time, market_open_time],
        y=[resistance_zone_lower, resistance_zone_lower, resistance_zone_upper, resistance_zone_upper],
        fill="toself",
        fillcolor="rgba(0, 160, 255, 0.2)",
        line=dict(color="rgba(0, 160, 255, 0.5)"),
        mode="lines",
        name='Resistance zone',
        hoverinfo="none",
    ))

    fig.add_trace(go.Scatter(
        x=[market_open_time, market_close_time, market_close_time, market_open_time],
        y=[support_zone_lower, support_zone_lower, support_zone_upper, support_zone_upper],
        fill="toself",
        fillcolor="rgba(172, 86, 49, 0.2)",
        line=dict(color="rgba(172, 86, 49, 0.5)"),
        mode="lines",
        name='Support zone',
        hoverinfo="none",
    ))

    fig.update_layout(
        title=f"Support / Resistance {ticker}",
        xaxis=dict(
            title="Time",
            type='date',
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            rangeslider=dict(visible=False),
            autorange=False,
            range=[market_open_time, market_close_time],
            fixedrange=True
        ),
        yaxis=dict(
            title="Price",
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            fixedrange=True
        ),
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font=dict(color='white'),
        hovermode='x unified',
        margin=dict(l=50, r=50, b=50, t=50),
        dragmode=False
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        text="Quant Power",
        showarrow=False,
        font=dict(size=80, color="rgba(255, 255, 255, 0.1)"),
        textangle=0,
    )

    return fig


# Callback для обновления страницы
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname'),
     Input('url', 'search')]
)
def display_page(pathname, search):
    if pathname == '/key-levels':
        return key_levels_page
    elif pathname == '/oi-volume':  # Новое условие
        return oi_volume_page
    elif pathname == '/options-summary':
        return options_summary_page
    elif pathname == '/how-to-use-gex':
        return how_to_use_gex_page
    elif pathname == '/disclaimer':
        return disclaimer_page
    else:
        # Check if there's a ticker parameter in the URL
        ticker_value = 'SPX'  # default
        if search:
            params = parse_qs(search.lstrip('?'))
            if 'ticker' in params:
                ticker_value = params['ticker'][0]

        # Update the index page with the ticker value
        updated_index = index_page
        # Find the ticker input in the children and update its value
        for child in updated_index.children:
            if hasattr(child, 'id') and child.id == 'ticker-input':
                child.value = ticker_value
                break
        return updated_index


# Callback для обновления графика на странице "Key Levels"
@app.callback(
    Output('key-levels-chart', 'figure'),
    [Input('search-button-key-levels', 'n_clicks'),
     Input('ticker-input-key-levels', 'n_submit')],
    [State('ticker-input-key-levels', 'value')],
    prevent_initial_call=False  # Разрешаем первоначальный вызов
)
def update_key_levels_chart_callback(n_clicks, n_submit, ticker):
    ctx = dash.callback_context

    # Если это первоначальный запуск и нет ввода от пользователя
    if not ctx.triggered or ticker is None:
        ticker = 'SPX'  # Устанавливаем SPX по умолчанию

    return update_key_levels_chart(ticker)


@cache.memoize(timeout=60)  # Кэшируем на 5 минут
def update_key_levels_chart(ticker):
    ticker = normalize_ticker(ticker)
    interval = '1m'
    data = get_historical_data_for_chart(ticker)

    if data.empty:
        return go.Figure()

    data['CumulativeVolume'] = data['Volume'].cumsum()
    data['CumulativePV'] = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum()
    data['VWAP'] = data['CumulativePV'] / data['CumulativeVolume']

    if not data.empty:
        open_price = data['Open'].iloc[0]
        current_price = data['Close'].iloc[-1]
    else:
        open_price = current_price = None

    # Определяем диапазон для всего графика (4% от цены открытия)
    if ticker in ["^SPX", "^NDX", "^RUT", "^DJI", "^VIX"]:
        chart_range = 0.032  # 4% для индексов
    elif ticker in ["SPY", "QQQ", "DIA", "XSP", "IWM"]:
        chart_range = 0.032  # 4% для ETF
    else:
        chart_range = 0.08  # 10% для акций

    if open_price:
        upper_limit = open_price * (1 + chart_range / 2)
        lower_limit = open_price * (1 - chart_range / 2)
    else:
        upper_limit = lower_limit = 0

    options_data, _, spot_price, _ = get_option_data(ticker, [])

    if options_data is None or options_data.empty:
        return go.Figure()

    # Фильтруем данные опционов в пределах всего диапазона графика
    options_data = options_data[
        (options_data['strike'] >= lower_limit) &
        (options_data['strike'] <= upper_limit)
        ]

    if options_data.empty:
        return go.Figure()

    # 1. Основные уровни в пределах 1% от текущей цены (ОБЯЗАТЕЛЬНЫЕ)
    if current_price:
        one_percent_range = current_price * 0.01
        one_percent_upper = current_price + one_percent_range
        one_percent_lower = current_price - one_percent_range

        # Сопротивление: максимальный объем коллов в пределах +1%
        resistance_near = options_data[
            (options_data['strike'] >= current_price) &
            (options_data['strike'] <= one_percent_upper)
            ]
        if not resistance_near.empty:
            main_resistance = resistance_near.loc[resistance_near['Call Volume'].idxmax(), 'strike']
        else:
            # Если нет данных в +1%, берем ближайший страйк выше текущей цены с максимальным объемом коллов
            resistance_above = options_data[options_data['strike'] >= current_price]
            if not resistance_above.empty:
                main_resistance = resistance_above.loc[resistance_above['Call Volume'].idxmax(), 'strike']
            else:
                main_resistance = None

        # Поддержка: максимальный объем путов в пределах -1%
        support_near = options_data[
            (options_data['strike'] <= current_price) &
            (options_data['strike'] >= one_percent_lower)
            ]
        if not support_near.empty:
            main_support = support_near.loc[support_near['Put Volume'].idxmax(), 'strike']
        else:
            # Если нет данных в -1%, берем ближайший страйк ниже текущей цены с максимальным объемом путов
            support_below = options_data[options_data['strike'] <= current_price]
            if not support_below.empty:
                main_support = support_below.loc[support_below['Put Volume'].idxmax(), 'strike']
            else:
                main_support = None
    else:
        main_resistance = main_support = None

    # 2. Глобальные уровни во всем диапазоне (дополнительные)
    max_call_vol_strike = options_data.loc[options_data['Call Volume'].idxmax(), 'strike']
    max_put_vol_strike = options_data.loc[options_data['Put Volume'].idxmax(), 'strike']
    max_negative_net_gex_strike = options_data.loc[options_data['Net GEX'].idxmin(), 'strike']
    max_ag_strike = options_data.loc[options_data['AG'].idxmax(), 'strike']
    max_positive_net_gex_strike = options_data.loc[options_data['Net GEX'].idxmax(), 'strike']

    # 3. Определяем G-Flip зону
    g_flip_zone = None
    gex_values = options_data['Net GEX'].values
    for i in range(len(gex_values) - 6):
        if gex_values[i] < 0 and all(gex_values[i + j] > 0 for j in range(1, 7)):
            g_flip_zone = options_data.iloc[i]['strike']
            break

    # Определяем шаг страйков
    strike_step = options_data['strike'].diff().dropna().min()
    if pd.isna(strike_step) or strike_step == 0:
        strike_step = 1 if ticker in ["^SPX", "^NDX"] else 0.5

    # Создаем график
    fig = go.Figure()

    # Добавляем свечной график
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="Price"
    ))

    # Добавляем VWAP
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['VWAP'],
        mode='lines',
        line=dict(color='#00ffcc', width=2),
        name='VWAP'
    ))

    # Время открытия/закрытия рынка
    market_open_time = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
    market_close_time = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)

    # 1. ОСНОВНЫЕ УРОВНИ (1% диапазон) - ПРИОРИТЕТНЫЕ
    if main_resistance:
        # Зона сопротивления (основная)
        res_zone_lower = main_resistance * 0.9995
        res_zone_upper = main_resistance * 1.0015
        fig.add_trace(go.Scatter(
            x=[market_open_time, market_close_time, market_close_time, market_open_time],
            y=[res_zone_lower, res_zone_lower, res_zone_upper, res_zone_upper],
            fill="toself",
            fillcolor="rgba(0, 100, 255, 0.3)",
            line=dict(color="rgba(0, 100, 255, 0.7)", width=2),
            mode="lines",
            name=f'Main Resistance ({main_resistance:.2f})',
            hoverinfo="none",
        ))

    if main_support:
        # Зона поддержки (основная)
        sup_zone_lower = main_support * 0.9985
        sup_zone_upper = main_support * 1.0005
        fig.add_trace(go.Scatter(
            x=[market_open_time, market_close_time, market_close_time, market_open_time],
            y=[sup_zone_lower, sup_zone_lower, sup_zone_upper, sup_zone_upper],
            fill="toself",
            fillcolor="rgba(255, 100, 0, 0.3)",
            line=dict(color="rgba(255, 100, 0, 0.7)", width=2),
            mode="lines",
            name=f'Main Support ({main_support:.2f})',
            hoverinfo="none",

        ))

    # 2. ДОПОЛНИТЕЛЬНЫЕ УРОВНИ (весь диапазон)
    # Глобальное сопротивление (макс объем коллов)
    fig.add_trace(go.Scatter(
        x=[market_open_time, market_close_time],
        y=[max_call_vol_strike, max_call_vol_strike],
        mode='lines',
        line=dict(color='#22b5ff', width=2, dash='dot'),
        name=f'Global Call Vol ({max_call_vol_strike:.2f})'
    ))

    # Глобальная поддержка (макс объем путов)
    fig.add_trace(go.Scatter(
        x=[market_open_time, market_close_time],
        y=[max_put_vol_strike, max_put_vol_strike],
        mode='lines',
        line=dict(color='#ff2d3d', width=2, dash='dot'),
        name=f'Global Put Vol ({max_put_vol_strike:.2f})'
    ))

    # Макс отрицательный Net GEX
    fig.add_trace(go.Scatter(
        x=[market_open_time, market_close_time],
        y=[max_negative_net_gex_strike, max_negative_net_gex_strike],
        mode='lines',
        line=dict(color='#ff0000', width=2, dash='dash'),
        name=f'Max Neg GEX ({max_negative_net_gex_strike:.2f})'
    ))

    # Макс положительный Net GEX (зеленый) - NEW
    fig.add_trace(go.Scatter(
        x=[market_open_time, market_close_time],
        y=[max_positive_net_gex_strike, max_positive_net_gex_strike],
        mode='lines',
        line=dict(color='#00ff00', width=2, dash='dash'),
        name=f'Max Pos GEX ({max_positive_net_gex_strike:.2f})'
    ))

    # Макс AG
    fig.add_trace(go.Scatter(
        x=[market_open_time, market_close_time],
        y=[max_ag_strike, max_ag_strike],
        mode='lines',
        line=dict(color='#ab47bc', width=2, dash='dash'),
        name=f'Max AG ({max_ag_strike:.2f})'
    ))

    # 3. G-FLIP ЗОНА
    if g_flip_zone:
        g_flip_lower = g_flip_zone - (strike_step / 2)
        g_flip_upper = g_flip_zone + (strike_step / 2)
        fig.add_trace(go.Scatter(
            x=[market_open_time, market_close_time, market_close_time, market_open_time],
            y=[g_flip_lower, g_flip_lower, g_flip_upper, g_flip_upper],
            fill="toself",
            fillcolor="rgba(102, 187, 106, 0.2)",
            line=dict(color="rgba(102, 187, 106, 0.5)"),
            mode="lines",
            name=f'G-Flip Zone ({g_flip_zone:.2f})',
            hoverinfo="none",
        ))

    # 4. СТАТИЧЕСКИЕ УРОВНИ (рассчитанные по всей гамме)
    resistance_levels, support_levels = calculate_static_levels(options_data, spot_price)
    fig = add_static_levels_to_chart(fig, resistance_levels, support_levels, market_open_time, market_close_time)

    # Настраиваем layout графика
    fig.update_layout(
        title=f"{ticker}",
        xaxis=dict(
            title="Time",
            type='date',
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            rangeslider=dict(visible=False),
            autorange=False,
            range=[market_open_time, market_close_time],
            fixedrange=True
        ),
        yaxis=dict(
            title="Price",
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            range=[lower_limit, upper_limit],
            fixedrange=True
        ),
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font=dict(color='white'),
        hovermode='x unified',
        margin=dict(l=50, r=50, b=50, t=80),
        dragmode=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10)
        )
    )

    # Добавляем водяной знак
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        text="Quant Power",
        showarrow=False,
        font=dict(size=80, color="rgba(255, 255, 255, 0.1)"),
        textangle=0,
    )

    return fig


# Callback for updating date dropdown on OI Volume page
@app.callback(
    [Output('date-dropdown-oi-volume', 'options'),
     Output('date-dropdown-oi-volume', 'value')],
    [Input('search-button-oi-volume', 'n_clicks'),
     Input('ticker-input-oi-volume', 'n_submit')],
    [State('ticker-input-oi-volume', 'value')],
    prevent_initial_call=False
)
def update_oi_volume_dates(n_clicks, n_submit, ticker):
    ctx = dash.callback_context

    if not ctx.triggered:
        ticker = 'SPX'
    elif not ticker:
        ticker = 'SPX'

    ticker = normalize_ticker(ticker)

    # Get dates from file
    file_dates = load_expirations_from_file(ticker)

    # Get dates from yfinance
    yfinance_dates = get_yfinance_expirations(ticker)

    # Combine dates, preserving order and uniqueness
    if file_dates:
        # If we have file dates, only add yfinance dates that are newer than the last file date
        last_file_date = datetime.strptime(file_dates[-1], '%Y-%m-%d')
        additional_dates = [
            date for date in yfinance_dates
            if datetime.strptime(date, '%Y-%m-%d') > last_file_date
        ]
        available_dates = file_dates + additional_dates
    else:
        # If no file or empty file, use yfinance dates only
        available_dates = yfinance_dates

    if not available_dates:
        print(f"No expiration dates available for {ticker}")
        return [], []

    options = [{'label': date, 'value': date} for date in available_dates]
    return options, [available_dates[0]] if available_dates else []


# Callback for updating selected parameters on OI Volume page
@app.callback(
    [Output('selected-params-oi-volume', 'data'),
     Output('btn-volume-spread', 'className'),
     Output('btn-call-oi-oi-volume', 'className'),
     Output('btn-put-oi-oi-volume', 'className'),
     Output('btn-call-vol-oi-volume', 'className'),
     Output('btn-put-vol-oi-volume', 'className')],
    [Input('btn-volume-spread', 'n_clicks'),
     Input('btn-call-oi-oi-volume', 'n_clicks'),
     Input('btn-put-oi-oi-volume', 'n_clicks'),
     Input('btn-call-vol-oi-volume', 'n_clicks'),
     Input('btn-put-vol-oi-volume', 'n_clicks')],
    State('selected-params-oi-volume', 'data')
)
def update_selected_params_oi_volume(btn_vol_spread, btn_call_oi, btn_put_oi, btn_call_vol, btn_put_vol,
                                     selected_params):
    ctx = dash.callback_context
    if not ctx.triggered:
        return (selected_params,
                "active", "parameter-button", "parameter-button",
                "parameter-button", "parameter-button")

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    button_map = {
        "btn-volume-spread": "Volume Spread",
        "btn-call-oi-oi-volume": "Call OI",
        "btn-put-oi-oi-volume": "Put OI",
        "btn-call-vol-oi-volume": "Call Volume",
        "btn-put-vol-oi-volume": "Put Volume"
    }

    param = button_map.get(button_id)

    if param:
        if param in selected_params:
            selected_params.remove(param)
        else:
            selected_params.append(param)

    button_classes = {
        "btn-volume-spread": "active" if "Volume Spread" in selected_params else "parameter-button",
        "btn-call-oi-oi-volume": "active" if "Call OI" in selected_params else "parameter-button",
        "btn-put-oi-oi-volume": "active" if "Put OI" in selected_params else "parameter-button",
        "btn-call-vol-oi-volume": "active" if "Call Volume" in selected_params else "parameter-button",
        "btn-put-vol-oi-volume": "active" if "Put Volume" in selected_params else "parameter-button"
    }

    return (selected_params,
            button_classes["btn-volume-spread"],
            button_classes["btn-call-oi-oi-volume"],
            button_classes["btn-put-oi-oi-volume"],
            button_classes["btn-call-vol-oi-volume"],
            button_classes["btn-put-vol-oi-volume"])


# Callback for updating OI Volume chart
@app.callback(
    Output('oi-volume-chart', 'figure'),
    [Input('search-button-oi-volume', 'n_clicks'),
     Input('ticker-input-oi-volume', 'n_submit'),
     Input('date-dropdown-oi-volume', 'value'),
     Input('selected-params-oi-volume', 'data')],
    [State('ticker-input-oi-volume', 'value')]
)
def update_oi_volume_chart(n_clicks, n_submit, dates, selected_params, ticker):
    ctx = dash.callback_context
    if not ctx.triggered or not dates or not selected_params:
        return go.Figure()

    ticker = normalize_ticker(ticker)

    # Get current VIX value
    current_vix = get_current_vix()

    # Determine price range based on VIX and ticker type
    if ticker == "^SPX":
        if current_vix < 20:
            price_range = 0.014  # 1.4%
        elif 20 <= current_vix < 25:
            price_range = 0.018  # 1.8%
        elif 25 <= current_vix < 30:
            price_range = 0.026  # 2.6%
        else:
            price_range = 0.04  # 4%
    elif ticker in ["^NDX", "^RUT", "^Dia"]:
        price_range = 0.017
    elif ticker in ["SPY", "QQQ", "DIA", "XSP", "IWM"]:
        price_range = 0.03
    elif ticker in ["^VIX"]:
        price_range = 0.5
    else:
        price_range = 0.12

    options_data, _, spot_price, _ = get_option_data(ticker, dates)
    if options_data is None or options_data.empty:
        return go.Figure()

    if spot_price:
        left_limit = spot_price - (spot_price * price_range)
        right_limit = spot_price + (spot_price * price_range)
        options_data = options_data[
            (options_data['strike'] >= left_limit) & (options_data['strike'] <= right_limit)
            ]

    # Calculate Volume Spread
    options_data['Volume Spread'] = (
            (options_data['Call Volume']) -
            (options_data['Put Volume'])
    ).round(1)

    # Create figure
    fig = go.Figure()

    # Original logic for displaying parameters
    for parameter in selected_params:
        hover_texts = [
            f"Strike: {strike}<br>Call OI: {coi}<br>Put OI: {poi}<br>Call Volume: {cvol}<br>Put Volume: {pvol}<br>{parameter}: {val}"
            for strike, coi, poi, cvol, pvol, val in zip(
                options_data['strike'],
                options_data['Call OI'],
                options_data['Put OI'],
                options_data['Call Volume'],
                options_data['Put Volume'],
                options_data[parameter]
            )
        ]

        if parameter == "Volume Spread":
            fig.add_trace(go.Bar(
                x=options_data['strike'],
                y=options_data['Volume Spread'],
                marker_color=['#22b5ff' if v >= 0 else 'red' for v in options_data['Volume Spread']],
                name="Volume Spread",
                hovertext=hover_texts,
                hoverinfo="text",
                marker=dict(line=dict(width=0))
            ))

        elif parameter == "Call OI":
            fig.add_trace(go.Scatter(
                x=options_data['strike'],
                y=options_data['Call OI'],
                mode='lines+markers',
                line=dict(shape='spline', smoothing=0.7),
                marker=dict(size=8, color='#02d432'),
                fill='tozeroy',
                name="Call OI",
                hovertext=hover_texts,
                hoverinfo="text",
                yaxis='y2'
            ))

        elif parameter == "Put OI":
            fig.add_trace(go.Scatter(
                x=options_data['strike'],
                y=options_data['Put OI'],
                mode='lines+markers',
                line=dict(shape='spline', smoothing=0.7),
                marker=dict(size=8, color='#f32d35'),
                fill='tozeroy',
                name="Put OI",
                hovertext=hover_texts,
                hoverinfo="text",
                yaxis='y2'
            ))

        elif parameter == "Call Volume":
            fig.add_trace(go.Scatter(
                x=options_data['strike'],
                y=options_data['Call Volume'],
                mode='lines+markers',
                line=dict(shape='spline', smoothing=0.7),
                marker=dict(size=8, color='#003cfe'),
                fill='tozeroy',
                name="Call Volume",
                hovertext=hover_texts,
                hoverinfo="text",
                yaxis='y2'
            ))

        elif parameter == "Put Volume":
            fig.add_trace(go.Scatter(
                x=options_data['strike'],
                y=options_data['Put Volume'],
                mode='lines+markers',
                line=dict(shape='spline', smoothing=0.7),
                marker=dict(size=8, color='#e55f04'),
                fill='tozeroy',
                name="Put Volume",
                hovertext=hover_texts,
                hoverinfo="text",
                yaxis='y2'
            ))

    if spot_price:
        fig.add_vline(
            x=spot_price,
            line_dash="solid",
            line_color="orange",
            annotation_text=f"Price: {spot_price:.2f}",
            annotation_position="top",
            annotation_font=dict(color="orange"),
        )

    # Original chart styling
    fig.update_layout(
        xaxis=dict(
            title="Strike",
            showgrid=False,
            zeroline=False,
            tickmode='array',
            tickvals=options_data['strike'].tolist(),
            tickformat='1',
            fixedrange=True
        ),
        yaxis=dict(
            title="Volume Spread",
            side="left",
            showgrid=False,
            zeroline=False,
            fixedrange=True
        ),
        yaxis2=dict(
            title="",
            side="right",
            overlaying="y",
            showgrid=False,
            zeroline=False,
            fixedrange=True
        ),
        title=f"difference between call and put volumes {ticker}",
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font=dict(color='white'),
        dragmode=False,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Add watermark
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        text="Quant Power",
        showarrow=False,
        font=dict(size=80, color="rgba(255, 255, 255, 0.1)"),
        textangle=0,
    )

    return fig

# Callback for updating OI Volume price chart
@app.callback(
    Output('oi-volume-price-chart', 'figure'),
    [Input('search-button-oi-volume', 'n_clicks'),
     Input('ticker-input-oi-volume', 'n_submit')],
    [State('ticker-input-oi-volume', 'value'),
     State('date-dropdown-oi-volume', 'value')],
    prevent_initial_call=False
)
def update_oi_volume_price_chart(n_clicks, n_submit, ticker, dates):
    ctx = dash.callback_context

    if not ctx.triggered or ticker is None:
        ticker = 'SPX'

    ticker = normalize_ticker(ticker)
    interval = '1m'
    data = get_historical_data_for_chart(ticker)

    if data.empty:
        return go.Figure()

    options_data, _, spot_price, _ = get_option_data(ticker, dates)

    if options_data is None or options_data.empty:
        return go.Figure()

    # Calculate Volume Spread
    options_data['Volume Spread'] = (
            (options_data['Call Volume']) -
            (options_data['Put Volume'])
    ).round(1)

    # Determine price range
    if ticker in ["^SPX", "^NDX", "^RUT", "^Dia"]:
        price_range = 0.01
    elif ticker in ["SPY", "QQQ", "DIA", "XSP", "IWM"]:
        price_range = 0.022
    else:
        price_range = 0.05

    if spot_price:
        left_limit = spot_price - (spot_price * price_range)
        right_limit = spot_price + (spot_price * price_range)
        options_data = options_data[
            (options_data['strike'] >= left_limit) & (options_data['strike'] <= right_limit)
            ]
    else:
        left_limit = right_limit = 0

    # Calculate VWAP
    data['CumulativeVolume'] = data['Volume'].cumsum()
    data['CumulativePV'] = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum()
    data['VWAP'] = data['CumulativePV'] / data['CumulativeVolume']

    # Get key levels
    max_call_vol_strike = options_data.loc[options_data['Call Volume'].idxmax(), 'strike']
    max_put_vol_strike = options_data.loc[options_data['Put Volume'].idxmax(), 'strike']
    max_volume_spread_strike = options_data.loc[options_data['Volume Spread'].idxmax(), 'strike']
    min_volume_spread_strike = options_data.loc[options_data['Volume Spread'].idxmin(), 'strike']

    # Market time references
    market_open_time = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
    market_close_time = datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)

    # Create figure
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="Price"
    ))


    # Add Call OI zone (left side)
    if not options_data.empty:
        call_oi_data = options_data[['strike', 'Call OI']].copy()
        call_oi_data = call_oi_data.sort_values('strike')
        max_call_oi = call_oi_data['Call OI'].max()
        if max_call_oi > 0:
            call_oi_data['Call OI Normalized'] = (call_oi_data['Call OI'] / max_call_oi) * 0.3

            fig.add_trace(go.Scatter(
                x=market_open_time + (market_close_time - market_open_time) * call_oi_data['Call OI Normalized'],
                y=call_oi_data['strike'],
                mode='lines+markers',
                line=dict(shape='spline', smoothing=0.8, color='#02d432', width=2),
                marker=dict(size=1, color='#02d432'),
                name='Call OI',
                fill='tozerox',
                fillcolor='rgba(2, 212, 50, 0.2)',
                hoverinfo='text',
                hovertext=[f'Strike: {strike:.2f}<br>Call OI: {oi:,.0f}'
                           for strike, oi in zip(call_oi_data['strike'], call_oi_data['Call OI'])]
            ))

    # Add Put OI zone (right side)
    if not options_data.empty:
        put_oi_data = options_data[['strike', 'Put OI']].copy()
        put_oi_data = put_oi_data.sort_values('strike')
        max_put_oi = put_oi_data['Put OI'].max()
        if max_put_oi > 0:
            put_oi_data['Put OI Normalized'] = (put_oi_data['Put OI'] / max_put_oi) * 0.3



            fig.add_trace(go.Scatter(
                x=market_open_time + (market_close_time - market_open_time) * put_oi_data['Put OI Normalized'],
                y=put_oi_data['strike'],
                mode='lines+markers',
                line=dict(shape='spline', smoothing=0.8, color='#f32d35', width=2),
                marker=dict(size=1, color='#f32d35'),
                name='Put OI',
                fill='tozerox',
                fillcolor='rgba(243, 45, 53, 0.2)',
                hoverinfo='text',
                hovertext=[f'Strike: {strike:.2f}<br>Put OI: {oi:,.0f}'
                           for strike, oi in zip(put_oi_data['strike'], put_oi_data['Put OI'])]
            ))

        # Add Call Volume zone (NEW - left side, above Call OI)
        if not options_data.empty:
            call_vol_data = options_data[['strike', 'Call Volume']].copy()
            call_vol_data = call_vol_data.sort_values('strike')
            max_call_vol = call_vol_data['Call Volume'].max()
            if max_call_vol > 0:
                call_vol_data['Call Vol Normalized'] = (call_vol_data['Call Volume'] / max_call_vol) * 0.3

                fig.add_trace(go.Scatter(
                    x=market_open_time + (market_close_time - market_open_time) * call_vol_data['Call Vol Normalized'],
                    y=call_vol_data['strike'],
                    mode='lines+markers',
                    line=dict(shape='spline', smoothing=0.8, color='#003cfe', width=2,),
                    marker=dict(size=1, color='#003cfe'),
                    name='Call Volume',
                    fill='tozerox',
                    fillcolor='rgba(0, 60, 254, 0.1)',
                    hoverinfo='text',
                    hovertext=[f'Strike: {strike:.2f}<br>Call Volume: {vol:,.0f}'
                               for strike, vol in zip(call_vol_data['strike'], call_vol_data['Call Volume'])]
                ))

        # Add Put Volume zone (right side, orange) - NEW
        if not options_data.empty:
            put_vol_data = options_data[['strike', 'Put Volume']].copy()
            put_vol_data = put_vol_data.sort_values('strike')
            max_put_vol = put_vol_data['Put Volume'].max()
            if max_put_vol > 0:
                put_vol_data['Put Vol Normalized'] = (put_vol_data['Put Volume'] / max_put_vol) * 0.3

                fig.add_trace(go.Scatter(
                    x=market_open_time + (market_close_time - market_open_time) * put_vol_data[
                        'Put Vol Normalized'],
                    y=put_vol_data['strike'],
                    mode='lines+markers',
                    line=dict(shape='spline', smoothing=0.8, color='#e55f04', width=2,),
                    marker=dict(size=1, color='#e55f04'),
                    name='Put Volume',
                    fill='tozerox',
                    fillcolor='rgba(229, 95, 4, 0.1)',
                    hoverinfo='text',
                    hovertext=[f'Strike: {strike:.2f}<br>Put Volume: {vol:,.0f}'
                               for strike, vol in zip(put_vol_data['strike'], put_vol_data['Put Volume'])]
                ))
    # Add key levels
    if max_call_vol_strike:
        fig.add_trace(go.Scatter(
            x=[market_open_time, market_close_time],
            y=[max_call_vol_strike, max_call_vol_strike],
            mode='lines',
            line=dict(color='#003cfe', width=2),
            name=f'Max Call Vol: {max_call_vol_strike:.2f}'
        ))

    if max_put_vol_strike:
        fig.add_trace(go.Scatter(
            x=[market_open_time, market_close_time],
            y=[max_put_vol_strike, max_put_vol_strike],
            mode='lines',
            line=dict(color='#e55f04', width=2),
            name=f'Max Put Vol: {max_put_vol_strike:.2f}'
        ))

    if max_volume_spread_strike:
        fig.add_trace(go.Scatter(
            x=[market_open_time, market_close_time],
            y=[max_volume_spread_strike, max_volume_spread_strike],
            mode='lines',
            line=dict(color='#22b5ff', width=2, dash='dot'),
            name=f'Max Vol Spread: {max_volume_spread_strike:.2f}'
        ))

    if min_volume_spread_strike:
        fig.add_trace(go.Scatter(
            x=[market_open_time, market_close_time],
            y=[min_volume_spread_strike, min_volume_spread_strike],
            mode='lines',
            line=dict(color='red', width=2, dash='dot'),
            name=f'Min Vol Spread: {min_volume_spread_strike:.2f}'
        ))

    fig.update_layout(
        title=f"{ticker} Price with OI/Volume Levels",
        xaxis=dict(
            title="Time",
            type='date',
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            rangeslider=dict(visible=False),
            autorange=False,
            range=[market_open_time, market_close_time],
            fixedrange=True
        ),
        yaxis=dict(
            title="Price",
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            fixedrange=True
        ),
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font=dict(color='white'),
        hovermode='x unified',
        margin=dict(l=50, r=50, b=50, t=80),
        dragmode=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10)
        )
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        text="Quant Power",
        showarrow=False,
        font=dict(size=80, color="rgba(255, 255, 255, 0.1)"),
        textangle=0,
    )

    return fig

# Callback для обновления таблицы Options Summary
@app.callback(
    Output('options-summary-table', 'data'),
    [Input('url', 'pathname')]  # Используем изменение URL как триггер
)
def update_options_summary_table(pathname):
    if pathname == '/options-summary':
        return get_pc_ratio_data()
    return []


@cache.memoize(timeout=600)
def get_pc_ratio_data():
    # Определяем индексы и ETF
    indices_etfs = ["SPX", "SPY", "QQQ", "VIX", "DIA", "IWM", "RUT"]
    # Показываем данные для всех запрошенных тикеров
    tickers = indices_etfs + [
        "NVDA", "AAPL", "TSLA", "META", "MSFT", "GOOG",
        "AMZN", "AVGO", "WMT", "JPM", "MU", "BA", "SNOW",
        "UBER", "ROKU", "PLTR", "GS", "COIN"
    ]

    table_data = []

    for ticker in tickers:
        normalized_ticker = normalize_ticker(ticker)

        try:
            # ИСПРАВЛЕНИЕ: используем get_yfinance_spot_price для получения правильной цены SPX
            price = get_yfinance_spot_price(normalized_ticker)
        except:
            price = None

        options_data, _, spot_price, _ = get_option_data(normalized_ticker, [])

        if options_data is None or options_data.empty or price is None:
            continue

        # Определяем диапазон цены в зависимости от типа тикера
        if ticker in indices_etfs:
            price_range = 0.01  # 1% для индексов и ETF
        else:
            price_range = 0.05  # 5% для акций

        # Фильтруем данные опционов в пределах диапазона цены (только для Resistance и Support)
        lower_limit = price * (1 - price_range)
        upper_limit = price * (1 + price_range)
        filtered_data = options_data[
            (options_data['strike'] >= lower_limit) &
            (options_data['strike'] <= upper_limit)
            ]

        if filtered_data.empty:
            continue

        # Рассчитываем Resistance (максимальный Call Volume в пределах диапазона)
        max_call_vol_strike = filtered_data.loc[filtered_data['Call Volume'].idxmax(), 'strike']

        # Рассчитываем Support (максимальный Put Volume или минимальный Net GEX в пределах диапазона)
        max_put_vol_strike = filtered_data.loc[filtered_data['Put Volume'].idxmax(), 'strike']
        max_negative_net_gex_strike = filtered_data.loc[filtered_data['Net GEX'].idxmin(), 'strike']

        if max_put_vol_strike < max_negative_net_gex_strike:
            support_strike = max_put_vol_strike
        else:
            support_strike = max_negative_net_gex_strike

        # Суммируем Call OI и Put OI по ВСЕМ опционам (не только в пределах диапазона)
        call_oi_amount = options_data['Call OI'].sum()
        put_oi_amount = options_data['Put OI'].sum()

        # Рассчитываем P/C Ratio
        pc_ratio = put_oi_amount / call_oi_amount if call_oi_amount != 0 else float('inf')

        table_data.append({
            'Ticker': ticker,
            'Price': round(price, 2),
            'Resistance': round(max_call_vol_strike, 2),
            'Support': round(support_strike, 2),
            'Call OI Amount': f"{call_oi_amount:,.0f}",
            'Put OI Amount': f"{put_oi_amount:,.0f}",
            'P/C Ratio': f"{pc_ratio:.2f}"
        })

    return table_data


# Запуск приложения
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False)

