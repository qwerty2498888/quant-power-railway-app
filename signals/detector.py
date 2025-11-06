import numpy as np
from scipy.signal import find_peaks


def find_gamma_flips(gex_series, sensitivity=3):
    """Находит точки смены знака гаммы"""
    flips = []
    for i in range(len(gex_series) - sensitivity):
        if gex_series[i] * gex_series[i + sensitivity] < 0:
            flips.append(i)
    return flips


def calculate_strength(level_type, distance_to_price):
    """Рассчитывает силу сигнала от 1 до 10"""
    type_weights = {
        'gamma_flip': 9,
        'call_wall': 7,
        'put_wall': 7,
        'ag_peak': 8
    }
    distance_weight = 1 - min(distance_to_price / 0.05, 1)  # Нормализуем расстояние

    return type_weights.get(level_type, 5) * distance_weight


def detect_levels(options_data, price):
    levels = []

    # 1. Gamma Flips
    flips = find_gamma_flips(options_data['Net GEX'])
    for idx in flips:
        strike = options_data.iloc[idx]['strike']
        levels.append({
            'type': 'gamma_flip',
            'strike': strike,
            'strength': calculate_strength('gamma_flip', abs(strike - price) / price)
        })

    # 2. Call/Put Walls (аналогично для puts)
    call_walls = options_data.nlargest(3, 'Call Volume')
    for _, row in call_walls.iterrows():
        levels.append({
            'type': 'call_wall',
            'strike': row['strike'],
            'strength': calculate_strength('call_wall', abs(row['strike'] - price) / price)
        })

    return sorted(levels, key=lambda x: -x['strength'])  # Сортируем по силе