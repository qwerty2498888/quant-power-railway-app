import yfinance as yf
import pandas as pd


def get_data(ticker):
    """Получение данных из Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if hist.empty:
            return None

        return hist
    except Exception as e:
        print(f"Ошибка получения данных: {str(e)}")
        return None