# Добавьте эти импорты в начало файла
from threading import Thread
import time
from .detector import detect_levels
from .data_fetcher import get_data


class AlertManager:
    def __init__(self):
        self.active_signals = set()

    def check_signals(self, ticker):
        """Основной метод проверки сигналов"""
        while True:
            try:
                # 1. Получаем данные
                data = get_data(ticker)
                if data is None:
                    time.sleep(60)
                    continue

                # 2. Определяем уровни
                spot_price = data['Close'].iloc[-1]  # Пример для Yahoo Finance
                levels = detect_levels(data, spot_price)

                # 3. Фильтруем новые сигналы
                new_signals = [
                    s for s in levels
                    if s['strike'] not in self.active_signals
                       and s['strength'] > 7
                ]

                # 4. Обрабатываем новые сигналы
                for signal in new_signals:
                    print(f"🚨 Новый сигнал: {signal['type']} на {signal['strike']}")
                    self.active_signals.add(signal['strike'])

                # 5. Пауза между проверками
                time.sleep(60)

            except Exception as e:
                print(f"⚠️ Ошибка в мониторинге: {str(e)}")
                time.sleep(10)


# Создаем экземпляр менеджера
alert_manager = AlertManager()

# Запускаем мониторинг в фоне (убедитесь, что это выполняется только один раз)
if __name__ == "__alert_manager__":
    Thread(target=alert_manager.check_signals, args=('SPX',), daemon=True).start()
if __name__ == "__main__":
    # Тестовый запуск
    alert_manager.check_signals('SPX')