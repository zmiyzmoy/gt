#!/usr/bin/env python3
"""
Тестовый скрипт для проверки конфигурации OpenSpiel.
Проверяет правильность параметров universal_poker.
"""

import sys
import os
import json
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("spiel_test")

try:
    import pyspiel
    from open_spiel.python import rl_environment
except ImportError as e:
    logger.error(f"Ошибка импорта OpenSpiel: {e}")
    logger.error("Убедитесь, что OpenSpiel установлен. Выход.")
    sys.exit(1)

def check_poker_game():
    """Проверяет наличие и доступность игры universal_poker."""
    logger.info("Проверка регистрации игры universal_poker...")
    
    # Получаем список зарегистрированных игр
    games = pyspiel.registered_games()
    logger.info(f"Всего зарегистрировано игр: {len(games)}")
    
    # Ищем universal_poker
    poker_game = next((g for g in games if g.short_name == "universal_poker"), None)
    if poker_game:
        logger.info(f"Найдена игра universal_poker: {poker_game.short_name}")
        logger.info(f"Параметры по умолчанию: {poker_game.default_loadable}")
        return True
    else:
        logger.error("Игра universal_poker не найдена!")
        return False

def test_simple_game_config():
    """Тестирует простую конфигурацию покера с правильными параметрами."""
    logger.info("Тестирование простой конфигурации игры...")
    
    # Базовая строка конфигурации
    config_string = (
        "universal_poker("
        "betting=nolimit,"  # Тип ставок: nolimit, limit или potlimit
        "numPlayers=2,"     # Количество игроков
        "numRounds=4,"      # Количество раундов торгов (улиц)
        "numSuits=4,"       # Количество мастей
        "numRanks=13,"      # Количество рангов карт
        "numHoleCards=2,"   # Количество карманных карт
        "numBoardCards=0 3 1 1,"  # Количество общих карт на каждой улице
        "stack=1000 1000,"  # Стеки для каждого игрока
        "blind=50 100"      # Блайнды для каждого игрока
        ")"
    )
    
    logger.info(f"Строка конфигурации: {config_string}")
    
    try:
        # Загружаем игру
        game = pyspiel.load_game(config_string)
        logger.info(f"Игра создана успешно: {game}")
        
        # Создаем среду
        env = rl_environment.Environment(game)
        time_step = env.reset()
        
        logger.info(f"Среда создана, текущий игрок: {time_step.observations['current_player']}")
        logger.info(f"Информационное состояние: {time_step.observations['info_state'][0][:10]}...")
        
        # Проверяем действия
        legal_actions = time_step.observations['legal_actions'][0]
        logger.info(f"Доступные действия: {legal_actions}")
        
        # Выполняем одно действие
        if legal_actions:
            next_time_step = env.step([legal_actions[0]])
            logger.info("Действие выполнено успешно")
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при создании игры: {e}")
        logger.error(traceback.format_exc())
        return False

def test_detailed_config():
    """Тестирует детальную конфигурацию с дополнительными параметрами."""
    logger.info("Тестирование детальной конфигурации игры...")
    
    # Создаем полную конфигурацию
    config = {
        "betting": "nolimit",
        "numPlayers": 8,
        "numRounds": 4,
        "numSuits": 4,
        "numRanks": 13,
        "numHoleCards": 2,
        "numBoardCards": [0, 3, 1, 1],  # 0 на префлопе, 3 на флопе, 1 на терне, 1 на ривере
        "stack": [10000] * 8,           # Стеки для каждого игрока
        "blind": [50, 100] + [0] * 6    # Блайнды: SB, BB, остальные 0
    }
    
    logger.info(f"Детальная конфигурация: {json.dumps(config, indent=2)}")
    
    # Преобразуем списки в строки для OpenSpiel
    processed_config = {}
    for key, value in config.items():
        if isinstance(value, list):
            processed_config[key] = " ".join(map(str, value))
        else:
            processed_config[key] = value
    
    logger.info(f"Обработанная конфигурация: {json.dumps(processed_config, indent=2)}")
    
    try:
        # Создаем среду
        env = rl_environment.Environment("universal_poker", **processed_config)
        time_step = env.reset()
        
        logger.info(f"Среда с детальной конфигурацией создана успешно")
        logger.info(f"Текущий игрок: {time_step.observations['current_player']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при создании детальной конфигурации: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Основная функция для запуска тестов."""
    logger.info("=" * 60)
    logger.info("Тестирование конфигурации OpenSpiel для Poker")
    logger.info("=" * 60)
    
    # Печатаем версию OpenSpiel
    try:
        version = pyspiel.__version__
        logger.info(f"Версия OpenSpiel: {version}")
    except:
        logger.warning("Не удалось определить версию OpenSpiel")
    
    # Проверка игры
    if not check_poker_game():
        logger.error("Не найдена игра universal_poker. Дальнейшие тесты невозможны.")
        return
    
    # Тестируем простую конфигурацию
    if test_simple_game_config():
        logger.info("Тест простой конфигурации пройден успешно!")
    else:
        logger.error("Тест простой конфигурации не пройден!")
    
    # Тестируем детальную конфигурацию
    if test_detailed_config():
        logger.info("Тест детальной конфигурации пройден успешно!")
    else:
        logger.error("Тест детальной конфигурации не пройден!")
    
    logger.info("=" * 60)
    logger.info("Тестирование конфигурации OpenSpiel завершено")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()