# environment.py
import numpy as np
import gymnasium as gym
from open_spiel.python import rl_environment
import pyspiel
from collections import deque
import logging
from typing import Dict, Any, List, Tuple
import traceback # Добавлено для более детального логирования ошибок

# Настройка логирования
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # Раскомментируй для детальной отладки
# Добавление обработчика для вывода логов (если еще не настроено глобально)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO) # Установите уровень по умолчанию

class OpponentStats:
    """Класс для хранения и обновления базовой статистики оппонентов."""
    def __init__(self, num_players):
        self.num_players = num_players
        self.stats = {}
        # VPIP, PFR, Aggression Factor (AF), Hands Played
        self.feature_size = 4
        self.reset()

    def reset(self):
        """Сбрасывает статистику для всех игроков."""
        self.stats = {}
        for i in range(self.num_players):
             self._ensure_player_stats(i)

    def _ensure_player_stats(self, player_id):
         """Гарантирует наличие записи для игрока."""
         player_id_int = int(player_id)
         if player_id_int not in self.stats:
             # Инициализируем полным набором ключей
             self.stats[player_id_int] = {
                 'vpip': 0.0, 'pfr': 0.0, 'af': 1.0, 'hands': 0,
                 'agg_actions': 0, 'pass_actions': 0,
                 'vpip_opportunities': 0, 'pfr_opportunities': 0,
                 'vpip_actions': 0, 'pfr_actions': 0,
                 'vpip_acted_this_hand': False, 'pfr_acted_this_hand': False
             }

    def get_features(self, player_id):
        """Возвращает нормализованную статистику для игрока как numpy array float32."""
        player_id_int = int(player_id)
        self._ensure_player_stats(player_id_int) # Гарантируем наличие ключа
        player_stats = self.stats[player_id_int]

        # Расчет VPIP/PFR/AF на лету
        hands = player_stats.get('hands', 0)
        vpip_opps = player_stats.get('vpip_opportunities', 0)
        pfr_opps = player_stats.get('pfr_opportunities', 0)
        vpip_actions = player_stats.get('vpip_actions', 0)
        pfr_actions = player_stats.get('pfr_actions', 0)
        agg_actions = player_stats.get('agg_actions', 0)
        pass_actions = player_stats.get('pass_actions', 0)

        vpip = float(vpip_actions / vpip_opps) if vpip_opps > 0 else 0.0
        pfr = float(pfr_actions / pfr_opps) if pfr_opps > 0 else 0.0

        if pass_actions > 0:
            af = float(agg_actions / pass_actions)
        elif agg_actions > 0:
            af = 5.0 # Условное высокое значение, если были только агрессивные действия
        else:
            af = 1.0 # Нейтральное значение по умолчанию

        # Нормализация (примерная)
        norm_vpip = np.clip(vpip, 0.0, 1.0)
        norm_pfr = np.clip(pfr, 0.0, 1.0)
        norm_af = np.clip(af / 5.0, 0.0, 1.0) # Делим на 5 для масштабирования AF к ~[0,1]
        norm_hands = np.clip(hands / 100.0, 0.0, 1.0) # Нормализуем количество рук (до 100)

        return np.array([norm_vpip, norm_pfr, norm_af, norm_hands], dtype=np.float32)

    def record_opportunity(self, player_id, is_vpip_opp, is_pfr_opp):
        """Записывает возможность для VPIP/PFR."""
        player_id_int = int(player_id)
        self._ensure_player_stats(player_id_int)
        if is_vpip_opp:
             self.stats[player_id_int]['vpip_opportunities'] = self.stats[player_id_int].get('vpip_opportunities', 0) + 1
             # Сбрасываем флаг действия в НАЧАЛЕ возможности, а не в конце руки
             # self.stats[player_id_int]['vpip_acted_this_hand'] = False # Этот сброс лучше делать в finalize_hand_stats или reset

        if is_pfr_opp:
             self.stats[player_id_int]['pfr_opportunities'] = self.stats[player_id_int].get('pfr_opportunities', 0) + 1
             # self.stats[player_id_int]['pfr_acted_this_hand'] = False # Этот сброс лучше делать в finalize_hand_stats или reset


    def update_on_action(self, player_id, action_type, street, is_voluntary_action):
        """Обновляет счетчики действий. Улучшена логика VPIP/PFR."""
        player_id_int = int(player_id)
        self._ensure_player_stats(player_id_int)
        stats = self.stats[player_id_int]

        # VPIP: Засчитывается первое добровольное действие (не фолд) на префлопе
        if street == 0 and is_voluntary_action and not stats['vpip_acted_this_hand']:
             stats['vpip_actions'] = stats.get('vpip_actions', 0) + 1
             stats['vpip_acted_this_hand'] = True # Отмечаем, что VPIP засчитан для этой руки

        # PFR: Засчитывается первый рейз на префлопе
        if street == 0 and action_type == 'raise' and not stats['pfr_acted_this_hand']:
             stats['pfr_actions'] = stats.get('pfr_actions', 0) + 1
             stats['pfr_acted_this_hand'] = True # Отмечаем, что PFR засчитан для этой руки

        # AF: Считаем агрессивные (бет/рейз) и пассивные (колл) действия ПОСТФЛОП
        if street > 0:
            if action_type == 'raise': # Включая 'bet'
                stats['agg_actions'] = stats.get('agg_actions', 0) + 1
            elif action_type == 'call': # Только колл, не чек
                stats['pass_actions'] = stats.get('pass_actions', 0) + 1
            # Чек не учитывается ни в числителе, ни в знаменателе AF

    def finalize_hand_stats(self, involved_players):
         """Вызывается в конце руки для увеличения счетчика рук и сброса флагов."""
         for player_id in involved_players:
             player_id_int = int(player_id)
             self._ensure_player_stats(player_id_int)
             self.stats[player_id_int]['hands'] += 1
             # Сбрасываем флаги действий в текущей руке для следующей раздачи
             self.stats[player_id_int]['vpip_acted_this_hand'] = False
             self.stats[player_id_int]['pfr_acted_this_hand'] = False


class PokerEnv(gym.Env):
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 4}

    def __init__(self, env_config: dict):
        super().__init__()
        logger.info("Initializing PokerEnv...")

        # Проверка наличия ключа 'config' (экземпляр PokerConfig)
        if "config" not in env_config:
             # Попытка найти конфиг уровнем выше (часто бывает в RLlib)
             if "env_config" in env_config:
                 env_config = env_config["env_config"]
             else:
                 raise ValueError("Missing 'config' key (PokerConfig instance) in env_config dictionary.")

        if "config" not in env_config:
            raise ValueError("Missing 'config' key (PokerConfig instance) in env_config dictionary even after checking nested 'env_config'.")

        self.config = env_config["config"] # Экземпляр PokerConfig (или подобного класса/dict)
        self.dtype = env_config.get("dtype", np.float32)

        # --- OpenSpiel Initialization ---
        # Доступ к параметрам игры через атрибуты конфига
        game_params_from_config = self.config.game_config if hasattr(self.config, 'game_config') else {}
        game_name = self.config.game_name if hasattr(self.config, 'game_name') else "leduc_poker" # Примерное значение по умолчанию

        processed_game_params = {}
        for k, v in game_params_from_config.items():
            # Преобразование списков/кортежей/numpy массивов в строки с пробелами
            if k in ["numBoardCards", "blind", "stack"] and isinstance(v, (list, tuple, np.ndarray)):
                processed_game_params[k] = " ".join(map(str, v))
            # Обработка булевых значений
            elif isinstance(v, bool):
                 processed_game_params[k] = str(v).lower()
            # Остальные значения просто преобразуем в строку (или оставляем как есть, если это числа)
            elif isinstance(v, (int, float, np.number)):
                 processed_game_params[k] = v
            else:
                 processed_game_params[k] = str(v)

        logger.info(f"Attempting to load OpenSpiel game: '{game_name}' with processed params: {processed_game_params}")
        try:
            # Загрузка игры с параметрами
            self.game = pyspiel.load_game(game_name, processed_game_params)
            logger.info(f"Game '{game_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load game '{game_name}' with params: {str(processed_game_params)} - Error: {e}", exc_info=True)
            raise # Перевыброс исключения, чтобы прервать инициализацию

        try:
            # Создание базовой среды OpenSpiel RL
            self._base_env = rl_environment.Environment(game=self.game, include_full_state=False) # include_full_state=False важно для info_state
            logger.info("OpenSpiel RL Environment created successfully.")
            # Получение спецификаций действий и наблюдений
            self._action_spec = self._base_env.action_spec()
            self._observation_spec = self._base_env.observation_spec() # Это спецификация от OpenSpiel, не финальная!
        except Exception as e:
            logger.error(f"Failed to create OpenSpiel RL Environment: {e}", exc_info=True)
            raise

        # Инициализация статистики оппонентов
        num_players_actual = self.game.num_players() # Получаем реальное кол-во игроков из загруженной игры
        self.stats = OpponentStats(num_players_actual)

        # Определение пространств Gymnasium (action_space, observation_space)
        self._define_spaces(num_players_actual)

        # Имена позиций (пример)
        self._position_names = ['SB', 'BB', 'UTG', 'MP1', 'MP2', 'MP3', 'CO', 'BTN'][:num_players_actual]
        self.current_street = 0
        self._current_time_step = None # Хранит текущий TimeStep от OpenSpiel

        logger.info(f"PokerEnv initialized for {num_players_actual} players.")

    def _reset_metrics(self):
        """Сбрасывает метрики для нового эпизода."""
        self.episode_rewards = {p: 0.0 for p in range(self.game.num_players())}
        self.current_hand_info = {
            'actions_by_player': {p: [] for p in range(self.game.num_players())},
            'involved_players': set(), # Игроки, сделавшие хотя бы одно действие (не фолд сразу)
            'pot_size': 0,
            'current_player': -1,
            'last_raiser': -1, # Отслеживание последнего агрессора для логики ставок
            'can_check': {} # Отслеживание, может ли игрок чекать
        }
        # Сброс статистики в начале каждой руки (если нужно обнулять чаще, чем раз в reset())
        # self.stats.reset() # Раскомментировать, если статистика нужна только в пределах одной руки

    def _define_spaces(self, num_players):
        """Определяет action_space и observation_space для Gymnasium."""
        # Пространство действий
        num_actions = self._action_spec["num_actions"]
        self.action_space = gym.spaces.Discrete(num_actions)
        logger.info(f"Action space defined: Discrete({self.action_space.n})")

        # Пространство наблюдений
        try:
            # Получаем ПРИМЕР одного наблюдения, чтобы определить его размер
            time_step = self._base_env.reset()
            # Определяем первого игрока (может быть Chance или Player)
            player_id = time_step.observations["current_player"]
            while player_id == pyspiel.PlayerId.CHANCE:
                 # Если начинает Chance node, делаем шаг, чтобы получить состояние первого игрока
                 time_step = self._base_env.step([]) # Пустой шаг для Chance

            player_id = time_step.observations["current_player"]
            if player_id < 0 or player_id == pyspiel.PlayerId.TERMINAL:
                # Если игра сразу закончилась или не удалось найти игрока
                raise ValueError("Could not determine initial player ID after reset.")

            # Получаем вектор наблюдения для этого первого игрока
            processed_obs_vector = self._get_one_observation(time_step, player_id)
            obs_size = len(processed_obs_vector)
            if obs_size <= 0:
                 raise ValueError(f"Calculated observation size is invalid: {obs_size}")

            logger.info(f"Successfully calculated Observation space size: {obs_size}")
            # Определяем Box с вычисленным размером
            # Используем -1 и 1 как границы для нормализованных данных, если возможно,
            # иначе -np.inf и np.inf, если не уверены в масштабе всех фич
            self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_size,), dtype=self.dtype)
            # self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=self.dtype) # Более безопасный вариант

        except Exception as e:
             logger.error(f"CRITICAL: Failed to determine observation space size dynamically: {e}. Using fallback size 512.", exc_info=True)
             # Запасной вариант с фиксированным размером, если динамический расчет не удался
             fallback_size = 512
             self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(fallback_size,), dtype=self.dtype)
             logger.warning(f"Observation space set to fallback Box(shape=({fallback_size},))")


    def _get_one_observation(self, time_step, player_id):
         """Собирает полный вектор наблюдения для заданного игрока."""
         if not hasattr(self, 'observation_space'):
              logger.error("Observation space not defined when _get_one_observation was called!")
              # Не можем вернуть осмысленный вектор, если не знаем его размер
              raise RuntimeError("Observation space is not initialized.")

         obs_shape = self.observation_space.shape
         fallback_obs = np.zeros(obs_shape, dtype=self.dtype) # На случай ошибок

         # Проверка player_id
         if player_id < 0 or player_id >= self.game.num_players():
             # Это может произойти, если time_step указывает на конец игры (TERMINAL) или Chance
             logger.warning(f"Invalid player_id ({player_id}) requested in _get_one_observation. Returning zeros.")
             return fallback_obs

         try:
            # 1. Базовый вектор наблюдения (info_state) от OpenSpiel
            base_obs_list = time_step.observations["info_state"][player_id]
            base_obs = np.array(base_obs_list, dtype=self.dtype)

            # 2. Фича позиции игрока (нормализованная)
            # Нормализуем от 0 до 1
            position_feature = np.array([float(player_id) / max(1, self.game.num_players() - 1)], dtype=self.dtype)

            # 3. Фичи статистики оппонентов
            opponent_features_list = []
            for pid in range(self.game.num_players()):
                if pid != player_id:
                    opponent_features_list.extend(self.stats.get_features(pid)) # Добавляем [vpip, pfr, af, hands] для каждого
            opponent_features = np.array(opponent_features_list, dtype=self.dtype)
            # Проверка размера статов оппонентов
            expected_opp_len = (self.game.num_players() - 1) * self.stats.feature_size
            if len(opponent_features) != expected_opp_len:
                 logger.warning(f"Opponent features length mismatch! Got {len(opponent_features)}, expected {expected_opp_len}. Check OpponentStats.feature_size.")
                 # Пытаемся исправить или возвращаем нули
                 opponent_features = np.pad(opponent_features, (0, expected_opp_len - len(opponent_features)), 'constant').astype(self.dtype)
                 if len(opponent_features) > expected_opp_len:
                      opponent_features = opponent_features[:expected_opp_len]


            # 4. Фича шансов банка (Pot Odds)
            pot_odds_feature = np.array([self._calculate_pot_odds(time_step, player_id)], dtype=self.dtype)

            # 5. Фича отношения стека к поту (Stack-to-Pot Ratio, SPR)
            stack_to_pot_feature = np.array([self._calculate_stack_to_pot(time_step, player_id)], dtype=self.dtype)

            # --- ПРОВЕРКА и СБОРКА Финального Вектора Наблюдения ---
            num_extra_features = len(position_feature) + len(pot_odds_feature) + len(stack_to_pot_feature) + len(opponent_features)
            expected_total_len = obs_shape[0]
            expected_base_len = expected_total_len - num_extra_features

            # КРИТИЧЕСКАЯ ПРОВЕРКА: Соответствует ли длина базового вектора ожиданиям?
            if len(base_obs) != expected_base_len:
                # --- ИСПРАВЛЕНО: Вместо padding/truncating -> ОШИБКА ---
                error_msg = (
                    f"FATAL: Observation length mismatch for player {player_id}! "
                    f"Expected total length: {expected_total_len}, Got base_obs length: {len(base_obs)}, "
                    f"Expected base_obs length: {expected_base_len}, "
                    f"Num extra features: {num_extra_features} "
                    f"(Pos: {len(position_feature)}, PO: {len(pot_odds_feature)}, SPR: {len(stack_to_pot_feature)}, OppStats: {len(opponent_features)})"
                )
                logger.error(error_msg)
                # Подробный вывод в лог для отладки
                logger.debug(f"Observation space shape: {obs_shape}")
                logger.debug(f"Base obs (first 10): {base_obs[:10]}")
                logger.debug(f"Opponent features: {opponent_features}")
                # Генерируем исключение, чтобы остановить выполнение и исследовать
                raise ValueError(error_msg)
                # return fallback_obs # Старый вариант: тихо вернуть нули
                # --- Конец исправления ---

            # Сборка финального вектора, если размеры сошлись
            full_obs = np.concatenate([
                base_obs,
                position_feature,
                pot_odds_feature,
                stack_to_pot_feature,
                opponent_features
            ]).astype(self.dtype)

            # Финальная проверка общей длины (на всякий случай)
            if len(full_obs) != expected_total_len:
                 logger.error(f"FINAL observation length mismatch after concatenation! Got {len(full_obs)}, expected {expected_total_len}. Returning zeros.")
                 return fallback_obs

            # Убедимся, что значения находятся в пределах [-1, 1], если пространство определено так
            if self.observation_space.low.min() >= -1 and self.observation_space.high.max() <= 1:
                 full_obs = np.clip(full_obs, -1.0, 1.0)

            return full_obs

         except Exception as e:
            logger.error(f"Error constructing observation for player {player_id}: {e}\n{traceback.format_exc()}")
            return fallback_obs # Возвращаем нули в случае любой другой ошибки

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Инициализация генератора случайных чисел Gymnasium
        logger.debug("Resetting environment...")
        try:
            self._current_time_step = self._base_env.reset()
            self.current_street = 0
            self._reset_metrics() # Сброс наград и информации о текущей руке

            # Запись возможностей VPIP/PFR (упрощенная)
            # TODO: Уточнить логику определения VPIP/PFR opportunity на основе позиции/блайндов
            is_vpip_opp = True # Предполагаем, что у всех есть VPIP opp на старте (кроме BB может быть?)
            is_pfr_opp = True # Предполагаем, что у всех есть PFR opp на старте
            for p in range(self.game.num_players()):
                self.stats.record_opportunity(p, is_vpip_opp, is_pfr_opp)

            # Обработка начального Chance узла, если он есть
            while self._current_time_step.observations["current_player"] == pyspiel.PlayerId.CHANCE:
                 logger.debug("Handling initial chance node.")
                 self._current_time_step = self._base_env.step([])

            # Определение текущего игрока
            current_player = self._current_time_step.observations["current_player"]
            if self._current_time_step.last() or current_player < 0:
                logger.warning("Environment terminated immediately after reset. Returning zero observation.")
                # Игра может закончиться сразу (редко), возвращаем нулевое наблюдение
                return np.zeros(self.observation_space.shape, dtype=self.dtype), {}

            # Получение первого наблюдения для активного игрока
            obs = self._get_one_observation(self._current_time_step, current_player)
            info = self._enhance_info({}, current_player) # Добавляем доп. информацию

            logger.debug(f"Reset complete. First player: {current_player}")
            return obs.astype(self.dtype), info # Гарантируем тип float32

        except Exception as e:
            logger.error(f"Error during environment reset: {e}\n{traceback.format_exc()}")
            # В случае ошибки возвращаем нулевое наблюдение и пустой info
            return np.zeros(self.observation_space.shape, dtype=self.dtype), {}

    def step(self, action):
        # Преобразование действия, если оно numpy тип
        if isinstance(action, np.generic): action = int(action)
        # Или если это тензор PyTorch
        if hasattr(action, 'item'): action = action.item()
        action = int(action) # Финальное преобразование в int

        if self._current_time_step is None:
            logger.error("Step called before reset or after error. Resetting environment.")
            obs, info = self.reset()
            # Возвращаем состояние после сброса, но с нулевой наградой и terminated=True
            return obs.astype(self.dtype), 0.0, True, False, info

        # Проверка, не закончился ли эпизод на предыдущем шаге
        if self._current_time_step.last():
            logger.warning("Step called on a terminal state. Resetting environment.")
            obs, info = self.reset()
            # Возвращаем состояние после сброса, награду 0, terminated=True
            return obs.astype(self.dtype), 0.0, True, False, info # Truncated=False т.к. это конец эпизода

        current_player = self._current_time_step.observations["current_player"]
        if current_player < 0 : # Должен быть ID игрока
             logger.error(f"Invalid current player ID ({current_player}) in step. Resetting.")
             obs, info = self.reset()
             return obs.astype(self.dtype), 0.0, True, False, info

        logger.debug(f"Player {current_player} attempts action: {action}")

        # Проверка легальности действия
        legal_actions = self._current_time_step.observations["legal_actions"][current_player]
        if not legal_actions: # Если нет легальных действий (редко, но возможно)
             logger.warning(f"Player {current_player} has no legal actions. Treating as terminal state.")
             # Завершаем эпизод, как если бы игра закончилась
             terminated = True
             truncated = False
             reward = 0.0 # Или последняя награда из time_step, если она там есть
             if self._current_time_step.rewards: reward = float(self._current_time_step.rewards[current_player])
             obs_vector = self._get_one_observation(self._current_time_step, current_player) # Последнее наблюдение
             info = self._enhance_info({'error': 'No legal actions'}, current_player)
             # Обновляем статистику в конце руки
             involved_players = self.current_hand_info.get('involved_players', set(range(self.game.num_players())))
             self.stats.finalize_hand_stats(involved_players)
             self._current_time_step = None # Сбрасываем time_step, т.к. эпизод завершен
             return obs_vector.astype(self.dtype), reward, terminated, truncated, info

        if action not in legal_actions:
            original_action = action
            action = self.np_random.choice(legal_actions) # Выбираем случайное легальное
            logger.warning(f"Player {current_player} chose illegal action {original_action}. Corrected to random legal action: {action}. Legal actions were: {legal_actions}")

        try:
            # Обновление статистики ДО выполнения шага в среде
            action_type = self._get_action_type(action)
            # Добровольное действие: не фолд на префлопе ИЛИ не фолд/чек на постфлопе (если была ставка)
            # Упрощенная логика: любое действие кроме фолда считается влияющим на VPIP/PFR (если возможно)
            is_voluntary = (action_type != 'fold')
            self.stats.update_on_action(current_player, action_type, self.current_street, is_voluntary)

            # Обновление информации о руке
            self.current_hand_info['actions_by_player'][current_player].append(action)
            if action_type != 'fold': # Добавляем в 'involved', только если не фолд сразу
                self.current_hand_info['involved_players'].add(current_player)
            if action_type == 'raise':
                 self.current_hand_info['last_raiser'] = current_player

            # Выполнение шага в среде OpenSpiel
            self._current_time_step = self._base_env.step([action])

            # Обновление информации о руке ПОСЛЕ шага
            self.current_hand_info['pot_size'] = self._get_pot_size(self._current_time_step)
            next_player = self._current_time_step.observations["current_player"] # Может быть Chance или Terminal
            self.current_hand_info['current_player'] = next_player

            # Обработка Chance узлов
            while self._current_time_step.observations["current_player"] == pyspiel.PlayerId.CHANCE:
                 logger.debug("Handling chance node.")
                 self._current_time_step = self._base_env.step([])
                 next_player = self._current_time_step.observations["current_player"]

            # Определение улицы (раунда)
            new_street = self._get_street(self._current_time_step)
            if new_street != self.current_street:
                logger.debug(f"Street changed from {self.current_street} to {new_street}")
                self.current_street = new_street
                # Сброс некоторых счетчиков для новой улицы, если нужно (например, maxRaises per street)

            # Определение награды, завершения и следующего наблюдения
            reward = float(self._current_time_step.rewards[current_player]) if self._current_time_step.rewards else 0.0
            terminated = self._current_time_step.last()
            truncated = False # В этой среде нет усечения по времени, только конец игры

            # Получение наблюдения для СЛЕДУЮЩЕГО игрока (или текущего, если игра закончилась)
            obs_player_id = next_player if not terminated else current_player
            if obs_player_id < 0: # Если next_player == TERMINAL, используем последнего активного
                 obs_player_id = current_player

            obs_vector = self._get_one_observation(self._current_time_step, obs_player_id)

            # Обновление суммарной награды за эпизод
            self.episode_rewards[current_player] = self.episode_rewards.get(current_player, 0.0) + reward

            # Завершающие действия, если эпизод закончился
            if terminated:
                logger.debug(f"Episode terminated. Final rewards: {self._current_time_step.rewards}")
                # Засчитываем руку всем, кто участвовал
                self.stats.finalize_hand_stats(self.current_hand_info['involved_players'])
                # Суммируем финальные награды для всех
                for p, r in enumerate(self._current_time_step.rewards):
                     # Добавляем финальную награду к уже накопленной за шаги
                     # Если игрок уже получил награду на этом шаге, это может быть двойной учет?
                     # Обычно финальные награды заменяют последнюю награду шага.
                     # Проверить логику OpenSpiel rl_environment!
                     # Безопаснее просто записать финальную награду.
                     self.episode_rewards[p] = float(r) # Записываем финальный результат
                logger.info(f"Episode finished. Total rewards: {self.episode_rewards}")


            # Формирование дополнительной информации (info dict)
            info = self._enhance_info({}, obs_player_id) # Передаем ID игрока, для которого наблюдение
            if terminated:
                 info['episode_rewards'] = self.episode_rewards # Добавляем финальные награды в info

            return obs_vector.astype(self.dtype), reward, terminated, truncated, info

        except Exception as e:
            logger.error(f"Error during environment step: {e}\n{traceback.format_exc()}")
            # В случае ошибки завершаем эпизод и возвращаем нулевые значения
            return np.zeros(self.observation_space.shape, dtype=self.dtype), 0.0, True, False, {"error": str(e)}

    # --- Вспомогательные методы ---

    def _get_state(self):
        """Безопасное получение внутреннего состояния OpenSpiel (если доступно)."""
        # Доступ к _state считается использованием внутреннего API, может измениться
        return getattr(self._base_env, '_state', None)

    def _get_pot_size(self, time_step):
        """Получает текущий размер банка из состояния OpenSpiel."""
        state = self._get_state()
        # Проверяем наличие метода pot()
        if state and hasattr(state, 'pot'):
            try:
                return float(state.pot())
            except Exception as e:
                logger.warning(f"Could not get pot size: {e}")
                return 0.0
        # Альтернативно, можно попробовать извлечь из info_state, если игра его туда кладет
        # (зависит от реализации конкретной игры в OpenSpiel)
        return 0.0

    def _get_action_type(self, action):
        """Определяет тип действия (fold, call, raise) по его ID."""
        state = self._get_state()
        current_player = self._current_time_step.observations["current_player"] if self._current_time_step else -1
        if state and hasattr(state, 'action_to_string') and current_player >= 0:
             try:
                 # Используем официальный метод для преобразования действия в строку
                 action_str = state.action_to_string(current_player, action).lower()
                 if "fold" in action_str: return 'fold'
                 # Check может быть представлен как Call(0)
                 if "check" in action_str or "call" in action_str: return 'call'
                 # Bet и Raise оба агрессивные
                 if "raise" in action_str or "bet" in action_str: return 'raise'
                 logger.warning(f"Could not determine action type for action {action} (string: '{action_str}'). Defaulting based on ID.")
             except Exception as e:
                 logger.warning(f"Error calling action_to_string for action {action}: {e}. Defaulting based on ID.")

        # Запасной вариант на основе стандартных ID действий (0: fold, 1: call/check, >=2: bet/raise)
        # Это НЕ НАДЕЖНО для всех игр!
        if action == 0: return 'fold'
        if action == 1: return 'call' # Может быть и check
        if action >= 2: return 'raise' # Может быть и bet
        return 'unknown' # Если ID неожиданный

    def _get_street(self, time_step):
        """Получает текущую улицу (раунд) игры."""
        state = self._get_state()
        # Используем метод round(), если он есть
        if state and hasattr(state, 'round'):
            try:
                return int(state.round())
            except Exception as e:
                logger.warning(f"Could not get current round/street: {e}")
                return 0 # Возвращаем 0 (префлоп) по умолчанию
        # Если нет метода round(), возможно, придется парсить info_state
        return 0

    def _calculate_pot_odds(self, time_step, player_id):
         """Рассчитывает шансы банка для игрока."""
         state = self._get_state()
         if state and hasattr(state, 'amount_to_call') and hasattr(state, 'pot') and player_id >=0:
             try:
                 call_amount = float(state.amount_to_call(player_id))
                 if call_amount < 0: call_amount = 0 # Не может быть отрицательным
                 pot_size = float(state.pot())
                 # Шансы банка = цена колла / (банк + цена колла)
                 total_pot_after_call = pot_size + call_amount
                 if total_pot_after_call > 1e-6: # Избегаем деления на ноль
                     # Нормализуем к [0, 1] (хотя обычно они и так в этом диапазоне)
                     pot_odds = np.clip(call_amount / total_pot_after_call, 0.0, 1.0)
                     return pot_odds
                 else:
                      # Если колл 0 и банк 0, шансы не определены или бесконечны. Вернем 0.
                      return 0.0
             except Exception as e:
                 logger.warning(f"Could not calculate pot odds for player {player_id}: {e}")
                 return 0.0 # Возвращаем 0 в случае ошибки
         return 0.0

    def _calculate_stack_to_pot(self, time_step, player_id):
         """Рассчитывает отношение эффективного стека к поту (SPR)."""
         state = self._get_state()
         if state and hasattr(state, 'stacks') and hasattr(state, 'pot') and player_id >= 0:
             try:
                 stacks = state.stacks()
                 pot_size = float(state.pot())
                 if player_id < len(stacks):
                      # Используем текущий стек игрока как "эффективный" (упрощение)
                      player_stack = float(stacks[player_id])
                      if pot_size > 1e-6:
                          # Ограничиваем SPR сверху (например, 100) для стабильности
                          spr = np.clip(player_stack / pot_size, 0.0, 100.0)
                          # Нормализуем к [0, 1] делением на максимальное значение
                          return spr / 100.0
                      else:
                           # Если пот 0, SPR очень большой. Вернем 1.0 (максимальное нормализованное значение).
                           return 1.0
                 else:
                      logger.warning(f"Player ID {player_id} out of range for stacks ({len(stacks)}).")
                      return 0.0
             except Exception as e:
                  logger.warning(f"Could not calculate SPR for player {player_id}: {e}")
                  return 0.0 # Возвращаем 0 в случае ошибки
         return 0.0 # Возвращаем 0, если состояние недоступно

    def _enhance_info(self, info, current_player_id):
        """Добавляет полезную отладочную информацию в словарь info."""
        # Добавляем легальные действия для текущего игрока (если он не терминальный)
        if hasattr(self,'_current_time_step') and self._current_time_step and not self._current_time_step.last() and current_player_id >= 0:
             try:
                 # Убедимся, что current_player_id действителен для legal_actions
                 if current_player_id < len(self._current_time_step.observations["legal_actions"]):
                     info['legal_actions'] = self._current_time_step.observations["legal_actions"][current_player_id]
                 else:
                      info['legal_actions'] = []
             except Exception as e:
                 logger.warning(f"Could not get legal actions for player {current_player_id} in enhance_info: {e}")
                 info['legal_actions'] = []
        else:
             info['legal_actions'] = [] # Нет легальных действий в терминальном состоянии

        # Добавляем текущую статистику оппонентов (в виде словаря)
        # info['opponent_stats_debug'] = {p: self.stats.stats.get(p, {}) for p in range(self.game.num_players())}
        # Добавляем текущие рассчитанные фичи статистики (как они идут в наблюдение)
        info['opponent_stats_features'] = {p: self.stats.get_features(p).tolist() for p in range(self.game.num_players())}

        # Можно добавить другую полезную информацию: размер банка, улица и т.д.
        info['pot_size'] = self.current_hand_info.get('pot_size', 0)
        info['street'] = self.current_street

        return info

    def render(self, mode='human'):
        """Выводит текущее состояние игры в консоль."""
        state = self._get_state()
        rendered = str(state) if state else "No state available for rendering."
        if mode == 'human':
            print(rendered)
        return rendered # Возвращает строковое представление для 'ansi'

    def close(self):
        """Освобождает ресурсы (если необходимо)."""
        logger.info("Closing PokerEnv")
        # В данном случае OpenSpiel среда и игра управляются Python GC,
        # поэтому явного закрытия ресурсов может не требоваться.
        self._base_env = None
        self.game = None
