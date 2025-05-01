# Решение проблем с покерным ИИ

Руководство по устранению типичных проблем, возникающих при обучении покерного ИИ.

## Проблемы с размерностью тензоров

### Симптом: `RuntimeError: forward() got tensor with wrong shape`

Эта ошибка возникает, когда модель получает вход неожиданного размера. Наиболее частые причины:

1. **Несоответствие между наблюдением среды и ожиданиями модели**

   **Решение:**
   ```python
   # В models_fixed.py в методе forward
   # Добавьте проверку и коррекцию размерности:
   if features.shape[-1] != self.obs_size:
       logger.warning(f"Shape mismatch! Expected {self.obs_size}, got {features.shape[-1]}")
       # Подгонка размера для совместимости
       if features.shape[-1] > self.obs_size:
           features = features[..., :self.obs_size]
       else:
           pad_size = self.obs_size - features.shape[-1]
           padding = torch.zeros(*features.shape[:-1], pad_size, device=features.device)
           features = torch.cat([features, padding], dim=-1)
   ```

2. **Несогласованность размера статистики оппонентов**

   **Решение:**
   ```python
   # В environment_fixed.py, метод get_features класса OpponentStats
   # Убедитесь, что размерность всегда фиксирована:
   def get_features(self, player_id):
       # ... существующий код ...
       # Убедитесь что размерность всегда фиксирована
       expected_size = 14  # vpip, pfr, 3 метрики * 4 улицы, кол-во рук
       if len(features) != expected_size:
           # Логирование и коррекция
           if len(features) < expected_size:
               features = np.pad(features, (0, expected_size - len(features)))
           else:
               features = features[:expected_size]
       return features
   ```

### Симптом: `ValueError: setting an array element with a sequence.`

Обычно возникает при преобразовании списков различной длины в numpy массивы.

**Решение:**
```python
# В environment_fixed.py, метод _get_observation
# Убедитесь, что все arrays оппонентов одинаковой длины:
opponents_stats = []
expected_opponent_size = self._opp_stats_size  # Фиксированный размер

for opp_id in range(self.num_players):
    if opp_id != player_id:
        opp_stats = self.stats.get_features(opp_id)
        if len(opp_stats) != expected_opponent_size:
            # Исправление размера
            if len(opp_stats) < expected_opponent_size:
                opp_stats = np.pad(opp_stats, (0, expected_opponent_size - len(opp_stats)))
            else:
                opp_stats = opp_stats[:expected_opponent_size]
        opponents_stats.append(opp_stats)
```

## Проблемы с памятью

### Симптом: `ray.exceptions.RayOutOfMemoryError`

Эта ошибка указывает, что Ray израсходовал всю доступную память.

**Решение:**

1. **Уменьшите количество воркеров и размеры батчей:**
   ```python
   # В config.py, класс TrainingConfig
   num_workers = 2               # Уменьшите с 7
   train_batch_size = 2048       # Уменьшите с 8192
   sgd_minibatch_size = 512      # Уменьшите с 1024
   num_envs_per_worker = 1       # Уменьшите с 2+
   ```

2. **Добавьте очистку памяти:**
   ```python
   # В train_fixed.py
   # После каждой итерации обучения
   import gc
   gc.collect()
   torch.cuda.empty_cache()  # Если используется GPU
   ```

3. **Мониторинг использования памяти:**
   ```python
   # Добавьте в train_fixed.py для мониторинга
   def log_memory_usage():
       process = psutil.Process(os.getpid())
       mem_info = process.memory_info()
       logger.info(f"Memory usage: RSS={mem_info.rss/1024/1024:.2f}MB, VMS={mem_info.vms/1024/1024:.2f}MB")
   ```

### Симптом: Процесс обучения замедляется со временем

Это может указывать на утечку памяти в коде.

**Решение:**

1. **Проверьте накопление данных в OpponentStats:**
   ```python
   # В environment_fixed.py, класс OpponentStats
   # Периодическая очистка статистики для очень длинных сессий
   def clean_old_stats(self, max_hands=10000):
       """Очищает слишком старые данные чтобы избежать утечек памяти"""
       for player_id in self.stats:
           if self.stats[player_id]["hands_played"] > max_hands:
               # Сохраняем только последние агрегированные метрики
               old_stats = self.stats[player_id].copy()
               self._ensure_player_stats(player_id)  # Новая чистая статистика
               
               # Копируем агрегированные метрики
               self.stats[player_id]["vpip_actions"] = old_stats["vpip_actions"] 
               self.stats[player_id]["vpip_opportunities"] = old_stats["vpip_opportunities"]
               # ... и другие важные метрики
   ```

2. **Регулярная очистка памяти в Ray:**
   ```python
   # В train_fixed.py
   # Регулярный перезапуск Ray для очистки памяти каждые N итераций
   if (iteration + 1) % 10 == 0:
       logger.info("Restarting Ray to clear memory...")
       ray.shutdown()
       setup_ray(train_cfg.log_level)
   ```

## Проблемы с OpenSpiel

### Симптом: `AssertionError: state->CurrentPlayer() >= 0` или некорректные состояния

Эта ошибка возникает, когда OpenSpiel пытается получить действие для неактивного игрока.

**Решение:**
```python
# В environment_fixed.py, метод step
# Добавьте проверку валидности игрока:
def step(self, action: int):
    try:
        if self._episode_ended:
            logger.warning("Episode already ended, reset needed")
            # ...

        # Проверка корректности текущего игрока
        player_id = self._current_player()
        if player_id < 0 or player_id >= self.num_players:
            logger.error(f"Invalid player ID: {player_id}, ending episode")
            self._episode_ended = True
            return self._get_observation(), 0.0, True, False, self._get_info()

        # ... остальной код метода step
```

### Симптом: Некорректная обработка ставок или игровой логики

Иногда игра не соответствует ожидаемой логике покера.

**Решение:**

1. **Проверьте конфигурацию покерной игры:**
   ```python
   # В config.py, проверьте параметры:
   game_config = {
       "numPlayers": 2,           # Убедитесь, что это верное число игроков
       "betting": "nolimit",      # "nolimit" или "limit"
       "numRounds": 4,            # 4 раунда (preflop, flop, turn, river)
       "blind": "50 100",         # Формат: "SB BB"
   }
   ```

2. **Проверка валидности переходов между улицами:**
   ```python
   # В environment_fixed.py, метод step
   # Добавьте отладочный вывод при смене улицы:
   street_before = self._get_street(self._current_time_step)
   # ... выполнение действия ...
   street_after = self._get_street(self._current_time_step)
   
   if street_after != street_before:
       logger.debug(f"Street changed: {street_before} -> {street_after}")
       # Проверка корректности перехода
       if street_after != street_before + 1 and not self.is_terminal():
           logger.warning(f"Unexpected street transition: {street_before} -> {street_after}")
   ```

## Проблемы с обучением

### Симптом: Награда не улучшается или колеблется

Отсутствие улучшения стратегии.

**Решение:**

1. **Настройте параметры обучения:**
   ```python
   # В config.py, TrainingConfig
   lr = 3e-5                  # Уменьшите с 5e-5 для более стабильного обучения
   entropy_coeff = 0.005      # Уменьшите с 0.01 если агент действует слишком случайно
   ```

2. **Добавьте убывающий learning rate:**
   ```python
   # В train_fixed.py, train_poker
   # Настройка убывающего learning rate
   lr_schedule = [
       (0, train_cfg.lr),
       (train_cfg.num_iterations // 2, train_cfg.lr / 2),
       (train_cfg.num_iterations, train_cfg.lr / 10),
   ]
   
   config = (
       PPOConfig()
       # ...
       .training(
           lr_schedule=lr_schedule,
           # ... остальные параметры
       )
   )
   ```

3. **Увеличьте количество итераций:**
   ```python
   # В config.py, TrainingConfig
   num_iterations = 3000         # Увеличьте с 1000
   ```

### Симптом: Нестабильный тренировочный процесс (ошибки на случайных итерациях)

**Решение:**

1. **Увеличьте стабильность с помощью клиппинга градиентов:**
   ```python
   # В train_fixed.py, train_poker, добавьте в конфигурацию:
   config = (
       PPOConfig()
       # ...
       .training(
           # ... существующие параметры
           grad_clip=0.5,        # Клиппинг градиентов
       )
   )
   ```

2. **Используйте `try-except` в критичных местах:**
   ```python
   # В models_fixed.py, forward
   def forward(self, input_dict, state, seq_lens):
       try:
           # ... существующий код ...
       except Exception as e:
           logger.error(f"Error in forward pass: {e}")
           # Возвращаем запасной вывод
           dummy_tensor = torch.zeros((input_dict["obs"].shape[0], self.num_outputs), 
                                     device=self._device)
           return dummy_tensor, state
   ```

## Проблемы с Ray и многопроцессностью

### Симптом: `ConnectionError: Connection refused by the master`

Проблема с соединением в кластере Ray.

**Решение:**
```python
# В train_fixed.py, setup_ray
def setup_ray(log_level="INFO"):
    try:
        # Всегда перезапускаем Ray для очистки состояния
        if ray.is_initialized():
            ray.shutdown()
        
        # Инициализируем с увеличенным таймаутом
        ray.init(
            logging_level=log_level,
            log_to_driver=True,
            ignore_reinit_error=True,
            _redis_max_memory=500 * 1024 * 1024,  # Увеличение памяти Redis (500MB)
            _system_config={
                "object_timeout_milliseconds": 5000,
                "worker_register_timeout_seconds": 60,
            },
        )
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Ray: {e}")
        return False
```

### Симптом: `TimeoutError: Timed out waiting for results`

Ray не может получить результаты от воркеров вовремя.

**Решение:**
```python
# В train_fixed.py, train_poker
# Увеличьте таймауты и добавьте восстановление:
config = (
    PPOConfig()
    # ...
    .resources(
        # ... существующие параметры
        _system_config={
            "timeout_ms_task_wait_for_death": 30000,  # 30 секунд
            "raylet_reconnect_timeout_ms": 30000,
            "task_retry_max_retries": 3,  # Повторы для неудачных тасков
        },
    )
)
```

## Мониторинг и отладка

### Полезные инструменты:

1. **Расширенный мониторинг:**
   ```python
   # В train_fixed.py
   def monitor_episode_metrics(result):
       """Подробный мониторинг метрик эпизода"""
       metrics = result.get("episode_reward_mean", 0)
       episode_len = result.get("episode_len_mean", 0)
       logger.info(f"Episode metrics: reward={metrics:.4f}, length={episode_len:.1f}")
       
       # Детальная статистика действий (если доступна)
       action_data = result.get("custom_metrics", {}).get("action_distribution", {})
       if action_data:
           fold_pct = action_data.get("fold_pct", 0)
           call_pct = action_data.get("call_pct", 0)
           raise_pct = action_data.get("raise_pct", 0)
           logger.info(f"Actions: fold={fold_pct:.2f}%, call={call_pct:.2f}%, raise={raise_pct:.2f}%")
   ```

2. **Сохранение и анализ информации о проблемных состояниях:**
   ```python
   # В environment_fixed.py, step
   def step(self, action: int):
       try:
           # ... существующий код ...
       except Exception as e:
           logger.error(f"Error in step: {e}")
           
           # Сохранение проблемного состояния для отладки
           error_dir = "error_states"
           os.makedirs(error_dir, exist_ok=True)
           timestamp = int(time.time())
           error_file = os.path.join(error_dir, f"error_state_{timestamp}.json")
           
           # Сохраняем информацию о состоянии
           error_info = {
               "action": action,
               "player_id": self._current_player(),
               "error": str(e),
               "street": self._get_street(self._current_time_step),
               "trace": traceback.format_exc(),
           }
           
           with open(error_file, "w") as f:
               json.dump(error_info, f, indent=2)
           
           logger.info(f"Saved error state to {error_file}")
           
           # Возвращаем запасное состояние
           observation = self._get_observation()
           return observation, 0.0, True, False, self._get_info()
   ```

3. **Мониторинг использования ресурсов:**
   ```bash
   # В отдельном терминале постоянно запущены:
   watch -n 1 "ps -o pid,ppid,cmd,%cpu,%mem --sort=-%mem | head -n 15"
   
   # Для GPU:
   watch -n 1 nvidia-smi
   ```

## Проблемы совместимости версий

### Симптом: `ImportError` или `AttributeError` в коде RLlib или OpenSpiel

**Решение:**

Убедитесь, что используете совместимые версии:

```bash
# Рекомендуемые версии:
pip install torch==2.0.1
pip install ray[rllib]==2.10.0
pip install gymnasium==0.29.1
pip install open_spiel==1.3
pip install numpy==1.24.1
```

Если проблемы с совместимостью сохраняются, создайте чистое виртуальное окружение и точно соблюдайте порядок установки:

```bash
python -m venv fresh_env
source fresh_env/bin/activate

pip install torch==2.0.1  # Сначала PyTorch
pip install numpy==1.24.1  # Затем NumPy определенной версии
pip install gymnasium==0.29.1  # Затем Gymnasium
pip install open_spiel==1.3  # Затем OpenSpiel
pip install ray[rllib]==2.10.0  # В конце Ray RLlib
```