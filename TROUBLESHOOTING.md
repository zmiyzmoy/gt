# Руководство по устранению проблем с PokerRL

В данном руководстве описаны основные проблемы, с которыми вы можете столкнуться при работе с PokerRL, и способы их решения.

## Проблемы с OpenSpiel

### Ошибка: "Unknown parameter 'stackSize'"

**Проблема:**
OpenSpiel ожидает параметр `stack` вместо `stackSize`.

**Решение:**
1. Используйте исправленные версии файлов: `environment_improved.py` вместо `environment_fixed.py`
2. Убедитесь, что в конфигурации используется параметр `stack` вместо `stackSize`

```python
# Неправильно
game_config["stackSize"] = 10000

# Правильно
game_config["stack"] = "10000 10000 10000 ..."  # Для каждого игрока
# или
game_config["stack"] = [10000, 10000, ...]  # С преобразованием в строку
```

### Ошибка: "cannot convert dictionary update sequence element #0 to a sequence"

**Проблема:**
Неправильный формат параметров при преобразовании из словаря Python в параметры OpenSpiel.

**Решение:**
Убедитесь, что все параметры правильно форматированы:

```python
# В environment_improved.py
def _process_game_params(self, game_params):
    processed = {}
    for key, value in game_params.items():
        if isinstance(value, (list, tuple)):
            processed[key] = " ".join(map(str, value))
        else:
            processed[key] = value
    return processed
```

### Ошибка сегментации (Segmentation Fault)

**Проблема:**
Неправильно настроенные параметры покерной игры могут приводить к ошибкам сегментации.

**Решение:**
1. Проверьте параметр `numBoardCards` - он должен содержать 4 числа (по одному для каждой улицы)
2. Проверьте параметр `blind` - он должен содержать значения для каждого игрока
3. Используйте `test_spiel_config.py` для проверки параметров:

```bash
python test_spiel_config.py
```

## Проблемы с Ray

### Ошибка: "module 'ray' has no attribute 'get_webui_url'"

**Проблема:**
Вы используете новую версию Ray (2.10.0+), в которой метод `get_webui_url()` устарел.

**Решение:**
Используйте исправленную версию `train_fixed.py` с удаленным или замененным методом:

```python
# Убрать эту строку
logger.info(f"Ray initialized successfully. Dashboard URL: {ray.get_webui_url()}")

# Заменить на
logger.info("Ray initialized successfully.")
```

### Ошибка инициализации Ray при многократных запусках

**Проблема:**
При повторных запусках Ray может возникать ошибка, если предыдущая сессия не была корректно закрыта.

**Решение:**
1. Добавьте гарантированный shutdown Ray в блоке finally:

```python
try:
    # Код с использованием Ray
finally:
    ray.shutdown()
```

2. Используйте опцию `ignore_reinit_error=True` при инициализации:

```python
ray.init(ignore_reinit_error=True)
```

## Проблемы с моделью

### Ошибка: "int() argument must be a string, a bytes-like object or a real number, not 'NoneType'"

**Проблема:**
Модель не может определить размерность входного наблюдения.

**Решение:**
Используйте улучшенную версию модели с проверкой типов:

```python
# В models_improved.py
if hasattr(obs_space, "spaces") and "obs" in obs_space.spaces:
    self.obs_size = int(np.prod(obs_space.spaces["obs"].shape))
elif hasattr(obs_space, "shape"):
    self.obs_size = int(np.prod(obs_space.shape))
else:
    self.obs_size = model_config.get("custom_model_config", {}).get("obs_size", 1000)
```

### Ошибка с размерностью тензоров

**Проблема:**
Несоответствие размерностей тензоров при обработке наблюдений или действий.

**Решение:**
1. Добавьте дополнительное логирование для отслеживания размерностей:

```python
logger.info(f"Observation shape: {observation.shape}")
logger.info(f"Action mask shape: {action_mask.shape}")
```

2. Используйте обработку ошибок в методе forward:

```python
try:
    # Код обработки наблюдения
except Exception as e:
    logger.error(f"Error in forward pass: {e}")
    # Возвращаем безопасные значения
```

## Проблемы с CUDA/GPU

### Ошибка: "CUDA out of memory"

**Проблема:**
Недостаточно памяти GPU для обучения.

**Решение:**
1. Уменьшите размер batch:

```python
train_batch_size: int = 4096  # Вместо 8192
sgd_minibatch_size: int = 512  # Вместо 1024
```

2. Уменьшите количество workers или используйте CPU:

```python
num_gpus: int = 0  # Отключить GPU
# или
num_gpus_per_worker: float = 0.0  # Не выделять GPU для worker'ов
```

### Ошибка инициализации CUDA

**Проблема:**
Ошибки при инициализации CUDA.

**Решение:**
1. Проверьте наличие CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

2. Принудительно используйте CPU:

```python
# В начале скрипта
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

## Общие проблемы и решения

### Ошибка: "Memory Error" или "Process killed"

**Проблема:**
Недостаточно ОЗУ на VPS.

**Решение:**
1. Уменьшите количество параллельных workers:

```python
num_workers: int = 2  # Вместо 7
```

2. Уменьшите размеры батчей и буферов траекторий:

```python
rollout_fragment_length: str = "auto"
train_batch_size: int = 2048
```

### Оптимизация для CPU-only обучения

Если у вас нет доступа к GPU, оптимизируйте конфигурацию:

```python
# В config.py
num_gpus: int = 0
num_workers: int = max(multiprocessing.cpu_count() - 1, 1)
train_batch_size: int = 2048
sgd_minibatch_size: int = 256
```

## Дополнительные инструменты диагностики

### Проверка OpenSpiel

```bash
python -c "import pyspiel; print(pyspiel.registered_games())"
```

### Проверка использования памяти

```bash
python -c "import psutil; print(f'Available memory: {psutil.virtual_memory().available / (1024**3):.2f} GB')"
```

### Мониторинг использования ресурсов во время обучения

```bash
# Установите htop если ещё не установлен
apt-get install htop

# Запустите мониторинг в отдельном терминале
htop
```

Если у вас возникнут другие проблемы, не описанные в этом руководстве, обратитесь к документации Ray и OpenSpiel, или создайте issue на GitHub.