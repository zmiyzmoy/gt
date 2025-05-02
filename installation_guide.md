# Руководство по установке PokerRL

Данное руководство содержит инструкции по установке и настройке проекта PokerRL для обучения покерного агента с использованием Ray RLlib и OpenSpiel.

## Системные требования

- Python 3.10+
- Минимум 8 ГБ RAM
- Желательно наличие GPU (для ускорения обучения)
- Ubuntu 20.04+ или другой Linux-дистрибутив

## Установка зависимостей

### 1. Создание виртуального окружения

```bash
python3 -m venv poker_env
source poker_env/bin/activate
```

### 2. Установка OpenSpiel

OpenSpiel требует несколько системных зависимостей:

```bash
sudo apt-get update
sudo apt-get install cmake clang build-essential python3-dev
```

Затем установите OpenSpiel через pip:

```bash
pip install open_spiel
```

### 3. Установка основных библиотек

```bash
pip install -r requirements.txt
```

Содержимое `requirements.txt`:

```
numpy>=1.24.0
torch>=1.13.0
ray[all]>=2.10.0
gymnasium>=0.28.1
wandb>=0.15.0
matplotlib>=3.7.0
psutil>=5.9.0
```

## Тестирование установки

После установки всех зависимостей рекомендуется выполнить проверку окружения:

```bash
python debug.py
```

Если нет ошибок, можно перейти к тестированию покерного окружения:

```bash
python test_spiel_config.py
python test_env.py --episodes 1
```

## Особенности настройки для VPS

### CUDA и GPU

Если вы используете GPU, убедитесь, что установлен CUDA toolkit:

```bash
# Проверка, доступен ли CUDA для PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

### Настройка ресурсов Ray

Для оптимальной производительности на VPS, рекомендуется настроить параметры Ray в `config.py`:

```python
# Для CPU-only
num_workers: int = (CPU_COUNT - 1) or 1
num_gpus: int = 0

# Для GPU
num_workers: int = (CPU_COUNT - 1) or 1
num_gpus: int = 1
```

## Решение проблем

### Ошибка "Unknown parameter 'stackSize'"

Если вы видите ошибку "Unknown parameter 'stackSize'", используйте исправленные версии файлов:

- `environment_improved.py` вместо `environment.py` или `environment_fixed.py`
- `models_improved.py` вместо `models.py` или `models_fixed.py`
- `register_models_improved.py` вместо `register_models.py`

### Ошибка при инициализации Ray 

Если вы видите ошибку "module 'ray' has no attribute 'get_webui_url'", используйте версию `train_fixed.py` с исправленной инициализацией Ray.

### Segmentation fault при создании покерной игры

Проверьте правильность форматирования параметра `numBoardCards`:

```python
# Правильный формат
"numBoardCards": "0 3 1 1"  # или [0, 3, 1, 1] с последующим преобразованием в строку
```

## Запуск обучения

После установки зависимостей и проверки окружения, запустите обучение:

```bash
python train_fixed.py --iterations 1000
```

Для мониторинга с Weights & Biases, сначала выполните логин:

```bash
wandb login
```

## Дополнительная информация

Для получения дополнительной информации обратитесь к документации:

- [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html)
- [OpenSpiel](https://github.com/deepmind/open_spiel)
- [PyTorch](https://pytorch.org/docs/stable/index.html)