# Руководство по установке покерного ИИ

Подробная инструкция по настройке окружения для обучения покерного ИИ с использованием RL.

## Базовая установка (Ubuntu/Debian)

```bash
# Обновление пакетов
sudo apt update
sudo apt upgrade -y

# Установка необходимых системных пакетов
sudo apt install -y python3-dev python3-pip git build-essential cmake

# Создание директории для проекта
mkdir -p ~/poker_ai
cd ~/poker_ai

# Клонирование репозитория (если используете Git)
git clone https://github.com/ваш_логин/имя_репозитория.git .
# ИЛИ скопируйте файлы вручную

# Создание виртуального окружения Python
python3 -m venv poker_venv
source poker_venv/bin/activate

# Установка базовых зависимостей
pip install -U pip setuptools wheel
pip install numpy torch

# Установка OpenSpiel
pip install open_spiel==1.3

# Установка остальных зависимостей
pip install ray[rllib]==2.10.0 gymnasium==0.29.1 psutil

# Проверка установки
python test_env.py --episodes 1
```

## Установка с GPU (NVIDIA)

```bash
# Убедитесь что у вас установлены драйверы NVIDIA
nvidia-smi

# Если драйверы отсутствуют, установите их
sudo apt install -y nvidia-driver-XXX  # выберите актуальную версию

# Установка CUDA (требуется для PyTorch с GPU)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update
sudo apt install -y cuda

# Установка PyTorch с поддержкой CUDA
pip install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Установка остальных пакетов
pip install ray[rllib]==2.10.0 gymnasium==0.29.1 open_spiel==1.3 psutil
```

## Установка на Windows

```powershell
# Установка Python 3.10+ с официального сайта python.org

# Создание виртуального окружения
python -m venv poker_venv
poker_venv\Scripts\activate

# Установка базовых зависимостей
pip install -U pip setuptools wheel
pip install numpy

# Установка PyTorch (CPU версия)
pip install torch 

# Для GPU версии:
# pip install torch --index-url https://download.pytorch.org/whl/cu118

# Установка OpenSpiel (может потребоваться установить CMake и VS Build Tools)
pip install open_spiel

# Установка остальных пакетов
pip install ray[rllib]==2.10.0 gymnasium==0.29.1 psutil

# Проверка установки
python test_env.py --episodes 1
```

## Конфигурация для разных мощностей оборудования

### Слабое оборудование (CPU-only, < 8GB RAM)

В файле `config.py` измените следующие параметры:

```python
# TrainingConfig
num_workers = 2                 # Уменьшаем число воркеров
num_envs_per_worker = 1         # По одной среде на воркера
train_batch_size = 1024         # Уменьшаем размер батча
sgd_minibatch_size = 256        # Уменьшаем размер мини-батча
num_sgd_iter = 5                # Меньше итераций SGD
```

### Среднее оборудование (CPU-only, 16GB RAM)

```python
# TrainingConfig
num_workers = 4                 # Увеличиваем число воркеров
num_envs_per_worker = 2         # Две среды на воркера
train_batch_size = 2048         # Средний размер батча
sgd_minibatch_size = 512        # Средний размер мини-батча
```

### Мощное оборудование (GPU, 32GB+ RAM)

```python
# TrainingConfig
num_workers = 7                 # Много воркеров 
num_gpus = 1                    # Использовать GPU
num_envs_per_worker = 4         # Много сред на воркера
train_batch_size = 8192         # Большой размер батча
sgd_minibatch_size = 1024       # Большой размер мини-батча
num_sgd_iter = 10               # Больше итераций SGD
```

## Мониторинг и Отладка

```bash
# Проверка использования памяти в процессе обучения
python -m train_fixed.py --debug

# Мониторинг использования GPU (для GPU тренировки)
watch -n 1 nvidia-smi

# Мониторинг использования CPU и памяти
htop

# Подключение к Ray Dashboard (сервер запускается автоматически)
# Откройте в браузере http://localhost:8265
```

## Типичные проблемы и их решение

### Ошибка: CUDA out of memory

**Решение**: Уменьшите параметры батчей в конфигурации:
```python
# TrainingConfig
train_batch_size = 4096         # Уменьшить с 8192
sgd_minibatch_size = 512        # Уменьшить с 1024
```

### Ошибка: Ray crashes without error message

**Решение**: Проверьте доступную память и уменьшите количество воркеров:
```python
# TrainingConfig
num_workers = 2                 # Уменьшить
```

### Ошибка: Shape mismatch in model forward pass

**Решение**: Убедитесь, что модель корректно обрабатывает входные размерности:
1. Проверьте размер `self.obs_size` в `AdvancedPokerModel.__init__`
2. Убедитесь, что среда возвращает наблюдения правильного размера

### Ошибка: ValueError: sample larger than population

**Решение**: Проблема может быть в `sgd_minibatch_size` > `train_batch_size`:
```python
# TrainingConfig
sgd_minibatch_size = 256        # Должно быть меньше train_batch_size
train_batch_size = 1024         # Убедитесь что это значение больше
```

## Дополнительные ресурсы

1. [Документация Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html)
2. [Документация OpenSpiel](https://github.com/deepmind/open_spiel)
3. [Документация PyTorch](https://pytorch.org/docs/stable/index.html)
4. [Статья о Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)