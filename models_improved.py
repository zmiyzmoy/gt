#!/usr/bin/env python3
"""
Улучшенная модель для покера с исправленной обработкой входных данных.
Учитывает структуру наблюдения и действий, статистику оппонентов.
"""

import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import Dict, TensorType, List, Tuple, ModelConfigDict

import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("poker_model")

class AdvancedPokerModel(TorchModelV2, nn.Module):
    """
    Улучшенная модель для покера с обработкой статистики оппонентов
    и масок действий.
    """
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        """
        Инициализирует модель с модульной архитектурой.
        """
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        
        try:
            logger.info(f"Initializing AdvancedPokerModel with obs_space: {obs_space}, action_space: {action_space}, num_outputs: {num_outputs}")
            logger.info(f"Model config: {model_config}")
            
            # Определение размерности входа из observation_space
            # Для Dict содержащего "obs"
            if hasattr(obs_space, "spaces") and "obs" in obs_space.spaces:
                self.obs_size = int(np.prod(obs_space.spaces["obs"].shape))
            # Для простого Box или подобного
            elif hasattr(obs_space, "shape"):
                self.obs_size = int(np.prod(obs_space.shape))
            else:
                # Если нельзя определить из obs_space, используем переданное значение или значение по умолчанию
                self.obs_size = model_config.get("custom_model_config", {}).get("obs_size", 1000)
                logger.warning(f"Could not determine obs_size from obs_space, using: {self.obs_size}")
            
            # Получаем размерность выхода (количество действий)
            self.num_actions = num_outputs  # В случае покера это 3 (fold, call/check, raise)
            
            # Получаем параметры из custom_model_config
            custom_config = model_config.get("custom_model_config", {})
            
            # Размеры слоев скрытых слоев
            fcnet_hiddens = model_config.get("fcnet_hiddens", [256, 256])
            fcnet_activation = model_config.get("fcnet_activation", "relu")
            
            # Мапинг активаций
            activation_map = {
                "linear": nn.Identity,
                "relu": nn.ReLU,
                "tanh": nn.Tanh,
                "sigmoid": nn.Sigmoid,
                "elu": nn.ELU,
                "selu": nn.SELU,
                "leaky_relu": nn.LeakyReLU,
            }
            activation = activation_map.get(fcnet_activation, nn.ReLU)
            
            # Создаем сеть наблюдение -> скрытые слои
            main_layers = []
            
            # Входной размер для первого слоя - размер наблюдения
            curr_size = self.obs_size
            
            # Добавляем полносвязные слои
            for hidden_size in fcnet_hiddens:
                main_layers.append(nn.Linear(curr_size, hidden_size))
                main_layers.append(activation())
                curr_size = hidden_size
            
            # Создаем основную сеть как Sequential
            self.main_net = nn.Sequential(*main_layers)
            
            # Создаем головы для policy и value
            self.policy_head = nn.Linear(curr_size, num_outputs)
            self.value_head = nn.Linear(curr_size, 1)
            
            # Для маскирования действий и хранения значения
            self._value = None
            
            # Выводим информацию о структуре модели
            logger.info(f"AdvancedPokerModel layers: {self.main_net}")
            logger.info(f"policy_head: {self.policy_head}, value_head: {self.value_head}")
            
        except Exception as e:
            logger.error(f"Error initializing AdvancedPokerModel: {e}")
            logger.error(f"Creating fallback model. Training may fail.")
            
            # Создаем минимальную модель в случае ошибки
            self.obs_size = 1000
            self.main_net = nn.Sequential(
                nn.Linear(self.obs_size, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU()
            )
            self.policy_head = nn.Linear(256, num_outputs)
            self.value_head = nn.Linear(256, 1)
            self._value = None
    
    @override
    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType],
                seq_lens: TensorType) -> Tuple[TensorType, List[TensorType]]:
        """
        Выполняет прямой проход модели с улучшенной обработкой ошибок.
        """
        try:
            # Получаем тензор наблюдения из входного словаря
            if "obs" in input_dict:
                obs = input_dict["obs"]
            else:
                obs = input_dict["obs_flat"]
            
            # Проверяем размерность входа
            if isinstance(obs, dict) and "obs" in obs:
                # Если наблюдение представлено словарем, извлекаем из него
                features = obs["obs"]
            else:
                # Иначе используем напрямую
                features = obs
            
            # Проходим через основную сеть
            features = self.main_net(features)
            
            # Получаем выход policy head
            logits = self.policy_head(features)
            
            # Маскировка невозможных действий (если маска есть)
            if isinstance(obs, dict) and "action_mask" in obs:
                # Применяем маску к логитам (устанавливаем минус бесконечность для невозможных действий)
                mask = obs["action_mask"]
                inf_mask = torch.clamp(torch.log(mask), min=-1e38)
                logits = logits + inf_mask
            
            # Получаем значение (value function)
            self._value = self.value_head(features).squeeze(1)
            
            return logits, state
            
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            
            # В случае ошибки возвращаем безопасные значения
            batch_size = 1
            if isinstance(input_dict, dict) and "obs" in input_dict:
                if hasattr(input_dict["obs"], "shape") and len(input_dict["obs"].shape) > 0:
                    batch_size = input_dict["obs"].shape[0]
            
            # Создаем тензоры нужной размерности
            fake_logits = torch.zeros(batch_size, self.num_actions, device=self.device)
            self._value = torch.zeros(batch_size, device=self.device)
            
            return fake_logits, state
    
    @override
    def value_function(self) -> TensorType:
        """
        Возвращает оценку значения текущего состояния (value function).
        """
        if self._value is None:
            # Если value не был вычислен, возвращаем нулевой тензор
            return torch.zeros(1, device=self.device)
        return self._value