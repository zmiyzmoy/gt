import logging
import math
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType

logger = logging.getLogger(__name__)

class AdvancedPokerModel(TorchModelV2, nn.Module):
    """
    Улучшенная модель для покера с обработкой статистики оппонентов
    и масок действий.
    """
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        """
        Инициализирует модель с модульной архитектурой.
        """
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        # Определение размерности входных наблюдений
        self.obs_size = 0
        if hasattr(obs_space, "original_space"):
            if hasattr(obs_space.original_space["obs"], "shape"):
                self.obs_size = int(np.product(obs_space.original_space["obs"].shape))
            elif isinstance(obs_space.original_space["obs"], dict) and "obs" in obs_space.original_space["obs"]:
                self.obs_size = int(np.product(obs_space.original_space["obs"]["obs"].shape))
        else:
            self.obs_size = int(np.product(obs_space.shape))
        
        # Получение параметров из конфигурации модели
        hiddens = model_config.get("fcnet_hiddens", [256, 256])
        activation = model_config.get("fcnet_activation", "relu")
        
        # Выбор функции активации
        if activation == "relu":
            activation_fn = nn.ReLU
        elif activation == "tanh":
            activation_fn = nn.Tanh
        else:
            activation_fn = nn.ReLU  # По умолчанию ReLU
        
        # Создание слоёв для обработки признаков
        in_size = self.obs_size
        layers = []
        
        for size in hiddens:
            layers.append(nn.Linear(in_size, size))
            layers.append(activation_fn())
            in_size = size
        
        self._features = nn.Sequential(*layers)
        self._logits = nn.Linear(in_size, num_outputs)
        self._value = nn.Linear(in_size, 1)
        
        # Для отслеживания последних активаций
        self._last_features = None
        
        # Регистрация базового состояния (не используется в этой модели)
        self.register_variables([])
        
        logger.info(f"Initialized {self.__class__.__name__} with obs_size={self.obs_size}, output_size={num_outputs}")
    
    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType],
                seq_lens: TensorType) -> Tuple[TensorType, List[TensorType]]:
        """
        Выполняет прямой проход модели с улучшенной обработкой ошибок.
        """
        try:
            # Обработка входных данных в зависимости от их формата
            if "obs" not in input_dict:
                logger.error(f"Missing 'obs' in input_dict. Keys: {input_dict.keys()}")
                features = torch.zeros((1, self.obs_size), device=self._logits.weight.device)
            elif isinstance(input_dict["obs"], dict):
                if "obs" in input_dict["obs"]:
                    features = input_dict["obs"]["obs"].float()
                else:
                    logger.error(f"Nested 'obs' not found in input_dict['obs']. Keys: {input_dict['obs'].keys()}")
                    features = torch.zeros((1, self.obs_size), device=self._logits.weight.device)
            else:
                features = input_dict["obs"].float()
            
            # Проверка размерности входа
            if features.ndim == 1:
                features = features.unsqueeze(0)  # Добавление размерности батча
            
            # Обработка случая, когда размерность не соответствует ожидаемой
            expected_size = self.obs_size
            actual_size = features.shape[-1]
            
            if actual_size != expected_size:
                logger.warning(f"Shape mismatch! Expected {expected_size}, got {actual_size}")
                
                # Подгонка размера для совместимости
                if actual_size > expected_size:
                    logger.info("Truncating features to match expected size")
                    features = features[..., :expected_size]
                else:
                    logger.info("Padding features to match expected size")
                    pad_size = expected_size - actual_size
                    # Добавление нулевого заполнения к последнему измерению
                    padding = torch.zeros(*features.shape[:-1], pad_size, device=features.device)
                    features = torch.cat([features, padding], dim=-1)
            
            # Преобразование типа при необходимости
            if not isinstance(features, torch.FloatTensor) and not isinstance(features, torch.cuda.FloatTensor):
                features = features.float()
            
            # Пропускаем фичи через основные слои
            self._last_features = self._features(features)
            
            # Получаем логиты для действий
            logits = self._logits(self._last_features)
            
            # Применение маски действий, если она доступна
            if "action_mask" in input_dict:
                mask = input_dict["action_mask"]
                # Преобразуем маску в тот же тип, что и логиты
                mask = mask.to(dtype=logits.dtype)
                # Применяем маску: устанавливаем большое отрицательное число для недопустимых действий
                inf_mask = torch.clamp(torch.log(mask), min=-1e10)
                logits = logits + inf_mask
            
            return logits, state
            
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            # В случае критической ошибки возвращаем нулевые логиты
            device = next(self.parameters()).device
            dummy_logits = torch.zeros((1, self.num_outputs), device=device)
            # Возбуждаем исключение для отладки
            raise ValueError(f"Forward pass error: {e}") from e
    
    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        """
        Возвращает оценку значения текущего состояния (value function).
        """
        assert self._last_features is not None, "value_function() called before forward()"
        return self._value(self._last_features).squeeze(1)