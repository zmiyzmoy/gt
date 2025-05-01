# models.py
import numpy as np
import gymnasium as gym # Используем gymnasium для spaces
import logging

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
from ray.rllib.utils.framework import try_import_torch

logger = logging.getLogger(__name__)
torch, nn = try_import_torch()

class AdvancedPokerModel(TorchModelV2, nn.Module):
    """
    Кастомная модель, обрабатывающая Dict observation space с ключами 'obs' и 'action_mask'.
    Использует 'obs' для вычислений и полагается на RLlib для применения 'action_mask'.
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # --- ИСПРАВЛЕНО: Получаем размер из подпространства "obs" ---
        if isinstance(obs_space, gym.spaces.Dict):
            # Проверяем наличие ключа 'obs'
            if "obs" not in obs_space.original_space.spaces:
                 raise ValueError("Observation space Dict must contain key 'obs'")
            self.feature_space = obs_space.original_space["obs"] # Получаем оригинальное Box пространство фич
            if not isinstance(self.feature_space, gym.spaces.Box):
                 raise ValueError(f"obs_space['obs'] must be a Box space, got {type(self.feature_space)}")
            self.obs_size = int(np.product(self.feature_space.shape))
            logger.info(f"Initializing AdvancedPokerModel '{name}'. Original feature space: {self.feature_space}")
        else:
            # Этот случай не должен возникать с вашей средой, но оставляем для общей совместимости
            logger.warning(f"Observation space is not Dict ({type(obs_space)}), using its shape directly. Action masking might not work correctly if expected.")
            self.feature_space = obs_space
            if hasattr(obs_space, 'shape'):
                 self.obs_size = int(np.product(obs_space.shape))
            else:
                 raise ValueError(f"Cannot determine size from non-Dict, non-Box obs_space: {obs_space}")
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

        # Проверка action_space (должно быть Discrete для PPO по умолчанию)
        if not isinstance(action_space, gym.spaces.Discrete):
             logger.warning(f"Expected Discrete action space, got {type(action_space)}")

        self.num_outputs = num_outputs
        hiddens = model_config.get("fcnet_hiddens", [256, 256])
        activation = model_config.get("fcnet_activation", "relu")

        logger.info(f"Building FC layers with hiddens: {hiddens}, activation: {activation}")
        logger.info(f"Feature size: {self.obs_size}, Action space: {action_space}, Num outputs: {num_outputs}")

        # Получение функции активации
        try:
            activation_fn = getattr(nn, activation.capitalize(), None)
            if activation_fn is None: # Если capitalize не дал имя класса
                 # Попробуем напрямую, если имя уже правильное (например, ReLU)
                 activation_fn = getattr(nn, activation, nn.ReLU)
                 if activation_fn == nn.ReLU and activation != "ReLU":
                      logger.warning(f"Activation '{activation}' not found in nn. Using default ReLU.")
            elif activation_fn == nn.ReLU and activation.lower() != "relu": # Проверка, если capitalize сработало, но это был не тот ReLU
                 logger.warning(f"Activation '{activation}' mapped to ReLU. If this is unintended, check spelling.")

        except AttributeError:
             logger.warning(f"Could not find activation '{activation}' in torch.nn. Using default ReLU.")
             activation_fn = nn.ReLU

        # Построение слоев
        layers = []
        last_layer_size = self.obs_size
        for i, size in enumerate(hiddens):
            layers.append(nn.Linear(last_layer_size, size))
            layers.append(activation_fn())
            last_layer_size = size

        self._features = nn.Sequential(*layers)
        # Голова для логитов действий
        self._logits = nn.Linear(last_layer_size, self.num_outputs)
        # Голова для значения состояния (Value Function)
        self._value_branch = nn.Linear(last_layer_size, 1)
        self._last_features = None # Буфер для хранения выхода self._features

        logger.info(f"Model '{name}' built successfully. Final layer sizes: Logits({self.num_outputs}), Value(1)")

    @override(ModelV2)
    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        """
        Выполняет прямой проход модели.
        Ожидает input_dict["obs"] в виде словаря {"obs": tensor, "action_mask": tensor}.
        Извлекает тензор "obs" и пропускает через сеть.
        Возвращает логиты действий и состояние (для RNN).
        """
        # --- ИЗМЕНЕНО: Извлекаем 'obs' из словаря input_dict['obs'] ---
        if isinstance(input_dict["obs"], dict):
            # Ожидаемый случай для Dict observation space
            if "obs" not in input_dict["obs"]:
                 raise ValueError("input_dict['obs'] is a dict but missing 'obs' key.")
            features = input_dict["obs"]["obs"].float()
            # Маска доступна в input_dict["obs"]["action_mask"], но модель ее не использует напрямую
        elif torch.is_tensor(input_dict["obs"]):
            # Если пришел тензор (например, obs_flat или среда без Dict)
            logger.debug("Received flattened obs tensor instead of dict. Using it directly.")
            features = input_dict["obs"].float()
        else:
            raise ValueError(f"Unexpected type for input_dict['obs']: {type(input_dict['obs'])}")
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        # Проверка размерности входа
        expected_size = self.obs_size
        if features.shape[-1] != expected_size:
            logger.warning(f"Input feature size mismatch! Expected {expected_size}, got {features.shape[-1]}. Ensure obs_space setup is correct.")
            # Можно попробовать изменить размер, но это рискованно
            # features = features.view(features.shape[0], expected_size)

        try:
             # Пропускаем фичи через основные слои
             self._last_features = self._features(features)
             # Получаем логиты для действий
             logits = self._logits(self._last_features)

             # RLlib автоматически применит маску к этим логитам!
             return logits, state
        except Exception as e:
             logger.error(f"Model forward pass error: {e}", exc_info=True)
             logger.error(f"Error occurred with input features shape: {features.shape}")
             raise

    @override(ModelV2)
    def value_function(self) -> TensorType:
        """Возвращает оценку состояния (value) на основе последнего выхода self._features."""
        assert self._last_features is not None, "Must call forward() first"
        try:
            # Пропускаем сохраненные фичи через value-голову
            value = self._value_branch(self._last_features).squeeze(1) # Убираем лишнюю размерность
            return value
        except Exception as e:
             logger.error(f"Model value_function error: {e}", exc_info=True)
             raise
