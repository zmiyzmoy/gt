# models.py
import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
from ray.rllib.utils.framework import try_import_torch
import logging

logger = logging.getLogger(__name__)
torch, nn = try_import_torch()

class AdvancedPokerModel(TorchModelV2, nn.Module):
    """Пример кастомной модели, совместимой с ModelV2 API."""
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.obs_size = int(np.product(obs_space.shape))
        self.num_outputs = num_outputs
        hiddens = model_config.get("fcnet_hiddens", [256, 256])
        activation = model_config.get("fcnet_activation", "relu")

        logger.info(f"Building AdvancedPokerModel '{name}' with hiddens: {hiddens}, activation: {activation}")
        logger.info(f"Obs size: {self.obs_size}, Action space: {action_space}")

        try:
            activation_fn = getattr(nn, activation.capitalize())
        except AttributeError:
            logger.warning(f"Activation '{activation}' not found. Using ReLU.")
            activation_fn = nn.ReLU

        layers = []
        last_layer_size = self.obs_size
        for i, size in enumerate(hiddens):
            layers.append(nn.Linear(last_layer_size, size))
            layers.append(activation_fn())
            last_layer_size = size

        self._features = nn.Sequential(*layers)
        self._logits = nn.Linear(last_layer_size, num_outputs)
        self._value_branch = nn.Linear(last_layer_size, 1)
        self._last_features = None

        logger.info(f"Model '{name}' built successfully.")

    @override(ModelV2)
    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        obs = input_dict.get("obs_flat", input_dict["obs"]).float()
        try:
             self._last_features = self._features(obs)
             logits = self._logits(self._last_features)
             return logits, state
        except Exception as e:
             logger.error(f"Model forward pass error: {e}", exc_info=True)
             raise

    @override(ModelV2)
    def value_function(self) -> TensorType:
        assert self._last_features is not None, "Must call forward() first"
        try:
            value = self._value_branch(self._last_features).squeeze(1)
            return value
        except Exception as e:
             logger.error(f"Model value_function error: {e}", exc_info=True)
             raise
