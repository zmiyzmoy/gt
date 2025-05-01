"""
Регистрация моделей для RLlib.
Импортируйте этот модуль перед запуском обучения.
"""
import os
import logging
import sys

logger = logging.getLogger(__name__)

try:
    import torch
    import numpy as np
    from ray.rllib.models import ModelCatalog
    from models_fixed import AdvancedPokerModel
    
    # Регистрация модели в каталоге моделей RLlib
    def register_models():
        """Регистрирует все нужные модели в RLlib."""
        logger.info("Registering custom models...")
        ModelCatalog.register_custom_model("AdvancedPokerModel", AdvancedPokerModel)
        logger.info("Models registered successfully")
        return True
        
except ImportError as e:
    logger.warning(f"Could not import required modules for model registration: {e}")
    logger.warning("Models will not be registered")
    
    def register_models():
        """Заглушка при отсутствии необходимых модулей."""
        logger.error("Models cannot be registered due to missing dependencies")
        return False

if __name__ == "__main__":
    # Эту функцию можно вызвать отдельно для проверки регистрации моделей
    success = register_models()
    if success:
        print("✅ Models registered successfully")
    else:
        print("❌ Failed to register models")
        sys.exit(1)