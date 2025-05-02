#!/usr/bin/env python3
"""
Регистрация моделей для RLlib с улучшенной обработкой ошибок.
Импортируйте этот модуль перед запуском обучения.
"""

import logging
import importlib
import traceback

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("register_models")

def register_models():
    """Регистрирует все нужные модели в RLlib."""
    
    logger.info("Registering custom models...")
    
    try:
        # Проверяем наличие необходимых модулей
        ray_available = importlib.util.find_spec("ray") is not None
        torch_available = importlib.util.find_spec("torch") is not None
        numpy_available = importlib.util.find_spec("numpy") is not None
        
        if not all([ray_available, torch_available, numpy_available]):
            missing = []
            if not ray_available: missing.append("ray")
            if not torch_available: missing.append("torch")
            if not numpy_available: missing.append("numpy")
            logger.warning(f"Required modules not available: {', '.join(missing)}")
            return
        
        # Импортируем модули, если они доступны
        import torch
        import numpy as np
        from ray.rllib.models import ModelCatalog
        
        # Импортируем наши модели
        try:
            from models_improved import AdvancedPokerModel
            logger.info("Using improved model implementations")
        except ImportError:
            logger.warning("Could not import models_improved, trying models_fixed")
            try:
                from models_fixed import AdvancedPokerModel
                logger.info("Using fixed model implementations")
            except ImportError:
                logger.error("Could not import neither models_improved nor models_fixed")
                return
        
        # Регистрируем модели в ModelCatalog Ray RLlib
        ModelCatalog.register_custom_model("AdvancedPokerModel", AdvancedPokerModel)
        logger.info("Models registered successfully")
        
    except Exception as e:
        logger.error(f"Error registering models: {e}")
        logger.error(traceback.format_exc())
        
# В случае прямого запуска модуля
if __name__ == "__main__":
    register_models()