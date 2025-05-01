#!/usr/bin/env python3
"""
Основной файл для веб-интерфейса проекта Poker RL.
Обратите внимание: в данный момент это просто заглушка для Replit.
"""

from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import sys
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Инициализация Flask
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default_secret_key_for_development")

@app.route('/')
def index():
    """Главная страница."""
    return """
    <html>
        <head>
            <title>Poker RL Project</title>
            <link rel="stylesheet" href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css">
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: var(--bs-dark);
                    color: var(--bs-light);
                }
                .container {
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }
                h1, h2 {
                    color: var(--bs-info);
                }
                .card {
                    background-color: var(--bs-dark-bg-subtle);
                    border: 1px solid var(--bs-border-color);
                    border-radius: 5px;
                    padding: 15px;
                    margin-bottom: 20px;
                }
                code {
                    background-color: var(--bs-dark);
                    color: var(--bs-info);
                    padding: 2px 4px;
                    border-radius: 3px;
                }
                .btn-primary {
                    background-color: var(--bs-primary);
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    color: white;
                    text-decoration: none;
                    display: inline-block;
                    margin-top: 10px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Poker RL Project</h1>
                <div class="card">
                    <h2>О проекте</h2>
                    <p>
                        Этот проект представляет собой систему обучения с подкреплением (Reinforcement Learning) 
                        для игры в покер, использующую фреймворки Ray/RLlib, PyTorch и OpenSpiel.
                    </p>
                    <p>
                        <strong>Обратите внимание:</strong> Replit не оптимизирован для запуска ресурсоемких
                        задач машинного обучения. Для полноценного обучения рекомендуется использовать 
                        локальную среду или VPS с GPU.
                    </p>
                </div>
                
                <div class="card">
                    <h2>Команды для запуска</h2>
                    <p>Для проверки окружения:</p>
                    <code>python debug.py</code>
                    <p>Для запуска тестового эпизода:</p>
                    <code>python test_env.py --episodes 1</code>
                    <p>Для обучения модели:</p>
                    <code>python train_fixed.py --iterations 100</code>
                </div>
                
                <div class="card">
                    <h2>Статус</h2>
                    <p>Проект находится в стадии активной разработки и отладки.</p>
                    <p>Текущие файлы:</p>
                    <ul>
                        <li><code>config.py</code> - Конфигурация покерной игры и параметры обучения</li>
                        <li><code>environment_fixed.py</code> - Исправленная среда покера</li>
                        <li><code>models_fixed.py</code> - Исправленная модель нейронной сети</li>
                        <li><code>train_fixed.py</code> - Скрипт обучения с оптимизацией ресурсов</li>
                        <li><code>loggers.py</code> - Логгеры для интеграции с Weights & Biases</li>
                        <li><code>register_models.py</code> - Регистрация моделей для RLlib</li>
                    </ul>
                </div>
            </div>
        </body>
    </html>
    """

@app.route('/status')
def status():
    """Эндпоинт для проверки статуса API."""
    return jsonify({
        "status": "ok",
        "project": "Poker RL",
        "version": "0.1.0",
        "environment": "Replit"
    })

if __name__ == '__main__':
    # Запуск сервера Flask
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)