### ResNet-18 Low-Bit Quantization using MHAQ.


![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![CI](https://github.com/iamyblitz/Quantization-using-MHAQ/actions/workflows/lint.yml/badge.svg)

#### О проекте (About)
Данный репозиторий содержит скрипты модификации и конфигурационные файлы для экстремально низкобитного квантования (1, 3 и 4 бита) модели **ResNet-18** с использованием фреймворка [MHAQ](https://github.com/aifoundry-org/MHAQ).

Архитектура решения:
1. Исходный фреймворк MHAQ склонирован локально.
2. Python-скрипты из папки `patches/` (`snippet1.py` и др.) используются для инъекции кода и исправления ошибок запуска калибровки в MHAQ.
3. Процесс обучения запускается на графическом ускорителе **NVIDIA Tesla P100**.

#### Установка и запуск (Usage)
1. Склонируйте оригинальный репозиторий:
   ```bash
   git clone https://github.com/aifoundry-org/MHAQ
   ```

2. Запустите скрипты для перезаписи целевых файлов фреймворка:
```
python patches/snippet1.py
python patches/snippet2.py
python patches/snippet3.py
```

3. Запустите процесс квантования с нужным конфигом (w1a1, w2a2 или w4a4).

#### Результаты
Эксперименты проводились с логированием метрик в Weights & Biases (WandB). В ходе экспериментов не было зафиксировано падений оригинального кода (благодаря примененным патчам).
[report](https://api.wandb.ai/links/iamyblitz-saint-petersburg-state-university/e08j9med)