from __future__ import annotations

from pathlib import Path

import pandas as pd


ROWS = [
    {"title": "Wireless earbuds with case", "vendor": "SoundMax", "description": "Bluetooth earphones for спорт и музыка", "category_ind": 0},
    {"title": "Gaming headset RGB", "vendor": "HyperBeat", "description": "Наушники для игр с микрофоном", "category_ind": 0},
    {"title": "Noise cancelling headphones", "vendor": "BassBoost", "description": "Полноразмерные наушники для поездок", "category_ind": 0},
    {"title": "USB C phone charger", "vendor": "VoltUp", "description": "Быстрая зарядка смартфона", "category_ind": 1},
    {"title": "65W wall adapter", "vendor": "VoltUp", "description": "Зарядное устройство для ноутбука", "category_ind": 1},
    {"title": "Power bank 20000 mah", "vendor": "EnergyBox", "description": "Портативный аккумулятор", "category_ind": 1},
    {"title": "Smart watch sport", "vendor": "TimeGo", "description": "Умные часы с пульсометром", "category_ind": 2},
    {"title": "Fitness band pro", "vendor": "TimeGo", "description": "Фитнес браслет с шагомером", "category_ind": 2},
    {"title": "Kids smart watch", "vendor": "KidSafe", "description": "Детские смарт часы с GPS", "category_ind": 2},
    {"title": "Wireless earbuds mini", "vendor": "MiniPods", "description": "Компактные наушники для телефона", "category_ind": 0},
    {"title": "Laptop charger type c", "vendor": "ChargeIT", "description": "Сетевой адаптер 100W", "category_ind": 1},
    {"title": "Smartwatch classic", "vendor": "Chronos", "description": "Часы с AMOLED экраном", "category_ind": 2},
]


if __name__ == "__main__":
    target = Path("project/data/demo/train.csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ROWS).to_csv(target, index=False)
    print(f"Demo dataset written to {target}")
