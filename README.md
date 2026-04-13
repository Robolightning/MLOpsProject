# MLOps Project

Проект по задаче классификации категории товара по текстовым полям карточки. Выполнил Богданов А.А.

## Презентация

[Ссылка на презентацию](https://drive.google.com/file/d/1XVlQ6BfxW9ZDalFrWMET6jgRzohgS8QN/view?usp=sharing)

## Что реализовано

- EDA по исходным данным
- baseline-модель для классификации категорий:
  - TF-IDF
  - Logistic Regression
- воспроизводимый end-to-end запуск эксперимента
- трекинг параметров, метрик и артефактов в MLflow
- экспорт задач для ручной проверки в Label Studio

## Структура проекта

```text
project/
├── configs/     # конфиги экспериментов
├── data/        # данные
├── reports/     # результаты EDA и материалы
├── scripts/     # скрипты запуска
├── src/         # исходный код
└── tests/       # smoke tests
```

## Запуск

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python project/scripts/make_demo_dataset.py
python project/scripts/run_eda.py --config project/configs/base.yaml
python project/scripts/run_experiment.py --config project/configs/base.yaml
python -m pytest -q project/tests
```

## Результаты запуска

После выполнения команд должны появиться:

* `project/reports/figures/` — графики EDA
* `project/reports/eda_summary.json` — сводка EDA
* `project/artifacts/metrics_summary.json` — результаты экспериментов
* `project/artifacts/best_model/model.joblib` — лучшая модель
* `project/artifacts/label_studio_tasks.json` — задачи для ручной валидации
* `project/mlruns/` — логи экспериментов MLflow