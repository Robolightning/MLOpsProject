$ErrorActionPreference = "Stop"

python project/scripts/make_demo_dataset.py
python project/scripts/run_eda.py --config project/configs/base.yaml
python project/scripts/run_experiment.py --config project/configs/base.yaml
pytest -q project/tests
