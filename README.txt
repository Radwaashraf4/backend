Run:
  pip install -r requirements.txt
  python integration_layer/api.py        -> http://localhost:5000/api/health
Tests:
  python -m unittest discover -s integration_layer -p "test_*.py"
build_model.py is only needed to rebuild the model: put earth_features.csv and mars_features.csv in data/features/ first.
