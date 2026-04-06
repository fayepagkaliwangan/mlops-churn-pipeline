install:
    pip install -r requirements.txt

test:
    pytest tests/

run:
    uvicorn api.app:app --reload

docker-build:
    docker build -t churn-api .

clean:
    rm -rf __pycache__ .pytest_cache .coverage .flake8
