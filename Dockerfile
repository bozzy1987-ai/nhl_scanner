FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn pandas numpy xgboost scikit-learn joblib pydantic

COPY model/ /app/model/
COPY data/ /app/data/
COPY main.py /app/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
