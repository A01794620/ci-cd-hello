FROM  python:3.9-slim

WORKDIR /app

COPY model_pkl/wine_model_.pkl /app/model_pkl/
COPY src/main.py /app/
COPY requirements.txt /app/

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]