# Используем Python 3.11 slim
FROM python:3.11-slim

# Рабочая директория внутри контейнера
WORKDIR /app

# Копируем весь проект segmentator внутрь контейнера
COPY . /app
RUN pip install -r requirements.txt
# добавляем корень в PYTHONPATH
ENV PYTHONPATH=/app
# Открываем порты на случай, если позже запустим FastAPI/Streamlit
EXPOSE 8000 8501

# Контейнер запускается интерактивно
CMD ["sh", "-c", "uvicorn ml.service.backend.main:app --host 0.0.0.0 --port 8000 & streamlit run ml/service/backend/app.py --server.port 8501 --server.address 0.0.0.0"]