FROM python:3.12-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY model_service/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy both model_service and database source code into the image
COPY model_service /app/model_service
COPY database /app/database

# Set PYTHONPATH so both model-service and database are importable
ENV PYTHONPATH=/app

WORKDIR /app/model-service

EXPOSE 8088

CMD ["env", "PYTHONPATH=/app", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8088", "--reload"]
