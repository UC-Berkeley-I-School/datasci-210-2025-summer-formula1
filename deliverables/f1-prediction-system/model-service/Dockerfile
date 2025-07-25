FROM python:3.12-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Set PYTHONPATH so model-service can import database package
ENV PYTHONPATH=/app/../

EXPOSE 8088

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8088", "--reload"]
