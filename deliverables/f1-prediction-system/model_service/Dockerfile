FROM python:3.12-slim

# Set working directory for the application
WORKDIR /app

# Copy Python dependencies and install them.  The requirements file lives at the
# root of the model_service context, not nested under a second "model_service"
# directory, so we copy it directly.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the service into the image.  This includes the FastAPI
# application under ``src`` as well as our vendored ``database`` client code.
COPY . /app

# Set PYTHONPATH so that both our code under ``src`` and the vendored
# ``database`` package are importable.
ENV PYTHONPATH=/app

# Expose the port the FastAPI server listens on.  The compose file maps this
# port to the host.
EXPOSE 8000

# Run the application.  We don't rely on a ``model_service`` package entry
# point; instead we invoke uvicorn directly with the module path to the FastAPI app.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]