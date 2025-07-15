# FastAPI Application

This is a FastAPI application that provides endpoints for health checking, greeting users, and making housing price predictions.

## Features

- `/lab/predict`: Accepts housing data and returns a housing price prediction.
- `/lab/bulk-predict`: Accepts multiple housing data and returns a housing price prediction for each.
- `/lab/hello`: Greets the user with a custom message.
- `/lab/health`: Returns the current UTC time for health checking.
- OpenAPI documentation is available at `/docs` and `/openapi.json`.

## Training the Model

To train the model, run the provided [train.py](./trainer/train.py) script. The trained model is saved to `model_pipeline.pkl`.

```bash
poetry run python trainer/train.py
```

## Building the Application

1. Ensure Docker is installed on your system.
2. Clone this repository:

   ```bash
   git clone git@github.com:UCB-W255/lab-3-caching-and-kubernetes-seansica.git
   cd lab-3-caching-and-kubernetes-seansica
   ```

3. Build the Docker image:

   ```bash
   docker build -t lab3-app:v2 .
   ```

### Docker Desktop vs Minikube Docker Environment

When working with Minikube, it’s important to understand that Minikube runs its own Docker environment, separate from Docker Desktop (or any other Docker engine you might be using on your system). This means:

- **Docker Desktop environment**: This is your default Docker environment that you typically interact with using `docker` commands on your system.
- **Minikube Docker environment**: Minikube runs its own Docker daemon inside the Minikube VM. If you build an image in Docker Desktop, Minikube won’t automatically have access to it unless you explicitly load the image into Minikube’s Docker environment.

### Switching Between Docker Environments

To interact with Minikube’s Docker environment, you need to switch your terminal’s Docker context to Minikube using the `eval` command:

```bash
eval $(minikube docker-env)
```

After running this command, any `docker` commands you execute in that terminal (e.g., `docker build`, `docker images`, etc.) will be applied inside the Minikube VM's Docker environment instead of Docker Desktop.

To switch back to your default Docker Desktop environment, you can run:

```bash
eval $(minikube docker-env -u)
```

This command resets your terminal to use your local Docker engine.

### Deploying the Application in Minikube

1. **Start Minikube** (if it's not running):

   ```bash
   minikube start
   ```

2. **Load the Docker image into Minikube**:

   If you've built the Docker image in Docker Desktop and need to load it into Minikube:

   ```bash
   minikube image load lab3-app:v2
   ```

   This command transfers the image from Docker Desktop to Minikube’s Docker environment.

   Alternatively, you can switch to the Minikube Docker environment using the `eval` command (as mentioned earlier) and build the image directly in Minikube:

   ```bash
   eval $(minikube docker-env)
   docker build -t lab3-app:v2 .
   ```

3. **Deploy the application**:

   Apply the Kubernetes resources (namespace, deployments, and services):

   ```bash
   kubectl apply -f infra/namespace.yaml
   kubectl apply -f infra/deployment-redis.yaml
   kubectl apply -f infra/deployment-pythonapi.yaml
   kubectl apply -f infra/service-redis.yaml
   kubectl apply -f infra/service-prediction.yaml
   ```

4. **Expose the FastAPI service using Minikube**:

   Run `minikube tunnel` to expose the `LoadBalancer` service locally:

   ```bash
   minikube tunnel
   ```

   After running the tunnel, you can check the services to find the external IP:

   ```bash
   kubectl get services -n w255
   ```

   You should now see an external IP for the `prediction-api` service.

5. **Test the API**:

   Once you have the external IP, you can test the health endpoint:

   ```bash
   curl http://<external-ip>/lab/health
   ```

   Example:

   ```bash
   curl http://127.0.0.1:8000/lab/health
   ```

### Monitoring the Kubernetes Deployment

To monitor and manage the running pods and services, use the following `kubectl` commands:

- **Check all pods in the `w255` namespace**:

   ```bash
   kubectl get pods -n w255
   ```

- **Check all services in the `w255` namespace**:

   ```bash
   kubectl get services -n w255
   ```

- **View the logs of a specific pod**:

   ```bash
   kubectl logs <pod-name> -n w255
   ```

- **Describe a specific pod for detailed info**:

   ```bash
   kubectl describe pod <pod-name> -n w255
   ```

- **Delete and recreate all pods in the `python-api` deployment** (if needed):

   ```bash
   kubectl delete pod -n w255 -l app=python-api
   ```

## Development Setup (Local Hot-Reload)

For developers who want to run the app locally with hot-reload:

1. Ensure you have Python 3.8+ and Poetry installed on your system.

2. Clone the repository (if you haven't already):

   ```bash
   git clone git@github.com:UCB-W255/lab-3-caching-and-kubernetes-seansica.git
   cd lab-3-caching-and-kubernetes-seansica
   ```

3. Install dependencies with Poetry (disable virtual environments):

   ```bash
   poetry config virtualenvs.create false
   poetry install
   ```

4. Run the application with Uvicorn in reload mode:

   ```bash
   uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
   ```

The API will be accessible at `http://localhost:8000`, and any changes to the code will trigger an automatic reload.

## Testing the Application

1. Ensure you have Python and pytest installed.
2. Run the tests:

   ```bash
   pytest
   ```

This will execute all test cases in the `tests/` directory.

### Manual Integration Tests

```bash
# Test single prediction endpoint with valid input
curl -X POST http://localhost/lab/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.984127,
    "AveBedrms": 1.02381,
    "Population": 322.0,
    "AveOccup": 2.555556,
    "Latitude": 37.88,
    "Longitude": -122.23
  }'
# {"prediction":4.413138531975459}%

# Test bulk prediction endpoint with multiple houses
curl -X POST http://localhost/lab/bulk-predict \
  -H "Content-Type: application/json" \
  -d '{
    "houses": [
      {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984127,
        "AveBedrms": 1.02381,
        "Population": 322.0,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23
      },
      {
        "MedInc": 7.2574,
        "HouseAge": 28.0,
        "AveRooms": 5.703427,
        "AveBedrms": 1.07381,
        "Population": 334.0,
        "AveOccup": 3.333556,
        "Latitude": 36.77,
        "Longitude": -121.87
      }
    ]
  }'
# {"predictions":[4.413138531975459,3.438628597491838]}%

# Test predict endpoint with invalid latitude (should return 422)
curl -X POST http://localhost/lab/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.984127,
    "AveBedrms": 1.02381,
    "Population": 322.0,
    "AveOccup": 2.555556,
    "Latitude": 95.0,
    "Longitude": -122.23
  }'
# {"detail":[{"type":"value_error","loc":["body","Latitude"],"msg":"Value error, Invalid value for Latitude","input":95.0,"ctx":{"error":{}}}]}%


# Test predict endpoint with missing field (should return 422)
curl -X POST http://localhost/lab/predict \
  -H "Content-Type: application/json" \
  -d '{
    "HouseAge": 41.0,
    "AveRooms": 6.984127,
    "AveBedrms": 1.02381,
    "Population": 322.0,
    "AveOccup": 2.555556,
    "Latitude": 37.88,
    "Longitude": -122.23
  }'
# {"detail":[{"type":"missing","loc":["body","MedInc"],"msg":"Field required","input":{"HouseAge":41.0,"AveRooms":6.984127,"AveBedrms":1.02381,"Population":322.0,"AveOccup":2.555556,"Latitude":37.88,"Longitude":-122.23}}]}%
```

## API Documentation

Once the application is running, you can access the API documentation:

- Swagger UI: `http://localhost/docs`
- OpenAPI JSON: `http://localhost/openapi.json`
