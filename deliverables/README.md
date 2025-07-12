# F1 Prediction System

## Software Planning

- [] Hydrate DB with time series (Timescale?)
  - Write a db hydration service
  - Runs indepdenently (separate image/pod)
  - Responsible for keeping the DB clean

- [] Serve models in containers/pods
  - Write a simple Python microservice
  - Loads model at runtime
  - Async hot start, nothing fancy
  - Has a REST interface
    ```
    GET /prediction
    {
        
    }
    ```
  - Returns 