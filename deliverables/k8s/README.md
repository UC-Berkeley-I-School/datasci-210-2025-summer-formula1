# K8S Setup Notes

## Build and Push Docker Images to AWS ECR

### Webapp

```bash
docker compose -f compose.prod.yaml build webapp
# deliverables-webapp:latest

aws ecr create-repository --repository-name f1/webapp

aws ecr get-login-password --region us-east-1 \
 | docker login --username AWS --password-stdin 590184119443.dkr.ecr.us-east-1.amazonaws.com/f1/webapp

docker tag deliverables-webapp:latest 590184119443.dkr.ecr.us-east-1.amazonaws.com/f1/webapp:latest

docker push 590184119443.dkr.ecr.us-east-1.amazonaws.com/f1/webapp:latest
```

### Database

```bash
docker compose -f compose.prod.yaml build db
# f1-timescaledb:latest

aws ecr create-repository --repository-name f1/database

aws ecr get-login-password --region us-east-1 \
 | docker login --username AWS --password-stdin 590184119443.dkr.ecr.us-east-1.amazonaws.com/f1/database

docker tag f1-timescaledb:latest 590184119443.dkr.ecr.us-east-1.amazonaws.com/f1/database:latest

docker push 590184119443.dkr.ecr.us-east-1.amazonaws.com/f1/database:latest
```

### Model Service

```bash
docker compose -f compose.prod.yaml build model_service
# deliverables-model_service:latest

aws ecr create-repository --repository-name f1/model_service

aws ecr get-login-password --region us-east-1 \
 | docker login --username AWS --password-stdin 590184119443.dkr.ecr.us-east-1.amazonaws.com/f1/model_service

docker tag deliverables-model_service:latest 590184119443.dkr.ecr.us-east-1.amazonaws.com/f1/model_service:latest

docker push 590184119443.dkr.ecr.us-east-1.amazonaws.com/f1/model_service:latest
```

### Monte Carlo Simulation

```bash
docker compose -f compose.prod.yaml build monte_carlo_sim
# deliverables-monte_carlo_sim

aws ecr create-repository --repository-name f1/monte_carlo_sim

aws ecr get-login-password --region us-east-1 \
 | docker login --username AWS --password-stdin 590184119443.dkr.ecr.us-east-1.amazonaws.com/f1/monte_carlo_sim

 docker tag deliverables-monte_carlo_sim:latest 590184119443.dkr.ecr.us-east-1.amazonaws.com/f1/monte_carlo_sim:latest

docker push 590184119443.dkr.ecr.us-east-1.amazonaws.com/f1/monte_carlo_sim:latest
```

### Nginx

```bash
docker compose -f compose.prod.yaml build nginx
# deliverables-nginx

aws ecr create-repository --repository-name f1/nginx

aws ecr get-login-password --region us-east-1 \
 | docker login --username AWS --password-stdin 590184119443.dkr.ecr.us-east-1.amazonaws.com/f1/nginx

 docker tag deliverables-nginx:latest 590184119443.dkr.ecr.us-east-1.amazonaws.com/f1/nginx:latest

docker push 590184119443.dkr.ecr.us-east-1.amazonaws.com/f1/nginx:latest
```

## Add Models to S3

Create an S3 bucket named `f1models` in the `us-east-1` region and make sure it's publicly accessible. Upload your model files (e.g., `models.zip`) to the bucket.

e.g., 
```
https://f1models.s3.us-east-1.amazonaws.com/models.zip
```