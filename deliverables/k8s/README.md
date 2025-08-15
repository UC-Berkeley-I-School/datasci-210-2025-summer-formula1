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

## Setup the EKS Cluster

Install `kubectl` if not already installed: https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/

```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
kubectl version --client
```

Install `eksctl` to manage EKS clusters: https://eksctl.io/installation/

```bash
# for ARM systems, set ARCH to: `arm64`, `armv6` or `armv7`
ARCH=amd64
PLATFORM=$(uname -s)_$ARCH

curl -sLO "https://github.com/eksctl-io/eksctl/releases/latest/download/eksctl_$PLATFORM.tar.gz"

# (Optional) Verify checksum
curl -sL "https://github.com/eksctl-io/eksctl/releases/latest/download/eksctl_checksums.txt" | grep $PLATFORM | sha256sum --check

tar -xzf eksctl_$PLATFORM.tar.gz -C /tmp && rm eksctl_$PLATFORM.tar.gz

sudo install -m 0755 /tmp/eksctl /usr/local/bin && rm /tmp/eksctl
```

Spin up a cluster with 3 x `m6i.large` nodes (adjust type/size as you like):
```
export CLUSTER_NAME=f1
export REGION=us-east-1

eksctl create cluster \
  --name "$CLUSTER_NAME" \
  --region "$REGION" \
  --nodes 3 \
  --node-type m6i.large \
  --node-volume-size 80 \
  --with-oidc
```

It can up to ~20 minutes to create the stacks in Cloudformation so don't worry if it takes a while. When you create the cluster, check the progress of the stack in Cloudformation console: https://console.aws.amazon.com/cloudformation/home.

Create a managed node group for the cluster:

```bash
# change the node type and size as needed
# e.g., m6i.large, m6i.xlarge, etc.
# see https://aws.amazon.com/ec2/instance-types/m6i/

eksctl create nodegroup \
  --cluster "$CLUSTER_NAME" \
  --region "$REGION" \
  --name ng-1 \
  --nodes 3 \
  --node-type m6i.large \
  --node-volume-size 80 \
  --managed
```

- **Note**: Why not Fargate? Fargate is great, but we are using the community nginx-ingress controller (Service type LoadBalancer), which is really awkward to run on Fargate; EC2 nodes are more straightforward.

Hook up `kubectl` to the EKS cluster:

```bash
aws eks update-kubeconfig --name "$CLUSTER_NAME" --region "$REGION"
```

Verify:
```bash
kubectl get nodes -n f1
# No resources found
```

## Apply Kubernetes Manifests

Nginx Ingress Controller is required for routing traffic to your services. Apply the Nginx Ingress Controller manifest:

```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/cloud/deploy.yaml
```

Install Cert-Manager for managing TLS certificates:

```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/latest/download/cert-manager.yaml
kubectl -n cert-manager get pods -w
```

Upgrade the gp2 storage class to gp3 (if applicable):

First, check the current storage classes. Most likely you will see just one called `gp2`:

```bash
$ kubectl get storageclass
# NAME   PROVISIONER             RECLAIMPOLICY   VOLUMEBINDINGMODE      ALLOWVOLUMEEXPANSION   AGE
# gp2    kubernetes.io/aws-ebs   Delete          WaitForFirstConsumer   false                  2d22h
```

```bash
# Remove "default" annotation from old gp2 (safe):
kubectl patch storageclass gp2 \
  -p '{"metadata":{"annotations":{"storageclass.kubernetes.io/is-default-class":"false"}}}'

# Create the new default
kubectl apply -f 05-storageclass.yaml
```

Afterwards, confirm that the `gp3` storage class is set as default:

```bash
$ kubectl get storageclasses
# NAME            PROVISIONER             RECLAIMPOLICY   VOLUMEBINDINGMODE      ALLOWVOLUMEEXPANSION   AGE
# gp2             kubernetes.io/aws-ebs   Delete          WaitForFirstConsumer   false                  3d
# gp3 (default)   ebs.csi.aws.com         Delete          WaitForFirstConsumer   false                  9s
```

Install the EBS CSI driver (needed for the `db` StatefulSet):

The database uses a `PersistentVolumeClaim`. Without the EBS CSI, PVCs stay Pending.

```bash

# Create an IAM OIDC provider for the cluster
eksctl utils associate-iam-oidc-provider --region=us-east-1 --cluster=f1 --approve

# Create an IAM role for the EBS CSI controller ServiceAccount
eksctl create iamserviceaccount \
  --cluster "$CLUSTER_NAME" \
  --region "$REGION" \
  --namespace kube-system \
  --name ebs-csi-controller-sa \
  --role-name AmazonEKS_EBS_CSI_DriverRole \
  --attach-policy-arn arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy \
  --approve --override-existing-serviceaccounts

# Install the managed add-on bound to that role
# ATTENTION: Replace <YOUR_ACCOUNT_ID> with your actual AWS account ID!!!
aws eks create-addon \
  --cluster-name "$CLUSTER_NAME" \
  --region "$REGION" \
  --addon-name aws-ebs-csi-driver \
  --service-account-role-arn arn:aws:iam::<YOUR_ACCOUNT_ID>:role/AmazonEKS_EBS_CSI_DriverRole \
  --resolve-conflicts OVERWRITE
```

Verify:

```bash
kubectl -n kube-system get pods -l app.kubernetes.io/name=aws-ebs-csi-driver
```

Deploy all Kubernetes manifests:

```bash
kubectl apply -f 00-namespace.yaml
kubectl apply -f 01-secrets.yaml
kubectl apply -f 02-configmaps.yaml
kubectl apply -f 10-db-statefulset.yaml
kubectl apply -f 11-model-service-deploy.yaml
kubectl apply -f 12-monte-carlo-deploy.yaml
kubectl apply -f 13-webapp-deploy.yaml
kubectl apply -f 20-ingress-issuer.yaml
kubectl apply -f 21-ingress.yaml
```