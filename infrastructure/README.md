# 🏗️ Cognify Infrastructure

This directory contains infrastructure configurations for deploying Cognify in production environments.

## 📁 Directory Structure

```
infrastructure/
├── kubernetes/          # Kubernetes manifests
│   ├── deployment.yaml  # App deployments (API, Qdrant, Redis)
│   └── service.yaml     # Services, Ingress, ConfigMaps, Secrets
├── docker/             # Docker-related configs
│   └── prometheus.yml  # Prometheus monitoring configuration
└── README.md           # This file
```

## 🚀 Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (v1.20+)
- kubectl configured
- Ingress controller (nginx)
- cert-manager for SSL certificates
- Storage class `fast-ssd` available

### Quick Deploy

```bash
# Create namespace
kubectl create namespace cognify

# Apply configurations
kubectl apply -f kubernetes/

# Check deployment status
kubectl get pods -n cognify
kubectl get services -n cognify
kubectl get ingress -n cognify
```

### Environment Variables

Before deploying, set these environment variables:

```bash
export DB_PASSWORD="your-secure-db-password"
export OPENAI_API_KEY="your-openai-api-key"
export JWT_SECRET_KEY="your-jwt-secret-key"
```

### Services Deployed

1. **Cognify API** (3 replicas)
   - Port: 8000
   - Health checks enabled
   - Resource limits: 1Gi RAM, 500m CPU

2. **Qdrant Vector Database**
   - Port: 6333 (HTTP), 6334 (gRPC)
   - Persistent storage: 10Gi
   - Resource limits: 2Gi RAM, 1000m CPU

3. **Redis Cache**
   - Port: 6379
   - Persistent storage: 5Gi
   - Resource limits: 512Mi RAM, 200m CPU

### Ingress Configuration

- **Domain**: api.cognify.ai
- **SSL**: Automatic via cert-manager
- **Rate Limiting**: 100 requests/minute
- **Force HTTPS**: Enabled

## 📊 Monitoring

### Prometheus Configuration

The `docker/prometheus.yml` configures monitoring for:

- **Cognify API**: Metrics endpoint `/metrics`
- **Qdrant**: Vector database metrics
- **Redis**: Cache performance metrics
- **PostgreSQL**: Database metrics
- **System**: Node exporter metrics

### Metrics Collection

```yaml
Scrape Intervals:
- API: 5s
- Databases: 10s
- System: 10s
```

### Setting Up Monitoring Stack

```bash
# Deploy Prometheus (example with Helm)
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --values infrastructure/docker/prometheus.yml
```

## 🔧 Configuration Management

### Secrets

Required secrets in `cognify-secrets`:
- `database-url`: PostgreSQL connection string
- `openai-api-key`: OpenAI API key
- `jwt-secret-key`: JWT signing key

### ConfigMaps

Configuration in `cognify-config`:
- `qdrant-url`: Qdrant service URL
- `redis-url`: Redis service URL
- `environment`: Production environment flag

## 📈 Scaling

### Horizontal Pod Autoscaler

```bash
# Enable HPA for API pods
kubectl autoscale deployment cognify-api \
  --namespace cognify \
  --cpu-percent=70 \
  --min=3 \
  --max=10
```

### Resource Monitoring

```bash
# Check resource usage
kubectl top pods -n cognify
kubectl top nodes
```

## 🔒 Security

### Security Features

- **Non-root containers**: All containers run as non-root
- **Read-only filesystem**: API containers use read-only root filesystem
- **Security contexts**: Proper security contexts applied
- **Network policies**: Isolated network communication
- **SSL/TLS**: End-to-end encryption

### Security Scanning

```bash
# Scan images for vulnerabilities
docker scan cognify/api:latest
```

## 🚨 Troubleshooting

### Common Issues

1. **Pod not starting**
   ```bash
   kubectl describe pod <pod-name> -n cognify
   kubectl logs <pod-name> -n cognify
   ```

2. **Service not accessible**
   ```bash
   kubectl get endpoints -n cognify
   kubectl get ingress -n cognify
   ```

3. **Database connection issues**
   ```bash
   kubectl exec -it <api-pod> -n cognify -- env | grep DATABASE
   ```

### Health Checks

```bash
# Check all service health
kubectl get pods -n cognify
curl -f https://api.cognify.ai/health
```

## 📚 Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Prometheus Monitoring](https://prometheus.io/docs/)
- [cert-manager](https://cert-manager.io/docs/)

## 🤝 Contributing

When modifying infrastructure:

1. Test changes in development environment
2. Update this README if needed
3. Validate Kubernetes manifests: `kubectl apply --dry-run=client`
4. Test deployment in staging before production
