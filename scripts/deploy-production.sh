#!/bin/bash

# Production Deployment Script for Cognify
# This script deploys Cognify to production environment

set -e

echo "🚀 Starting Cognify Production Deployment..."

# Configuration
DOCKER_REGISTRY="cognify"
IMAGE_TAG="latest"
NAMESPACE="cognify"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check if .env file exists
    if [ ! -f ".env" ]; then
        log_error ".env file not found. Copy .env.production to .env and configure it."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    docker build -t ${DOCKER_REGISTRY}/api:${IMAGE_TAG} .
    
    if [ $? -eq 0 ]; then
        log_success "Docker image built successfully"
    else
        log_error "Failed to build Docker image"
        exit 1
    fi
}

# Deploy with Docker Compose
deploy_docker() {
    log_info "Deploying with Docker Compose..."
    
    # Load environment variables
    source .env
    
    # Deploy services
    docker-compose -f docker-compose.production.yml up -d
    
    if [ $? -eq 0 ]; then
        log_success "Docker Compose deployment successful"
    else
        log_error "Docker Compose deployment failed"
        exit 1
    fi
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    sleep 30
    
    # Check service health
    check_service_health
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Create namespace if it doesn't exist
    kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply Kubernetes manifests
    kubectl apply -f infrastructure/kubernetes/ -n ${NAMESPACE}
    
    if [ $? -eq 0 ]; then
        log_success "Kubernetes deployment successful"
    else
        log_error "Kubernetes deployment failed"
        exit 1
    fi
    
    # Wait for rollout to complete
    kubectl rollout status deployment/cognify-api -n ${NAMESPACE}
    
    # Check pod status
    kubectl get pods -n ${NAMESPACE}
}

# Check service health
check_service_health() {
    log_info "Checking service health..."
    
    # Check API health
    for i in {1..30}; do
        if curl -f http://localhost:8000/health > /dev/null 2>&1; then
            log_success "API is healthy"
            break
        else
            log_warning "Waiting for API to be healthy... (attempt $i/30)"
            sleep 10
        fi
        
        if [ $i -eq 30 ]; then
            log_error "API health check failed"
            exit 1
        fi
    done
    
    # Check Qdrant health
    if curl -f http://localhost:6333/health > /dev/null 2>&1; then
        log_success "Qdrant is healthy"
    else
        log_warning "Qdrant health check failed"
    fi
    
    # Check Redis health
    if docker exec cognify-redis redis-cli ping > /dev/null 2>&1; then
        log_success "Redis is healthy"
    else
        log_warning "Redis health check failed"
    fi
}

# Run production tests
run_tests() {
    log_info "Running production tests..."
    
    # Run API tests
    python -m pytest tests/test_production.py -v
    
    if [ $? -eq 0 ]; then
        log_success "Production tests passed"
    else
        log_error "Production tests failed"
        exit 1
    fi
}

# Main deployment function
main() {
    echo "🎯 Cognify Production Deployment"
    echo "================================"
    
    # Parse command line arguments
    DEPLOYMENT_TYPE=${1:-docker}
    
    case $DEPLOYMENT_TYPE in
        docker)
            log_info "Deploying with Docker Compose"
            check_prerequisites
            build_image
            deploy_docker
            run_tests
            ;;
        kubernetes|k8s)
            log_info "Deploying to Kubernetes"
            check_prerequisites
            build_image
            deploy_kubernetes
            run_tests
            ;;
        *)
            log_error "Invalid deployment type. Use 'docker' or 'kubernetes'"
            echo "Usage: $0 [docker|kubernetes]"
            exit 1
            ;;
    esac
    
    echo ""
    echo "🎉 Deployment completed successfully!"
    echo "================================"
    echo "API URL: http://localhost:8000"
    echo "API Docs: http://localhost:8000/docs"
    echo "Grafana: http://localhost:3000"
    echo "Prometheus: http://localhost:9090"
    echo ""
    echo "🔍 To check logs:"
    echo "docker-compose -f docker-compose.production.yml logs -f"
    echo ""
    echo "🛑 To stop services:"
    echo "docker-compose -f docker-compose.production.yml down"
}

# Run main function
main "$@"
