#!/bin/bash

# Staging Deployment Script for Cognify
# This script deploys Cognify to staging environment for testing

set -e

echo "🚀 Starting Cognify Staging Deployment..."

# Configuration
DOCKER_REGISTRY="cognify"
IMAGE_TAG="staging"
COMPOSE_FILE="docker-compose.staging.yml"

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
    
    # Check if docker-compose is available
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check if staging compose file exists
    if [ ! -f "$COMPOSE_FILE" ]; then
        log_error "Staging compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Clean up existing staging deployment
cleanup_staging() {
    log_info "Cleaning up existing staging deployment..."
    
    # Stop and remove existing containers
    docker-compose -f $COMPOSE_FILE down --volumes --remove-orphans 2>/dev/null || true
    
    # Remove staging image if exists
    docker rmi ${DOCKER_REGISTRY}/api:${IMAGE_TAG} 2>/dev/null || true
    
    log_success "Cleanup completed"
}

# Build staging image
build_staging_image() {
    log_info "Building staging Docker image..."
    
    docker build -t ${DOCKER_REGISTRY}/api:${IMAGE_TAG} .
    
    if [ $? -eq 0 ]; then
        log_success "Staging Docker image built successfully"
    else
        log_error "Failed to build staging Docker image"
        exit 1
    fi
}

# Deploy staging environment
deploy_staging() {
    log_info "Deploying staging environment..."
    
    # Start services
    docker-compose -f $COMPOSE_FILE up -d
    
    if [ $? -eq 0 ]; then
        log_success "Staging deployment successful"
    else
        log_error "Staging deployment failed"
        exit 1
    fi
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 45
    
    # Check service health
    check_staging_health
}

# Check staging health
check_staging_health() {
    log_info "Checking staging service health..."
    
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
            show_logs
            exit 1
        fi
    done
    
    # Check other services
    check_service "Qdrant" "http://localhost:6334/health"
    check_service "Prometheus" "http://localhost:9091/-/healthy"
    check_service "Grafana" "http://localhost:3001/api/health"
}

# Check individual service
check_service() {
    local service_name=$1
    local health_url=$2
    
    if curl -f $health_url > /dev/null 2>&1; then
        log_success "$service_name is healthy"
    else
        log_warning "$service_name health check failed (may be normal for some services)"
    fi
}

# Run staging tests
run_staging_tests() {
    log_info "Running staging tests..."
    
    # Run production tests against staging
    python -m pytest tests/test_production.py -v --tb=short
    
    if [ $? -eq 0 ]; then
        log_success "Staging tests passed"
    else
        log_warning "Some staging tests failed (check logs for details)"
    fi
}

# Show service logs
show_logs() {
    log_info "Showing service logs..."
    docker-compose -f $COMPOSE_FILE logs --tail=50
}

# Show staging status
show_staging_status() {
    echo ""
    echo "🎉 Staging Deployment Status"
    echo "============================"
    echo ""
    
    # Show container status
    log_info "Container Status:"
    docker-compose -f $COMPOSE_FILE ps
    
    echo ""
    log_info "Service URLs:"
    echo "  🌐 API: http://localhost:8000"
    echo "  📚 API Docs: http://localhost:8000/docs"
    echo "  🔍 Health: http://localhost:8000/health"
    echo "  📊 Grafana: http://localhost:3001 (admin/staging_grafana_admin_123)"
    echo "  📈 Prometheus: http://localhost:9091"
    echo "  🗄️ Qdrant: http://localhost:6334"
    
    echo ""
    log_info "Useful Commands:"
    echo "  📋 View logs: docker-compose -f $COMPOSE_FILE logs -f"
    echo "  🛑 Stop staging: docker-compose -f $COMPOSE_FILE down"
    echo "  🔄 Restart service: docker-compose -f $COMPOSE_FILE restart cognify-api"
    echo "  🧪 Run tests: python -m pytest tests/test_production.py -v"
}

# Main deployment function
main() {
    echo "🎯 Cognify Staging Deployment"
    echo "============================="
    
    # Parse command line arguments
    ACTION=${1:-deploy}
    
    case $ACTION in
        deploy)
            log_info "Starting full staging deployment"
            check_prerequisites
            cleanup_staging
            build_staging_image
            deploy_staging
            run_staging_tests
            show_staging_status
            ;;
        cleanup)
            log_info "Cleaning up staging environment"
            cleanup_staging
            log_success "Staging cleanup completed"
            ;;
        logs)
            log_info "Showing staging logs"
            show_logs
            ;;
        status)
            log_info "Showing staging status"
            show_staging_status
            ;;
        test)
            log_info "Running staging tests only"
            run_staging_tests
            ;;
        *)
            log_error "Invalid action. Use: deploy, cleanup, logs, status, or test"
            echo "Usage: $0 [deploy|cleanup|logs|status|test]"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
