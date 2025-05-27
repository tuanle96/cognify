#!/bin/bash

# Production Deployment Script for Cognify RAG System
# This script builds and deploys Cognify in production mode with all services

set -e  # Exit on any error

echo "🚀 Starting Cognify Production Deployment"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if .env.production exists
if [ ! -f ".env.production" ]; then
    print_error ".env.production file not found!"
    print_warning "Please create .env.production with your production configuration"
    exit 1
fi

# Copy .env.production to .env for Docker Compose
print_status "Setting up environment configuration..."
cp .env.production .env
print_success "Environment configuration ready"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose > /dev/null 2>&1; then
    print_error "Docker Compose is not installed. Please install Docker Compose and try again."
    exit 1
fi

# Stop any existing containers
print_status "Stopping existing containers..."
docker-compose -f docker-compose.production.yml down --remove-orphans || true
print_success "Existing containers stopped"

# Remove old images (optional - uncomment if you want to force rebuild)
# print_status "Removing old images..."
# docker-compose -f docker-compose.production.yml down --rmi all || true

# Build the application
print_status "Building Cognify application..."
docker-compose -f docker-compose.production.yml build --no-cache
print_success "Application built successfully"

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p logs
mkdir -p infrastructure/docker/ssl
print_success "Directories created"

# Start the services
print_status "Starting production services..."
docker-compose -f docker-compose.production.yml up -d

# Wait for services to be ready
print_status "Waiting for services to start..."
sleep 30

# Check service health
print_status "Checking service health..."

# Check PostgreSQL
if docker-compose -f docker-compose.production.yml exec -T postgres pg_isready -U cognify -d cognify_production > /dev/null 2>&1; then
    print_success "PostgreSQL is ready"
else
    print_warning "PostgreSQL is not ready yet"
fi

# Check Redis
if docker-compose -f docker-compose.production.yml exec -T redis redis-cli ping > /dev/null 2>&1; then
    print_success "Redis is ready"
else
    print_warning "Redis is not ready yet"
fi

# Check Qdrant
if curl -f http://localhost:6333/health > /dev/null 2>&1; then
    print_success "Qdrant is ready"
else
    print_warning "Qdrant is not ready yet"
fi

# Check Cognify API
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    print_success "Cognify API is ready"
else
    print_warning "Cognify API is not ready yet"
fi

# Show running containers
print_status "Running containers:"
docker-compose -f docker-compose.production.yml ps

# Show logs for the last few minutes
print_status "Recent logs:"
docker-compose -f docker-compose.production.yml logs --tail=20

echo ""
echo "🎉 Cognify Production Deployment Complete!"
echo "=========================================="
echo ""
echo "📊 Service URLs:"
echo "  • Cognify API: http://localhost:8000"
echo "  • API Documentation: http://localhost:8000/docs"
echo "  • Health Check: http://localhost:8000/health"
echo "  • Prometheus: http://localhost:9090"
echo "  • Grafana: http://localhost:3000 (admin/cognify_grafana_admin_2025)"
echo "  • Qdrant: http://localhost:6333"
echo ""
echo "🔧 Management Commands:"
echo "  • View logs: docker-compose -f docker-compose.production.yml logs -f"
echo "  • Stop services: docker-compose -f docker-compose.production.yml down"
echo "  • Restart services: docker-compose -f docker-compose.production.yml restart"
echo "  • Update services: ./deploy-production.sh"
echo ""
echo "📝 Next Steps:"
echo "  1. Test the API endpoints"
echo "  2. Configure SSL certificates for HTTPS"
echo "  3. Set up domain name and DNS"
echo "  4. Configure monitoring alerts"
echo "  5. Set up backup procedures"
echo ""
print_success "Deployment completed successfully!"
