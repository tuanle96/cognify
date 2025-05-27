#!/bin/bash

# Start Development Environment Script for Cognify
# This script starts Docker Compose and prepares the development environment
# Date: 2025-01-26

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}🚀 Cognify Development Environment Startup${NC}"
echo -e "${PURPLE}=========================================${NC}"
echo ""

# Check if Docker is running
echo -e "${YELLOW}🔍 Checking Docker status...${NC}"
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker is running${NC}"

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}⚠️  docker-compose not found, trying docker compose...${NC}"
    if ! docker compose version &> /dev/null; then
        echo -e "${RED}❌ Neither docker-compose nor docker compose is available${NC}"
        exit 1
    fi
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi
echo -e "${GREEN}✅ Docker Compose is available${NC}"

# Stop any existing containers
echo -e "${YELLOW}🛑 Stopping existing containers...${NC}"
$DOCKER_COMPOSE down --remove-orphans > /dev/null 2>&1 || true
echo -e "${GREEN}✅ Existing containers stopped${NC}"

# Clean up old volumes if requested
if [ "$1" = "--clean" ]; then
    echo -e "${YELLOW}🧹 Cleaning up volumes...${NC}"
    $DOCKER_COMPOSE down -v > /dev/null 2>&1 || true
    docker volume prune -f > /dev/null 2>&1 || true
    echo -e "${GREEN}✅ Volumes cleaned${NC}"
fi

# Build and start services
echo -e "${YELLOW}🏗️  Building and starting services...${NC}"
echo -e "${BLUE}This may take a few minutes on first run...${NC}"

if $DOCKER_COMPOSE up -d --build; then
    echo -e "${GREEN}✅ Services started successfully${NC}"
else
    echo -e "${RED}❌ Failed to start services${NC}"
    exit 1
fi

# Wait for services to be healthy
echo -e "${YELLOW}⏳ Waiting for services to be healthy...${NC}"

# Function to check service health
check_service_health() {
    local service=$1
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if $DOCKER_COMPOSE ps $service | grep -q "healthy"; then
            echo -e "${GREEN}✅ $service is healthy${NC}"
            return 0
        fi
        
        if [ $attempt -eq 1 ]; then
            echo -e "${YELLOW}   Waiting for $service...${NC}"
        fi
        
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo -e "${RED}❌ $service failed to become healthy${NC}"
    return 1
}

# Check each service
services=("postgres" "qdrant" "redis" "cognify-api")
all_healthy=true

for service in "${services[@]}"; do
    if ! check_service_health $service; then
        all_healthy=false
    fi
done

if [ "$all_healthy" = false ]; then
    echo -e "${RED}❌ Some services failed to start properly${NC}"
    echo -e "${YELLOW}📋 Service status:${NC}"
    $DOCKER_COMPOSE ps
    exit 1
fi

# Show service status
echo ""
echo -e "${BLUE}📊 Service Status:${NC}"
$DOCKER_COMPOSE ps

# Show service URLs
echo ""
echo -e "${BLUE}🌐 Service URLs:${NC}"
echo -e "${GREEN}  🔌 Cognify API:        http://localhost:8001${NC}"
echo -e "${GREEN}  📊 API Documentation:  http://localhost:8001/docs${NC}"
echo -e "${GREEN}  🗄️  PostgreSQL:         localhost:5432${NC}"
echo -e "${GREEN}  🔍 Qdrant:             http://localhost:6333${NC}"
echo -e "${GREEN}  📦 Redis:              localhost:6379${NC}"

# Show admin URLs (if admin profile is used)
if $DOCKER_COMPOSE --profile admin ps | grep -q "Up"; then
    echo ""
    echo -e "${BLUE}🛠️  Admin Tools:${NC}"
    echo -e "${GREEN}  📊 pgAdmin:            http://localhost:5050${NC}"
    echo -e "${GREEN}  🔍 Qdrant Web UI:      http://localhost:3000${NC}"
    echo -e "${GREEN}  📦 Redis Commander:    http://localhost:8081${NC}"
fi

# Test API health
echo ""
echo -e "${YELLOW}🔍 Testing API health...${NC}"
sleep 5  # Give API a moment to fully start

if curl -s http://localhost:8001/health > /dev/null; then
    echo -e "${GREEN}✅ API health check passed${NC}"
    
    # Get API info
    api_info=$(curl -s http://localhost:8001/ | python3 -m json.tool 2>/dev/null || echo "API info not available")
    if [ "$api_info" != "API info not available" ]; then
        echo -e "${BLUE}📋 API Info:${NC}"
        echo "$api_info" | head -10
    fi
else
    echo -e "${RED}❌ API health check failed${NC}"
    echo -e "${YELLOW}📋 API logs:${NC}"
    $DOCKER_COMPOSE logs cognify-api --tail 20
fi

# Show logs command
echo ""
echo -e "${BLUE}📝 Useful Commands:${NC}"
echo -e "${GREEN}  View logs:           $DOCKER_COMPOSE logs -f${NC}"
echo -e "${GREEN}  View API logs:       $DOCKER_COMPOSE logs -f cognify-api${NC}"
echo -e "${GREEN}  Stop services:       $DOCKER_COMPOSE down${NC}"
echo -e "${GREEN}  Restart API:         $DOCKER_COMPOSE restart cognify-api${NC}"
echo -e "${GREEN}  Run tests:           python scripts/test_runners/run_comprehensive_tests.py${NC}"

# Database info
echo ""
echo -e "${BLUE}🗄️  Database Info:${NC}"
echo -e "${GREEN}  Host:     localhost${NC}"
echo -e "${GREEN}  Port:     5432${NC}"
echo -e "${GREEN}  Database: cognify_db${NC}"
echo -e "${GREEN}  User:     cognify${NC}"
echo -e "${GREEN}  Password: cognify_password${NC}"

echo ""
echo -e "${PURPLE}🎉 Development environment is ready!${NC}"
echo -e "${YELLOW}💡 Next steps:${NC}"
echo -e "  1. Run comprehensive tests: ${GREEN}python scripts/test_runners/run_comprehensive_tests.py${NC}"
echo -e "  2. Check API docs: ${GREEN}http://localhost:8001/docs${NC}"
echo -e "  3. Monitor logs: ${GREEN}$DOCKER_COMPOSE logs -f${NC}"
echo ""
echo -e "${GREEN}✨ Happy coding!${NC}"
