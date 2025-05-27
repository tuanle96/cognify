#!/bin/bash

# Cognify Production Deployment Script
# This script deploys Cognify to production environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_DIR/.env.production"
BACKUP_DIR="/var/backups/cognify"
LOG_FILE="/var/log/cognify/deployment.log"

# Functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$LOG_FILE"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if running as root or with sudo
    if [[ $EUID -eq 0 ]]; then
        warn "Running as root. Consider using a dedicated user for production."
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
    fi
    
    # Check environment file
    if [[ ! -f "$ENV_FILE" ]]; then
        error "Production environment file not found: $ENV_FILE"
    fi
    
    # Check required environment variables
    source "$ENV_FILE"
    
    if [[ -z "$DB_PASSWORD" ]]; then
        error "DB_PASSWORD not set in environment file"
    fi
    
    if [[ -z "$SECRET_KEY" ]]; then
        error "SECRET_KEY not set in environment file"
    fi
    
    if [[ -z "$OPENAI_API_KEY" ]]; then
        error "OPENAI_API_KEY not set in environment file"
    fi
    
    log "Prerequisites check passed"
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    sudo mkdir -p /var/log/cognify
    sudo mkdir -p /var/lib/cognify/uploads
    sudo mkdir -p /var/lib/cognify/data
    sudo mkdir -p "$BACKUP_DIR"
    
    # Set permissions
    sudo chown -R $USER:$USER /var/log/cognify
    sudo chown -R $USER:$USER /var/lib/cognify
    sudo chown -R $USER:$USER "$BACKUP_DIR"
    
    log "Directories created successfully"
}

# Backup existing data
backup_existing_data() {
    log "Creating backup of existing data..."
    
    BACKUP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_PATH="$BACKUP_DIR/backup_$BACKUP_TIMESTAMP"
    
    mkdir -p "$BACKUP_PATH"
    
    # Backup database if exists
    if docker ps | grep -q cognify-postgres; then
        log "Backing up database..."
        docker exec cognify-postgres-prod pg_dump -U cognify cognify > "$BACKUP_PATH/database.sql"
    fi
    
    # Backup uploads if exists
    if [[ -d "/var/lib/cognify/uploads" ]]; then
        log "Backing up uploads..."
        cp -r /var/lib/cognify/uploads "$BACKUP_PATH/"
    fi
    
    # Backup configuration
    if [[ -f "$ENV_FILE" ]]; then
        cp "$ENV_FILE" "$BACKUP_PATH/"
    fi
    
    log "Backup created at: $BACKUP_PATH"
}

# Build Docker images
build_images() {
    log "Building Docker images..."
    
    cd "$PROJECT_DIR"
    
    # Build production image
    docker build -f Dockerfile.production -t cognify:production \
        --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
        --build-arg VERSION=$(git describe --tags --always) \
        --build-arg VCS_REF=$(git rev-parse HEAD) \
        .
    
    log "Docker images built successfully"
}

# Deploy services
deploy_services() {
    log "Deploying services..."
    
    cd "$PROJECT_DIR"
    
    # Load environment variables
    export $(cat "$ENV_FILE" | grep -v '^#' | xargs)
    
    # Deploy with Docker Compose
    docker-compose -f docker-compose.production.yml up -d
    
    log "Services deployed successfully"
}

# Initialize database
initialize_database() {
    log "Initializing database..."
    
    # Wait for database to be ready
    log "Waiting for database to be ready..."
    sleep 30
    
    # Run database initialization
    cd "$PROJECT_DIR"
    export $(cat "$ENV_FILE" | grep -v '^#' | xargs)
    
    python scripts/production_init.py
    
    log "Database initialized successfully"
}

# Health checks
run_health_checks() {
    log "Running health checks..."
    
    # Check database
    if ! docker exec cognify-postgres-prod pg_isready -U cognify -d cognify; then
        error "Database health check failed"
    fi
    
    # Check Redis
    if ! docker exec cognify-redis-prod redis-cli ping; then
        error "Redis health check failed"
    fi
    
    # Check application
    sleep 10  # Wait for app to start
    if ! curl -f http://localhost:8000/health; then
        error "Application health check failed"
    fi
    
    # Check Qdrant
    if ! curl -f http://localhost:6333/health; then
        error "Qdrant health check failed"
    fi
    
    log "All health checks passed"
}

# Setup monitoring
setup_monitoring() {
    log "Setting up monitoring..."
    
    # Check if Prometheus is running
    if curl -f http://localhost:9090/-/healthy &> /dev/null; then
        log "Prometheus is running"
    else
        warn "Prometheus health check failed"
    fi
    
    # Check if Grafana is running
    if curl -f http://localhost:3000/api/health &> /dev/null; then
        log "Grafana is running"
    else
        warn "Grafana health check failed"
    fi
    
    log "Monitoring setup completed"
}

# Setup SSL certificates (if needed)
setup_ssl() {
    log "Setting up SSL certificates..."
    
    # Check if SSL directory exists
    if [[ ! -d "$PROJECT_DIR/nginx/ssl" ]]; then
        mkdir -p "$PROJECT_DIR/nginx/ssl"
        
        # Generate self-signed certificate for testing
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout "$PROJECT_DIR/nginx/ssl/cognify.key" \
            -out "$PROJECT_DIR/nginx/ssl/cognify.crt" \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
        
        warn "Self-signed SSL certificate generated. Replace with real certificate for production."
    fi
    
    log "SSL setup completed"
}

# Setup log rotation
setup_log_rotation() {
    log "Setting up log rotation..."
    
    # Create logrotate configuration
    sudo tee /etc/logrotate.d/cognify > /dev/null <<EOF
/var/log/cognify/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 $USER $USER
    postrotate
        docker kill -s USR1 cognify-app-prod 2>/dev/null || true
    endscript
}
EOF
    
    log "Log rotation setup completed"
}

# Setup firewall
setup_firewall() {
    log "Setting up firewall..."
    
    # Check if ufw is available
    if command -v ufw &> /dev/null; then
        # Allow SSH
        sudo ufw allow ssh
        
        # Allow HTTP and HTTPS
        sudo ufw allow 80/tcp
        sudo ufw allow 443/tcp
        
        # Allow application port (if needed)
        sudo ufw allow 8000/tcp
        
        # Allow monitoring ports (restrict to local network)
        sudo ufw allow from 10.0.0.0/8 to any port 9090
        sudo ufw allow from 10.0.0.0/8 to any port 3000
        
        # Enable firewall
        sudo ufw --force enable
        
        log "Firewall configured"
    else
        warn "UFW not available. Please configure firewall manually."
    fi
}

# Main deployment function
main() {
    log "Starting Cognify production deployment..."
    
    # Create log file
    sudo mkdir -p "$(dirname "$LOG_FILE")"
    sudo touch "$LOG_FILE"
    sudo chown $USER:$USER "$LOG_FILE"
    
    # Run deployment steps
    check_prerequisites
    create_directories
    backup_existing_data
    build_images
    setup_ssl
    deploy_services
    initialize_database
    run_health_checks
    setup_monitoring
    setup_log_rotation
    setup_firewall
    
    log "Cognify production deployment completed successfully!"
    
    # Display important information
    echo ""
    echo "=========================================="
    echo "🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!"
    echo "=========================================="
    echo ""
    echo "📊 Service URLs:"
    echo "  Application: http://localhost:8000"
    echo "  Grafana:     http://localhost:3000"
    echo "  Prometheus:  http://localhost:9090"
    echo ""
    echo "📁 Important Paths:"
    echo "  Logs:        /var/log/cognify/"
    echo "  Data:        /var/lib/cognify/"
    echo "  Backups:     $BACKUP_DIR"
    echo ""
    echo "🔧 Next Steps:"
    echo "  1. Configure your domain and SSL certificates"
    echo "  2. Set up external monitoring and alerting"
    echo "  3. Configure automated backups"
    echo "  4. Review security settings"
    echo ""
    echo "📖 Documentation: https://docs.cognify.ai"
    echo "🆘 Support: support@cognify.ai"
    echo ""
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Cognify Production Deployment Script"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --check        Run prerequisites check only"
        echo "  --backup       Create backup only"
        echo "  --health       Run health checks only"
        echo ""
        exit 0
        ;;
    --check)
        check_prerequisites
        exit 0
        ;;
    --backup)
        backup_existing_data
        exit 0
        ;;
    --health)
        run_health_checks
        exit 0
        ;;
    *)
        main
        ;;
esac
