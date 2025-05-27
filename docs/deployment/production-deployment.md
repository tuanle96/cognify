# 🚀 Cognify Production Deployment Guide

## 📋 **Quick Start Checklist**

### ✅ **Prerequisites**
- [ ] Linux server (Ubuntu 20.04+ recommended)
- [ ] Docker & Docker Compose installed
- [ ] 4GB+ RAM, 2+ CPU cores
- [ ] 50GB+ storage space
- [ ] Domain name (optional but recommended)
- [ ] SSL certificate (optional but recommended)

### ✅ **Required API Keys**
- [ ] OpenAI API key
- [ ] Database password
- [ ] JWT secret keys
- [ ] Admin credentials

---

## 🔧 **Step 1: Environment Setup**

### **1.1 Copy Environment File**
```bash
cp .env.production.example .env.production
```

### **1.2 Configure Critical Settings**
Edit `.env.production` and set these **REQUIRED** values:

```bash
# Database (CRITICAL)
DB_PASSWORD=your_very_secure_database_password_here

# Security (CRITICAL)
SECRET_KEY=your_32_character_secret_key_here
JWT_SECRET_KEY=your_32_character_jwt_secret_here

# AI Service (CRITICAL)
OPENAI_API_KEY=sk-your_real_openai_api_key_here

# Admin User (CRITICAL)
ADMIN_EMAIL=admin@yourcompany.com
ADMIN_PASSWORD=your_secure_admin_password_here

# Organization (REQUIRED)
ORG_NAME=Your Company Name
ORG_SLUG=your-company
```

### **1.3 Generate Secure Keys**
```bash
# Generate SECRET_KEY
python -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(32))"

# Generate JWT_SECRET_KEY
python -c "import secrets; print('JWT_SECRET_KEY=' + secrets.token_urlsafe(32))"
```

---

## 🚀 **Step 2: Database Initialization**

### **2.1 Reset Database (Clean Start)**
```bash
# Make scripts executable
chmod +x scripts/*.sh

# Run complete database reset
./scripts/full-reset.sh
```

### **2.2 Manual Database Setup (Alternative)**
```bash
# Run database initialization
psql -h localhost -U postgres -d cognify_db -f scripts/init-complete-db.sql
```

---

## 🚀 **Step 3: One-Command Deployment**

### **3.1 Make Script Executable**
```bash
chmod +x scripts/deploy_production.sh
```

### **3.2 Run Deployment**
```bash
./scripts/deploy_production.sh
```

**This script will:**
- ✅ Check prerequisites
- ✅ Create necessary directories
- ✅ Backup existing data
- ✅ Build Docker images
- ✅ Deploy all services
- ✅ Initialize database with admin user
- ✅ Run health checks
- ✅ Setup monitoring
- ✅ Configure security

---

## 🔍 **Step 4: Verification**

### **4.1 Check Service Status**
```bash
docker-compose -f docker-compose.production.yml ps
```

### **4.2 Test Application**
```bash
# Health check
curl http://localhost:8000/health

# API test
curl http://localhost:8000/api/v1/health
```

### **4.3 Access Admin Panel**
- **URL**: `http://localhost:8000/admin`
- **Email**: Your `ADMIN_EMAIL` from `.env.production`
- **Password**: Your `ADMIN_PASSWORD` from `.env.production`

---

## 📊 **Step 5: Monitoring Setup**

### **5.1 Access Monitoring Dashboards**
- **Grafana**: `http://localhost:3000`
  - Username: `admin`
  - Password: Your `GRAFANA_PASSWORD` from `.env.production`
- **Prometheus**: `http://localhost:9090`

### **5.2 Configure Alerts**
```bash
# Edit Grafana alert rules
docker exec -it cognify-grafana-prod /bin/bash
```

---

## 🔒 **Step 6: Security Hardening**

### **6.1 SSL Certificate Setup**
```bash
# For Let's Encrypt (recommended)
sudo apt install certbot
sudo certbot certonly --standalone -d yourdomain.com

# Copy certificates
sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem nginx/ssl/
sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem nginx/ssl/
```

### **6.2 Firewall Configuration**
```bash
# UFW (Ubuntu)
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### **6.3 Update Nginx Configuration**
Edit `nginx/nginx.conf` to use your domain and SSL certificates.

---

## 🗄️ **Step 7: Database Management**

### **7.1 Database Access**
```bash
# Connect to database
docker exec -it cognify-postgres-prod psql -U cognify -d cognify

# Run SQL commands
\dt  # List tables
\q   # Quit
```

### **7.2 Create Additional Admin Users**
```bash
# Run initialization script
python scripts/production_init.py
```

### **7.3 Database Backup**
```bash
# Manual backup
docker exec cognify-postgres-prod pg_dump -U cognify cognify > backup.sql

# Restore backup
docker exec -i cognify-postgres-prod psql -U cognify -d cognify < backup.sql
```

---

## 🔧 **Step 8: Maintenance**

### **8.1 View Logs**
```bash
# Application logs
docker logs cognify-app-prod -f

# Database logs
docker logs cognify-postgres-prod -f

# All services
docker-compose -f docker-compose.production.yml logs -f
```

### **8.2 Update Application**
```bash
# Pull latest code
git pull origin main

# Rebuild and redeploy
./scripts/deploy_production.sh
```

### **8.3 Scale Services**
```bash
# Scale application instances
docker-compose -f docker-compose.production.yml up -d --scale cognify-app=3
```

---

## 🚨 **Troubleshooting**

### **Common Issues**

#### **Database Connection Failed**
```bash
# Check database status
docker exec cognify-postgres-prod pg_isready -U cognify

# Reset database password
docker exec -it cognify-postgres-prod psql -U postgres
ALTER USER cognify PASSWORD 'new_password';
```

#### **Application Won't Start**
```bash
# Check environment variables
docker exec cognify-app-prod env | grep -E "(DB_|OPENAI_|SECRET_)"

# Check application logs
docker logs cognify-app-prod --tail 100
```

#### **Vector Database Issues**
```bash
# Check Qdrant status
curl http://localhost:6333/health

# Restart Qdrant
docker restart cognify-qdrant-prod
```

---

## 📈 **Performance Optimization**

### **Database Optimization**
```sql
-- Connect to database and run:
ANALYZE;
REINDEX DATABASE cognify;
VACUUM FULL;
```

### **Application Scaling**
```bash
# Increase worker processes
# Edit docker-compose.production.yml:
# command: ["python", "-m", "uvicorn", "app.main:app", "--workers", "8"]
```

### **Caching Optimization**
```bash
# Check Redis memory usage
docker exec cognify-redis-prod redis-cli info memory

# Clear cache if needed
docker exec cognify-redis-prod redis-cli flushall
```

---

## 📞 **Support & Resources**

### **Documentation**
- 📖 [API Documentation](http://localhost:8000/docs)
- 🔧 [Database Setup](database-setup.md)
- 🛠️ [Development Guide](../development/getting-started.md)

### **Monitoring URLs**
- 🖥️ **Application**: `http://localhost:8000`
- 📊 **Grafana**: `http://localhost:3000`
- 📈 **Prometheus**: `http://localhost:9090`
- 🔍 **Qdrant**: `http://localhost:6333/dashboard`

### **Important Files**
- 🔧 **Configuration**: `.env.production`
- 📝 **Logs**: `/var/log/cognify/`
- 💾 **Data**: `/var/lib/cognify/`
- 🗄️ **Backups**: `/var/backups/cognify/`

---

## ✅ **Post-Deployment Checklist**

- [ ] All services running (`docker ps`)
- [ ] Health checks passing (`curl localhost:8000/health`)
- [ ] Admin user can login
- [ ] Database initialized with multi-tenant schema
- [ ] SSL certificates configured
- [ ] Monitoring dashboards accessible
- [ ] Backups configured
- [ ] Firewall rules applied
- [ ] Domain name configured (if applicable)
- [ ] Email notifications setup (if applicable)

**🎉 Congratulations! Your Cognify production environment is ready!**
