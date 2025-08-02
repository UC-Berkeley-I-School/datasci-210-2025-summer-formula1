# HTTPS Setup Guide for F1 Prediction System

This guide explains how to set up HTTPS (SSL/TLS) for the F1 Prediction System using Nginx and Let's Encrypt.

## Prerequisites

1. **Domain Name**: You need a registered domain name pointing to your EC2 instance
2. **EC2 Security Group**: Ensure ports 80 and 443 are open for inbound traffic:
   - HTTP (80) from 0.0.0.0/0
   - HTTPS (443) from 0.0.0.0/0
3. **Docker & Docker Compose**: Already installed if you've been running the application

## Architecture

The HTTPS setup uses:
- **Nginx**: Reverse proxy for SSL termination and HTTP to HTTPS redirect
- **Let's Encrypt**: Free SSL certificates with automatic renewal
- **Certbot**: Certificate management tool running as a container

## Quick Setup (Recommended)

This is the simplest way to get HTTPS working:

1. **Point your domain to the EC2 instance**:
   - Add an A record in your DNS provider pointing to your EC2 public IP
   - Example: `f1capstone.com` → `3.82.227.64`
   - Add a CNAME for `www.f1capstone.com` → `f1capstone.com`

2. **Run the automated SSL setup script**:
   ```bash
   cd /home/ubuntu/datasci-210-2025-summer-formula1/deliverables
   sudo ./setup-ssl.sh
   ```
   
   The script will prompt you for:
   - Your domain name (e.g., `f1capstone.com`)
   - Your email for Let's Encrypt notifications

3. **Wait for completion**:
   - The script will automatically obtain SSL certificates
   - All containers will start with HTTPS enabled
   - You'll see confirmation messages when complete

4. **Verify the setup**:
   - Visit `https://your-domain.com`
   - Check that the browser shows a secure connection (🔒)
   - HTTP requests should automatically redirect to HTTPS

## That's it! 

The script handles everything automatically:
- ✅ Nginx configuration
- ✅ SSL certificate generation  
- ✅ Container startup with HTTPS
- ✅ Automatic certificate renewal setup

## Manual Setup

If you prefer manual setup or need to customize:

1. **Update nginx configuration**:
   ```bash
   # Edit nginx/nginx.conf
   # Replace your-domain.com with your actual domain
   vim nginx/nginx.conf
   ```

2. **Create certificate directories**:
   ```bash
   mkdir -p certbot/conf
   mkdir -p certbot/www
   ```

3. **Start services without SSL first**:
   ```bash
   docker compose -f compose.prod.yaml up -d nginx
   ```

4. **Get SSL certificate**:
   ```bash
   docker run --rm \
     -v "$(pwd)/certbot/conf:/etc/letsencrypt" \
     -v "$(pwd)/certbot/www:/var/www/certbot" \
     certbot/certbot certonly \
     --webroot \
     --webroot-path=/var/www/certbot \
     --email your-email@example.com \
     --agree-tos \
     --no-eff-email \
     -d your-domain.com \
     -d www.your-domain.com
   ```

5. **Start all services**:
   ```bash
   docker compose -f compose.prod.yaml down
   docker compose -f compose.prod.yaml up -d
   ```

## Certificate Renewal

Certificates are automatically renewed by the certbot container running 24/7. The cron job checks twice daily.

**Manual renewal** (if needed):
```bash
# Option 1: Use the setup script
sudo ./setup-ssl.sh

# Option 2: Manual certbot command  
docker compose -f compose.prod.yaml exec certbot certbot renew
```

## Troubleshooting

### DNS Issues
```bash
# Check if domain resolves to your server
nslookup your-domain.com
dig your-domain.com
```

### Certificate Issues
```bash
# Check certificate status
docker compose -f compose.prod.yaml run --rm certbot certificates

# View nginx logs
docker compose -f compose.prod.yaml logs nginx

# Test configuration
docker compose -f compose.prod.yaml exec nginx nginx -t
```

### Port Issues
```bash
# Check if ports are open
sudo netstat -tlnp | grep -E ':(80|443)'

# Check EC2 security group
# Ensure inbound rules allow:
# - HTTP (80) from 0.0.0.0/0
# - HTTPS (443) from 0.0.0.0/0
```