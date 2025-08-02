#!/bin/bash

# SSL Setup Script for F1 Prediction System
# This script helps set up Let's Encrypt SSL certificates

set -e

# Configuration
DOMAIN="f1capstone.com"
EMAIL="ssica@berkeley.edu"
STAGING=0  # Set to 0 for production certificates

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   print_error "This script must be run as root or with sudo"
   exit 1
fi

# Get domain name
if [ -z "$1" ]; then
    read -p "Enter your domain name (e.g., example.com): " DOMAIN
else
    DOMAIN=$1
fi

# Get email
if [ -z "$2" ]; then
    read -p "Enter your email for Let's Encrypt notifications: " EMAIL
else
    EMAIL=$2
fi

# Validate inputs
if [ -z "$DOMAIN" ] || [ -z "$EMAIL" ]; then
    print_error "Domain and email are required"
    exit 1
fi

print_info "Setting up SSL for domain: $DOMAIN"

# Update nginx config with actual domain
print_info "Updating nginx configuration..."
sed -i "s/your-domain.com/$DOMAIN/g" nginx/nginx.conf

# Create required directories
print_info "Creating certificate directories..."
mkdir -p certbot/conf
mkdir -p certbot/www

# Check if certificates already exist
if [ -d "certbot/conf/live/$DOMAIN" ]; then
    print_warn "Certificates already exist for $DOMAIN"
    read -p "Do you want to renew them? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Keeping existing certificates"
        exit 0
    fi
fi

# Check if this is initial setup or renewal
if [ -d "certbot/conf/live/$DOMAIN" ]; then
    print_info "Renewing existing certificates - using production nginx..."
    # For renewals, production nginx is already running with ACME challenge support
else
    print_info "Initial certificate generation - starting bootstrap nginx..."
    # For initial setup, start bootstrap nginx that doesn't require SSL certs
    docker compose -f compose.bootstrap.yaml up -d nginx_bootstrap
    sleep 5
fi

# Determine staging flag
STAGING_FLAG=""
if [ $STAGING -eq 1 ]; then
    STAGING_FLAG="--staging"
    print_warn "Using Let's Encrypt staging environment (for testing)"
else
    print_info "Using Let's Encrypt production environment"
fi

# Request certificate
print_info "Requesting SSL certificate from Let's Encrypt..."
docker run --rm \
    -v "$(pwd)/certbot/conf:/etc/letsencrypt" \
    -v "$(pwd)/certbot/www:/var/www/certbot" \
    certbot/certbot certonly \
    --webroot \
    --webroot-path=/var/www/certbot \
    --email $EMAIL \
    --agree-tos \
    --no-eff-email \
    --force-renewal \
    $STAGING_FLAG \
    -d $DOMAIN \
    -d www.$DOMAIN

# Check if certificate was created successfully
if [ $? -eq 0 ]; then
    print_info "SSL certificate created successfully!"
    
    # Create a temporary self-signed certificate if using staging
    if [ $STAGING -eq 1 ]; then
        print_warn "Creating temporary self-signed certificate for testing..."
        mkdir -p certbot/conf/live/$DOMAIN
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout certbot/conf/live/$DOMAIN/privkey.pem \
            -out certbot/conf/live/$DOMAIN/fullchain.pem \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=$DOMAIN"
    fi
    
    # Stop bootstrap nginx if it was used for initial setup
    if [ ! -f "certbot/conf/live/$DOMAIN/fullchain.pem.bak" ]; then
        print_info "Stopping bootstrap nginx and starting production services..."
        docker compose -f compose.bootstrap.yaml down 2>/dev/null || true
        docker compose -f compose.prod.yaml up -d
    else
        print_info "Restarting nginx to use renewed certificates..."
        docker compose -f compose.prod.yaml restart nginx
    fi
    
    print_info "SSL setup complete!"
    print_info "Your application should now be accessible at:"
    print_info "  https://$DOMAIN"
    print_info "  https://www.$DOMAIN"
    
    if [ $STAGING -eq 1 ]; then
        print_warn "You're using staging certificates. To get production certificates:"
        print_warn "  1. Edit this script and set STAGING=0"
        print_warn "  2. Run the script again"
    fi
else
    print_error "Failed to obtain SSL certificate"
    print_error "Please check your domain DNS settings and try again"
    # Clean up bootstrap nginx if it was started
    docker compose -f compose.bootstrap.yaml down 2>/dev/null || true
    exit 1
fi

# Set up automatic renewal
print_info "Setting up automatic certificate renewal..."
(crontab -l 2>/dev/null; echo "0 0 * * * cd $(pwd) && docker compose -f compose.prod.yaml exec certbot certbot renew") | crontab -

print_info "Setup complete! Automatic renewal has been configured via cron."