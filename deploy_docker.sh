#!/bin/bash

# Docker Build ve Deployment Script
# Magnum Quality Control System

set -e

echo "🍦 Magnum Quality Control Docker Deployment"
echo "=========================================="

# Renk kodları
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Gerekli dizinleri kontrol et
echo -e "${BLUE}📁 Checking required directories...${NC}"
required_dirs=("good_images" "test_images" "seperated_images")

for dir in "${required_dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        echo -e "${YELLOW}⚠️  Creating missing directory: $dir${NC}"
        mkdir -p "$dir"
    else
        echo -e "${GREEN}✅ Directory exists: $dir${NC}"
    fi
done

# Reports ve logs dizinlerini oluştur
mkdir -p logs reports

# NVIDIA Docker runtime kontrolü
echo -e "${BLUE}🔧 Checking NVIDIA Docker support...${NC}"
if docker info | grep -q nvidia; then
    echo -e "${GREEN}✅ NVIDIA Docker runtime detected${NC}"
else
    echo -e "${YELLOW}⚠️  NVIDIA Docker runtime not detected. GPU acceleration may not work.${NC}"
fi

# Docker image build
echo -e "${BLUE}🏗️  Building Docker images...${NC}"
docker-compose build --no-cache

# Önceki container'ları temizle
echo -e "${BLUE}🧹 Cleaning up previous containers...${NC}"
docker-compose down -v 2>/dev/null || true

# System'i başlat
echo -e "${BLUE}🚀 Starting Magnum Quality Control System...${NC}"
docker-compose up -d

# Container durumlarını kontrol et
echo -e "${BLUE}📊 Container status:${NC}"
docker-compose ps

# Logları göster
echo -e "${BLUE}📝 System logs:${NC}"
echo -e "${YELLOW}Main system logs:${NC}"
docker-compose logs --tail=20 magnum-quality-control

# Web dashboard kontrolü
if docker-compose ps | grep -q magnum-dashboard; then
    echo -e "${GREEN}🌐 Web Dashboard available at: http://localhost:8080${NC}"
fi

# Monitoring kontrolü
if docker-compose ps | grep -q magnum-monitor; then
    echo -e "${GREEN}📈 Monitoring available at: http://localhost:3000${NC}"
    echo -e "${YELLOW}   Default credentials: admin/magnum123${NC}"
fi

echo -e "${GREEN}✅ Deployment completed!${NC}"
echo ""
echo -e "${BLUE}📋 Quick Commands:${NC}"
echo "  View logs:      docker-compose logs -f magnum-quality-control"
echo "  Stop system:    docker-compose down"
echo "  Restart:        docker-compose restart"
echo "  Update:         docker-compose build --no-cache && docker-compose up -d"
