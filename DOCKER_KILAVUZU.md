# 🍦 Magnum Quality Control - Docker Deployment Guide

## Hızlı Başlangıç

### 1. Sistem Gereksinimleri
- Docker & Docker Compose
- NVIDIA Docker (GPU kullanımı için)
- En az 4GB RAM
- 10GB disk alanı

### 2. Kurulum

```bash
# Deployment script'i çalıştır
./deploy_docker.sh
```

### 3. Manuel Kurulum

```bash
# 1. Image'ları build et
docker-compose build

# 2. System'i başlat
docker-compose up -d

# 3. Logları kontrol et
docker-compose logs -f magnum-quality-control
```

## 📁 Dizin Yapısı

```
resnet_frame_embedder_test_v1/
├── good_images/          # Referans iyi görüntüler
├── test_images/          # Test edilecek görüntüler
├── seperated_images/     # Anomali tespit edilen görüntüler
├── logs/                 # System logları
├── reports/              # Analiz raporları
├── Dockerfile            # Ana sistem container
├── Dockerfile.web        # Web dashboard container
├── docker-compose.yml    # Orchestration
└── deploy_docker.sh      # Otomatik deployment
```

## 🛠️ Konfigürasyon

### Ana Sistem Parametreleri

`docker-compose.yml` içinde:

```yaml
command: >
  python ice_cream_quality_control_v3.py
    --good_dir good_images
    --test_dir test_images
    --anomaly_folder seperated_images
    --architectures resnet18 resnet50  # Model seçimi
    --tiling_strategy magnum_stick     # Magnum'a özel
    --confidence 0.95                  # Güven eşiği
    --device cuda                      # GPU kullanımı
    --save_reports                     # Rapor kaydetme
```

### GPU Kullanımı

NVIDIA Docker runtime gerekli:

```bash
# NVIDIA Docker kurulumu
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## 🌐 Web Arayüzleri

### Ana Dashboard
- URL: http://localhost:8080
- Anomali sonuçları görüntüleme
- İstatistikler ve raporlar

### Monitoring (Grafana)
- URL: http://localhost:3000
- Kullanıcı: admin
- Şifre: magnum123

## 📊 Kullanım

### 1. Görüntü Ekleme

```bash
# İyi örnekleri good_images/ klasörüne koyun
cp /path/to/good/*.jpg good_images/

# Test görüntülerini test_images/ klasörüne koyun
cp /path/to/test/*.jpg test_images/
```

### 2. Analiz Çalıştırma

System otomatik olarak çalışır. Manuel tetikleme:

```bash
docker-compose restart magnum-quality-control
```

### 3. Sonuçları Görüntüleme

- Anomali görüntüleri: `seperated_images/`
- Loglar: `docker-compose logs magnum-quality-control`
- Web dashboard: http://localhost:8080

## 🔧 Komutlar

### Temel Komutlar

```bash
# System'i başlat
docker-compose up -d

# System'i durdur
docker-compose down

# Logları görüntüle
docker-compose logs -f magnum-quality-control

# System'i yeniden başlat
docker-compose restart

# Image'ları güncelle
docker-compose build --no-cache
docker-compose up -d
```

### Bakım Komutları

```bash
# Tüm container'ları temizle
docker-compose down -v
docker system prune -a

# Disk kullanımını kontrol et
docker system df

# Container içine gir (debug için)
docker-compose exec magnum-quality-control bash
```

## 🐛 Sorun Giderme

### GPU Kullanılamıyor

```bash
# NVIDIA runtime kontrolü
docker info | grep nvidia

# GPU erişimi test et
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

### Container Başlamıyor

```bash
# Detaylı logları kontrol et
docker-compose logs magnum-quality-control

# Image'ı yeniden build et
docker-compose build --no-cache magnum-quality-control
```

### Bellek Sorunu

```bash
# Docker bellek limitini artır
# ~/.docker/daemon.json
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-shm-size": "2g"
}
```

## 📈 Performans

### Optimize Edilmiş Ayarlar

Yüksek performans için `docker-compose.yml` güncellemesi:

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0
  - OMP_NUM_THREADS=4
  - TORCH_HOME=/root/.cache/torch
deploy:
  resources:
    limits:
      memory: 8G
    reservations:
      memory: 4G
```

### Batch Processing

Büyük görüntü setleri için:

```yaml
command: >
  python ice_cream_quality_control_v3.py
    --batch_size 16
    --workers 4
    --cache_embeddings
```

## 🔒 Güvenlik

### Production Ayarları

```yaml
# Hassas bilgileri environment file'dan al
env_file:
  - .env

# Network izolasyonu
networks:
  magnum_net:
    driver: bridge
```

### Backup

```bash
# Önemli verileri yedekle
docker run --rm -v $(pwd):/backup alpine tar czf /backup/magnum_backup.tar.gz seperated_images reports logs
```

## 📞 Destek

Sorun yaşadığınızda:

1. Logları kontrol edin: `docker-compose logs`
2. System durumunu kontrol edin: `docker-compose ps`
3. Resource kullanımını kontrol edin: `docker stats`

---

**Not:** Bu sistem Magnum çubuklu dondurmaların kalite kontrolü için özelleştirilmiştir. Folyo paketleme sorunlarını tespit etmek için gelişmiş görüntü işleme algoritmaları kullanır.
