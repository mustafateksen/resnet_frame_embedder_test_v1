# Magnum Ice Cream Quality Control System
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Maintainer bilgisi
LABEL maintainer="mustafasabanteksen"
LABEL description="Magnum Ice Cream Quality Control System with ResNet"
LABEL version="3.0"

# Çalışma dizinini ayarla
WORKDIR /app

# Sistem paketlerini güncelle ve gerekli paketleri yükle
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python paketlerini yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Proje dosyalarını kopyala
COPY . .

# Gerekli klasörleri oluştur
RUN mkdir -p good_images test_images seperated_images logs reports

# Script'leri çalıştırılabilir yap
RUN chmod +x *.sh

# Port expose et (web arayüzü için)
EXPOSE 8000

# Varsayılan kullanıcı oluştur (güvenlik için)
RUN useradd -m -s /bin/bash magnum
RUN chown -R magnum:magnum /app
USER magnum

# Sağlık kontrolü
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch, cv2, numpy; print('Health check passed')" || exit 1

# Varsayılan komut
CMD ["python", "ice_cream_quality_control_v3.py", "--help"]
