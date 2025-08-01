# 🍦 Magnum Quality Control - Docker Kurulum Rehberi

## Docker Kurulumu (macOS)

Docker sisteminizde kurulu değil. Kurulum için:

### 1. Docker Desktop Kurulumu

```bash
# Homebrew ile kurulum (önerilen)
brew install --cask docker

# Manuel kurulum için Docker Desktop indirin:
# https://docs.docker.com/desktop/install/mac-install/
```

### 2. Docker Desktop'ı Başlatın

1. Applications klasöründen Docker Desktop'ı açın
2. Docker daemon'ın başlamasını bekleyin
3. Terminal'de test edin:

```bash
docker --version
docker compose version
```

### 3. NVIDIA Docker (GPU desteği için - isteğe bağlı)

macOS'ta NVIDIA GPU desteği sınırlı. Metal Performance Shaders (MPS) kullanılabilir:

```bash
# PyTorch MPS desteği için sistem güncellemesi gerekebilir
```

## Kurulum Sonrası

Docker kurulumu tamamlandıktan sonra:

```bash
# Magnum sistem klasörüne gidin
cd /Users/mustafasabanteksen/Desktop/research/resnet_frame_embedder_test_v1

# Docker image'larını build edin
docker compose build --no-cache

# Sistemi başlatın
docker compose up -d
```

## CPU-Only Versiyonu (macOS için optimize)

NVIDIA GPU yoksa, `docker-compose.yml` dosyasını düzenleyin:

```yaml
# GPU bölümünü kaldırın veya yorumlayın:
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
#           count: 1
#           capabilities: [gpu]

# Environment'a ekleyin:
environment:
  - PYTHONUNBUFFERED=1
  - TORCH_DEVICE=cpu  # CPU kullanımı için
```

## Alternatif: Python Virtual Environment

Docker kurmak istemiyorsanız, Python virtual environment kullanabilirsiniz:

```bash
# Virtual environment oluştur
python3 -m venv magnum_env

# Aktive et
source magnum_env/bin/activate

# Bağımlılıkları kur
pip install -r requirements.txt

# Sistemi çalıştır
python ice_cream_quality_control_v3.py --good_dir good_images --test_dir test_images --device cpu
```

---

**Docker kurulduktan sonra `DOCKER_KILAVUZU.md` dosyasındaki talimatları takip edebilirsiniz.**
