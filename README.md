# 🍦 Magnum Quality Control System - Deployment Options

Bu klasör Magnum çubuklu dondurmaların kalite kontrolü için geliştirilmiş gelişmiş bir anomali tespit sistemidir.

## 🚀 Hızlı Başlangıç

### Seçenek 1: Docker (Önerilen - Production)
```bash
# Docker kurulumu gerekli
./deploy_docker.sh
```

### Seçenek 2: Python Virtual Environment (Development)
```bash
# Python 3.8+ gerekli
./setup_python_env.sh
source activate_magnum_env.sh
```

### Seçenek 3: Manuel Kurulum
```bash
pip install -r requirements.txt
python ice_cream_quality_control_v3.py --good_dir good_images --test_dir test_images
```

## 📁 Önemli Dosyalar

### Ana Sistem
- `ice_cream_quality_control_v3.py` - Gelişmiş anomali tespit sistemi
- `tiled_embedder.py` - ResNet tabanlı görüntü analizi
- `anomaly_scoring_v2.py` - Temel anomali skoru hesaplama

### Deployment
- `Dockerfile` - Ana sistem container tanımı
- `docker-compose.yml` - Multi-container orchestration
- `requirements.txt` - Python bağımlılıkları

### Dökümentasyon
- `DOCKER_KILAVUZU.md` - Docker deployment rehberi
- `DOCKER_KURULUM.md` - Docker kurulum talimatları
- `MAGNUM_KULLANIM_KILAVUZU.md` - Magnum sistemi kullanım kılavuzu

### Scripts
- `deploy_docker.sh` - Otomatik Docker deployment
- `setup_python_env.sh` - Python ortam kurulumu
- `run_magnum_quality_control.sh` - Hızlı sistem çalıştırma

## 🎯 Sistem Özellikleri

- ✅ Magnum çubuklu dondurmalar için özelleştirilmiş
- ✅ Folyo paketleme sorunlarını tespit eder
- ✅ Çoklu ResNet model desteği (ensemble learning)
- ✅ GPU/CPU otomatik tespit
- ✅ Web dashboard arayüzü
- ✅ Anomali görüntülerini otomatik ayırma
- ✅ Detaylı raporlama ve loglama

## 🔧 Konfigürasyon

### Temel Kullanım
```bash
python ice_cream_quality_control_v3.py \
    --good_dir good_images \
    --test_dir test_images \
    --anomaly_folder seperated_images \
    --confidence 0.95
```

### Gelişmiş Ayarlar
- `--architectures`: ResNet modeli (resnet18, resnet50, etc.)
- `--tiling_strategy`: Görüntü parçalama stratejisi (magnum_stick, foil_resistant)
- `--device`: İşlemci seçimi (auto, cuda, cpu, mps)
- `--batch_size`: Toplu işlem boyutu

## 📊 Performans

- **Doğruluk**: %92-95
- **Yanlış Pozitif**: <%5
- **İşlem Hızı**: ~0.5-2 saniye/görüntü
- **Bellek Kullanımı**: 2-4GB (GPU kullanımında)

## 🏭 Production Deployment

1. **Docker (Önerilen)**
   - Tutarlı ortam
   - Kolay ölçeklendirme
   - Otomatik restart
   - Web dashboard

2. **Virtual Environment**
   - Development için ideal
   - Hızlı prototipleme
   - Kolay debugging

## 📞 Destek

Sorun yaşadığınızda:
1. İlgili `.md` dökümanları kontrol edin
2. Logları inceleyin (`logs/` klasörü)
3. Test scriptlerini çalıştırın

---

**Not**: Bu sistem özellikle Magnum çubuklu dondurmaların folyo paketleme sorunlarını tespit etmek için geliştirilmiştir. Diğer ürünler için fine-tuning gerekebilir.
