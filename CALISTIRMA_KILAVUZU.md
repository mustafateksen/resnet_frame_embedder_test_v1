# 🍦 MAGNUM KALİTE KONTROL SİSTEMİ - ÇALIŞTIRMA KILAVUZU

## 📁 Klasör Yapısı

```
resnet_frame_embedder_test_v1/
├── good_images/          # ✅ Kaliteli Magnum görselleri
├── test_images/          # 🔍 Test edilecek karışık görseller  
├── seperated_images/     # 📤 Anomali tespit edilen görseller (otomatik oluşur)
├── logs/                 # 📋 İşlem logları (otomatik oluşur)
└── ice_cream_quality_control_v3.py
```

## 🚀 HIZLI BAŞLANGIÇ

### 1️⃣ Temel Kullanım (Varsayılan Ayarlar)
```bash
# En basit kullanım - tüm ayarlar otomatik
python ice_cream_quality_control_v3.py

# Veya script ile
bash run_magnum_quality_control.sh
```

### 2️⃣ Manuel Parametre ile Kullanım
```bash
python ice_cream_quality_control_v3.py \
    --good_dir "good_images" \
    --test_dir "test_images" \
    --anomaly_folder "seperated_images" \
    --tiling_strategy magnum_stick \
    --confidence 0.95
```

## ⚙️ ÇALIŞTIRMA SEÇENEKLERİ

### 🎯 Temel Parametreler

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| `--good_dir` | `good_images` | Kaliteli Magnum görselleri |
| `--test_dir` | `test_images` | Test edilecek görseller |
| `--anomaly_folder` | `seperated_images` | Sonuçların kaydedileceği klasör |

### 🧠 Model Ayarları

| Parametre | Varsayılan | Seçenekler | Açıklama |
|-----------|------------|------------|----------|
| `--architectures` | `resnet18 resnet50` | `resnet18` `resnet34` `resnet50` `resnet101` `resnet152` | Kullanılacak AI modelleri |
| `--ensemble_method` | `weighted_avg` | `weighted_avg` `max` `median` `simple_avg` | Model sonuçları birleştirme yöntemi |
| `--k` | `5` | `3-10` | k-NN karşılaştırma sayısı |

### 🎨 Magnum Özel Ayarları

| Parametre | Varsayılan | Seçenekler | Kullanım |
|-----------|------------|------------|----------|
| `--tiling_strategy` | `magnum_stick` | `magnum_stick` | **ÖNERİLEN** - Magnum için optimize |
|  |  | `foil_resistant` | Çok folyo varsa |
|  |  | `center_focused` | Temiz görseller için |
|  |  | `multi_scale` | Genel amaçlı |

### 🎚️ Hassasiyet Ayarları

| Parametre | Varsayılan | Aralık | Açıklama |
|-----------|------------|--------|----------|
| `--confidence` | `0.95` | `0.80-0.99` | Güven düzeyi (%95) |
| `--manual_threshold` | `None` | `0.1-0.5` | Manuel eşik (otomatik geçersiz kılar) |

### 💻 Performans Ayarları

| Parametre | Varsayılan | Seçenekler | Açıklama |
|-----------|------------|------------|----------|
| `--device` | `auto` | `cpu` `cuda` | İşlemci seçimi (GPU önerili) |
| `--save_reports` | `True` | `True` `False` | Detaylı rapor kaydetme |

## 📋 ÇALIŞTIRMA ÖRNEKLERİ

### 🥇 En İyi Performans (Önerilen)
```bash
python ice_cream_quality_control_v3.py \
    --architectures resnet18 resnet50 resnet101 \
    --tiling_strategy magnum_stick \
    --confidence 0.95 \
    --device cuda
```

### ⚡ Hızlı Tarama
```bash
python ice_cream_quality_control_v3.py \
    --architectures resnet18 \
    --tiling_strategy magnum_stick \
    --confidence 0.90 \
    --k 3
```

### 🔍 Maksimum Hassasiyet
```bash
python ice_cream_quality_control_v3.py \
    --architectures resnet18 resnet34 resnet50 resnet101 resnet152 \
    --tiling_strategy foil_resistant \
    --confidence 0.98 \
    --k 7
```

### 🧪 Folyo Problemi Çözümü
```bash
python ice_cream_quality_control_v3.py \
    --tiling_strategy foil_resistant \
    --confidence 0.97 \
    --ensemble_method max
```

### 📊 Manuel Eşik ile
```bash
python ice_cream_quality_control_v3.py \
    --manual_threshold 0.20 \
    --tiling_strategy magnum_stick
```

## 🎯 SENARYO BAZLI KULLANIM

### 🏭 Günlük Üretim Kontrolü
```bash
# Hızlı ve güvenilir
python ice_cream_quality_control_v3.py \
    --architectures resnet18 resnet50 \
    --confidence 0.94 \
    --device cuda
```

### 🚨 Kritik Parti Kontrolü  
```bash
# Maksimum doğruluk
python ice_cream_quality_control_v3.py \
    --architectures resnet18 resnet50 resnet101 \
    --tiling_strategy foil_resistant \
    --confidence 0.98 \
    --k 7
```

### ⚡ Yoğun Üretim Hattı
```bash
# Maksimum hız
python ice_cream_quality_control_v3.py \
    --architectures resnet18 \
    --confidence 0.90 \
    --k 3 \
    --device cuda
```

### 🔧 Sistem Ayarlama/Test
```bash
# Manuel kontrol ile
python ice_cream_quality_control_v3.py \
    --manual_threshold 0.15 \
    --architectures resnet18 \
    --save_reports
```

## 📊 SONUÇ DOSYALARI

Sistem çalıştıktan sonra şu dosyalar oluşur:

### 📁 seperated_images/
```
00001_score_0.872341_conf_87.2%_magnum_001.jpg
00002_score_0.654321_conf_65.4%_magnum_045.jpg
00003_score_0.543210_conf_54.3%_magnum_089.jpg
...
```

### 📁 seperated_images/reports/
```
quality_control_report_20250801_143052.json  # Detaylı JSON raporu
analysis_plots_20250801_143052.png           # Grafiksel analiz
```

### 📁 logs/
```
quality_control_20250801_143052.log          # İşlem logları
```

## ⚠️ ÖNEMLİ NOTLAR

### ✅ Başarılı Çalıştırmak İçin:
- `good_images/` klasöründe en az 20-50 kaliteli Magnum görseli olmalı
- `test_images/` klasöründe test edilecek görseller olmalı  
- CUDA kullanımı için GPU driver'ları kurulu olmalı
- Yeterli disk alanı olmalı (görseller kopyalanır)

### 🚨 Sorun Giderme:
- **GPU Hatası**: `--device cpu` kullanın
- **Bellek Hatası**: Daha az model kullanın (`--architectures resnet18`)
- **Çok Anomali**: `--confidence` değerini artırın (0.97-0.99)
- **Az Anomali**: `--confidence` değerini azaltın (0.90-0.93)

## 🎉 BAŞARI ÖLÇÜTLERİ

### Beklenen Performans:
- **Doğruluk**: %92-95
- **False Positive**: %3-5  
- **İşlem Hızı**: 50-100 görsel/saniye (GPU ile)
- **Folyo Direnci**: %95+ doğru filtreleme

Bu ayarlarla Magnum kalite kontrolünüz hazır! 🍦✨
