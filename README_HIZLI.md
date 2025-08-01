# 🍦 MAGNUM KALİTE KONTROL - HIZLI BAŞLANGIÇ

## 📁 1. KLASÖR HAZIRLIĞI

```bash
cd resnet_frame_embedder_test_v1/

# Klasörler otomatik oluşacak, görselleri yerleştirin:
good_images/     # ✅ Kaliteli Magnum görselleri (10-50 adet)
test_images/     # 🔍 Test edilecek karışık görseller
```

## 🚀 2. ÇALIŞTIRMA SEÇENEKLERİ

### ⚡ En Hızlı Test
```bash
bash quick_test.sh
```

### 🎯 Önerilen Kullanım  
```bash
bash run_magnum_quality_control.sh
```

### 🔧 Manuel Kontrol
```bash
python ice_cream_quality_control_v3.py
```

## 📊 3. SONUÇLAR

Sistem çalıştıktan sonra:
- **seperated_images/**: Anomali tespit edilen görseller
- **logs/**: İşlem günlükleri  
- **reports/**: Detaylı analiz raporları

## ⚙️ 4. YAYGIN AYARLAR

| Kullanım | Komut |
|----------|-------|
| **Hızlı** | `--architectures resnet18 --confidence 0.90` |
| **Dengeli** | `--architectures resnet18 resnet50 --confidence 0.95` |
| **Hassas** | `--architectures resnet18 resnet50 resnet101 --confidence 0.98` |
| **Folyo Sorunu** | `--tiling_strategy foil_resistant --confidence 0.97` |

## 📋 5. SORUN GİDERME

| Sorun | Çözüm |
|-------|-------|
| GPU hatası | `--device cpu` ekleyin |
| Çok anomali | `--confidence 0.97` yapın |
| Az anomali | `--confidence 0.90` yapın |
| Bellek hatası | `--architectures resnet18` kullanın |

**🎯 En çok önerilen**: `bash run_magnum_quality_control.sh`
