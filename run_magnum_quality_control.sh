#!/bin/bash
# Magnum Çubuklu Dondurma Kalite Kontrol Sistemi
# Paketleme folyosu interference'ına karşı optimize edilmiş

echo "🍦 MAGNUM DONDURMA KALİTE KONTROL SİSTEMİ 🍦"
echo "=================================================="

# Klasörlerin var olup olmadığını kontrol et
if [ ! -d "good_images" ]; then
    echo "❌ HATA: 'good_images' klasörü bulunamadı!"
    echo "   Lütfen kaliteli Magnum görselleri bu klasöre koyun."
    exit 1
fi

if [ ! -d "test_images" ]; then
    echo "❌ HATA: 'test_images' klasörü bulunamadı!"
    echo "   Lütfen test edilecek karışık görselleri bu klasöre koyun."
    exit 1
fi

# Sonuç klasörünü oluştur
mkdir -p seperated_images

echo "✅ Klasörler kontrol edildi!"
echo ""

# Temel Magnum kontrolü (önerilen - varsayılan ayarlar)
echo "🎯 Temel Magnum Kalite Kontrolü Başlatılıyor..."
echo "   📁 İyi görseller: good_images/"
echo "   � Test görselleri: test_images/"
echo "   📁 Sonuç klasörü: seperated_images/"
echo ""

python ice_cream_quality_control_v3.py \
    --good_dir "good_images" \
    --test_dir "test_images" \
    --anomaly_folder "seperated_images" \
    --architectures resnet18 resnet50 \
    --tiling_strategy magnum_stick \
    --ensemble_method weighted_avg \
    --confidence 0.95 \
    --use_adaptive_roi \
    --save_reports \
    --device cuda

echo ""
echo "✅ Magnum kalite kontrolü tamamlandı!"
echo "📋 Sonuçlar 'seperated_images' klasöründe!"
echo ""

# Dosya sayılarını göster
good_count=$(find good_images -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" \) | wc -l)
test_count=$(find test_images -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" \) | wc -l)
anomaly_count=$(find seperated_images -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" \) | wc -l)

echo "📊 İşlem Özeti:"
echo "   ✅ İyi görseller: $good_count adet"
echo "   🔍 Test görselleri: $test_count adet"
echo "   🚨 Tespit edilen anomaliler: $anomaly_count adet"
echo ""
echo "🎉 MAGNUM KALİTE KONTROL İŞLEMİ TAMAMLANDI!"
echo "=================================================="
