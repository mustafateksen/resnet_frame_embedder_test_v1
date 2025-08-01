#!/bin/bash
# Hızlı Magnum Kalite Kontrol Testi

echo "🍦 HIZLI MAGNUM TEST"
echo "==================="

# Klasör kontrolü
echo "📁 Klasör kontrolü..."
if [ ! -d "good_images" ]; then
    echo "❌ 'good_images' klasörü yok! Oluşturuluyor..."
    mkdir -p good_images
    echo "   ⚠️  Lütfen kaliteli Magnum görselleri 'good_images/' klasörüne koyun"
fi

if [ ! -d "test_images" ]; then
    echo "❌ 'test_images' klasörü yok! Oluşturuluyor..."
    mkdir -p test_images  
    echo "   ⚠️  Lütfen test edilecek görselleri 'test_images/' klasörüne koyun"
fi

# Görsel sayısı kontrolü
good_count=$(find good_images -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" \) 2>/dev/null | wc -l)
test_count=$(find test_images -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" \) 2>/dev/null | wc -l)

echo "📊 Bulunan görseller:"
echo "   ✅ İyi görseller: $good_count adet"
echo "   🔍 Test görselleri: $test_count adet"

if [ $good_count -eq 0 ]; then
    echo ""
    echo "❌ HATA: 'good_images' klasöründe görsel yok!"
    echo "   📝 Yapılacaklar:"
    echo "   1. Kaliteli Magnum görselleri 'good_images/' klasörüne koyun"
    echo "   2. En az 10-20 farklı kaliteli görsel olmalı"
    echo "   3. JPG, PNG, BMP formatları desteklenir"
    exit 1
fi

if [ $test_count -eq 0 ]; then
    echo ""
    echo "❌ HATA: 'test_images' klasöründe görsel yok!"
    echo "   📝 Yapılacaklar:"
    echo "   1. Test edilecek görselleri 'test_images/' klasörüne koyun"
    echo "   2. Hem kaliteli hem sorunlu görseller olabilir"
    echo "   3. JPG, PNG, BMP formatları desteklenir"
    exit 1
fi

echo ""
echo "✅ Klasörler hazır! Hızlı test başlatılıyor..."
echo ""

# Hızlı test parametreleri
echo "🚀 Hızlı Test Parametreleri:"
echo "   🧠 Model: ResNet18 (hızlı)"
echo "   🎯 Tiling: magnum_stick"
echo "   📊 Güven: %90"
echo "   💻 Device: auto"
echo ""

# Test çalıştır
python ice_cream_quality_control_v3.py \
    --good_dir "good_images" \
    --test_dir "test_images" \
    --anomaly_folder "seperated_images" \
    --architectures resnet18 \
    --tiling_strategy magnum_stick \
    --confidence 0.90 \
    --k 3 \
    --save_reports

echo ""
echo "🎉 HIZLI TEST TAMAMLANDI!"
echo ""

# Sonuç özeti
if [ -d "seperated_images" ]; then
    anomaly_count=$(find seperated_images -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" \) 2>/dev/null | wc -l)
    echo "📋 Test Sonucu:"
    echo "   🚨 Tespit edilen anomali: $anomaly_count adet"
    echo "   📁 Sonuçlar: seperated_images/ klasöründe"
    
    if [ $anomaly_count -gt 0 ]; then
        echo ""
        echo "🔍 Tespit edilen anomaliler:"
        ls seperated_images/*.jpg seperated_images/*.jpeg seperated_images/*.png seperated_images/*.bmp 2>/dev/null | head -5 | while read file; do
            if [ -f "$file" ]; then
                filename=$(basename "$file")
                echo "   📸 $filename"
            fi
        done
        
        if [ $anomaly_count -gt 5 ]; then
            echo "   ... ve $(($anomaly_count - 5)) adet daha"
        fi
    else
        echo "   ✅ Hiç anomali tespit edilmedi - tüm görseller kaliteli!"
    fi
fi

echo ""
echo "📚 Daha detaylı analiz için:"
echo "   bash run_magnum_quality_control.sh"
echo ""
echo "🔧 Ayar değiştirmek için:"
echo "   python ice_cream_quality_control_v3.py --help"
