#!/bin/bash
# SSH Bağlantısı ve Magnum Kalite Kontrol Transfer Scripti

echo "🍦 MAGNUM KALİTE KONTROL - UZAKTAN ÇALIŞTIRMA"
echo "=============================================="

# Hedef makine bilgileri
REMOTE_IP="192.168.0.21"
REMOTE_USER="mustafasabanteksen"
REMOTE_DIR="~/magnum_quality_control"
LOCAL_DIR="/Users/mustafasabanteksen/Desktop/research/resnet_frame_embedder_test_v1"

echo "🔗 Hedef Makine: $REMOTE_USER@$REMOTE_IP"
echo "📁 Yerel Klasör: $LOCAL_DIR"
echo "📁 Uzak Klasör: $REMOTE_DIR"
echo ""

# SSH bağlantısını test et
echo "🔍 SSH bağlantısı test ediliyor..."
if ssh -o ConnectTimeout=5 $REMOTE_USER@$REMOTE_IP "echo 'SSH bağlantısı başarılı!'" 2>/dev/null; then
    echo "✅ SSH bağlantısı başarılı!"
else
    echo "❌ SSH bağlantısı başarısız!"
    echo "   Kontrol edin:"
    echo "   - IP adresi doğru mu? ($REMOTE_IP)"
    echo "   - SSH server çalışıyor mu?"
    echo "   - Kullanıcı adı doğru mu? ($REMOTE_USER)"
    echo "   - WiFi ağı aynı mı?"
    exit 1
fi

echo ""
echo "📦 Proje dosyaları transfer ediliyor..."

# Uzak klasörü oluştur
ssh $REMOTE_USER@$REMOTE_IP "mkdir -p $REMOTE_DIR"

# RSYNC ile transfer (progress bar ile)
rsync -avz --progress \
    --exclude="*.pyc" \
    --exclude="__pycache__" \
    --exclude=".DS_Store" \
    --exclude="seperated_images" \
    --exclude="logs" \
    $LOCAL_DIR/ $REMOTE_USER@$REMOTE_IP:$REMOTE_DIR/

if [ $? -eq 0 ]; then
    echo "✅ Transfer başarılı!"
else
    echo "❌ Transfer başarısız!"
    exit 1
fi

echo ""
echo "🔧 Uzak makinede kurulum kontrol ediliyor..."

# Python ve gerekli paketleri kontrol et
ssh $REMOTE_USER@$REMOTE_IP "
cd $REMOTE_DIR
echo '📍 Mevcut dizin:' && pwd
echo '📁 Dosyalar:' && ls -la
echo '🐍 Python versiyonu:' && python3 --version
echo '📦 PyTorch kontrol:' && python3 -c 'import torch; print(f\"PyTorch: {torch.__version__}\")' 2>/dev/null || echo 'PyTorch yok!'
echo '🖼️  OpenCV kontrol:' && python3 -c 'import cv2; print(f\"OpenCV: {cv2.__version__}\")' 2>/dev/null || echo 'OpenCV yok!'
"

echo ""
echo "🎯 Uzak makinede Magnum kalite kontrolü başlatılıyor..."
echo ""

# Uzak makinede test çalıştır
ssh -t $REMOTE_USER@$REMOTE_IP "
cd $REMOTE_DIR
echo '🍦 UZAKTAN MAGNUM KALİTE KONTROL TEST'
echo '===================================='
echo ''

# Klasörleri oluştur
mkdir -p good_images test_images seperated_images

# Klasör durumunu kontrol et
echo '📁 Klasör Durumu:'
echo \"   good_images: \$(find good_images -type f 2>/dev/null | wc -l) dosya\"
echo \"   test_images: \$(find test_images -type f 2>/dev/null | wc -l) dosya\"
echo ''

if [ \$(find good_images -type f 2>/dev/null | wc -l) -eq 0 ]; then
    echo '⚠️  good_images klasörü boş! Test görselleri gerekli.'
    echo '   Çözüm: Yerel makineden görselleri transfer edin:'
    echo '   scp -r local_good_images/* $REMOTE_USER@$REMOTE_IP:$REMOTE_DIR/good_images/'
    echo ''
fi

if [ \$(find test_images -type f 2>/dev/null | wc -l) -eq 0 ]; then
    echo '⚠️  test_images klasörü boş! Test edilecek görseller gerekli.'
    echo '   Çözüm: Yerel makineden görselleri transfer edin:'
    echo '   scp -r local_test_images/* $REMOTE_USER@$REMOTE_IP:$REMOTE_DIR/test_images/'
    echo ''
fi

echo '🔧 Sistem hazırlığı tamamlandı!'
echo 'Kalite kontrolü için:'
echo '  bash quick_test.sh            # Hızlı test'  
echo '  python ice_cream_quality_control_v3.py  # Tam sistem'
echo ''
echo '📊 SSH bağlantısı açık kalıyor...'
"

echo ""
echo "🎉 UZAKTAN BAĞLANTI KURULUMU TAMAMLANDI!"
echo ""
echo "📋 Sonraki Adımlar:"
echo "1. Görselleri transfer edin:"
echo "   scp -r local_good_images/* $REMOTE_USER@$REMOTE_IP:$REMOTE_DIR/good_images/"
echo "   scp -r local_test_images/* $REMOTE_USER@$REMOTE_IP:$REMOTE_DIR/test_images/"
echo ""
echo "2. Uzak makineye bağlanın:"
echo "   ssh $REMOTE_USER@$REMOTE_IP"
echo "   cd $REMOTE_DIR"
echo ""
echo "3. Kalite kontrolünü çalıştırın:"
echo "   bash quick_test.sh"
echo ""
