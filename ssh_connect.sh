#!/bin/bash
# SSH Bağlantı Arayüzü - Magnum Compute Unit

clear
echo "🍦 MAGNUM COMPUTE UNIT SSH BAĞLANTISI"
echo "====================================="
echo ""

# Bağlantı bilgileri
REMOTE_IP="192.168.0.21"
REMOTE_USER="mustafasabanteksen"
REMOTE_DIR="~/magnum_quality_control"

echo "🔗 Hedef: $REMOTE_USER@$REMOTE_IP"
echo "📁 Klasör: $REMOTE_DIR"
echo ""

# Bağlantı seçenekleri
echo "Seçenekler:"
echo "1️⃣  SSH ile bağlan (Terminal)"
echo "2️⃣  Dosya transfer et (SCP)"
echo "3️⃣  Proje senkronize et (RSYNC)"
echo "4️⃣  Hızlı kalite kontrol çalıştır"
echo "5️⃣  SSH bağlantısını test et"
echo "0️⃣  Çıkış"
echo ""

read -p "Seçiminizi yapın (1-5): " choice

case $choice in
    1)
        echo "🔗 SSH bağlantısı kuruluyor..."
        ssh -t $REMOTE_USER@$REMOTE_IP "
            cd $REMOTE_DIR 2>/dev/null || cd ~
            echo '🍦 Magnum Compute Unit - Hoş Geldiniz!'
            echo '===================================='
            echo 'Mevcut konum:' \$(pwd)
            echo 'Tarih:' \$(date)
            echo 'Sistem:' \$(uname -a)
            echo ''
            echo '📋 Kullanılabilir komutlar:'
            echo '  ls -la                    # Dosyaları listele'
            echo '  cd magnum_quality_control # Proje klasörüne git'
            echo '  bash quick_test.sh        # Hızlı test'
            echo '  python ice_cream_quality_control_v3.py  # Tam sistem'
            echo '  exit                      # SSH çıkışı'
            echo ''
            bash
        "
        ;;
    2)
        echo "📦 Dosya transferi için komutlar:"
        echo ""
        echo "Yerel -> Uzak (görseller):"
        echo "scp -r local_images/* $REMOTE_USER@$REMOTE_IP:$REMOTE_DIR/good_images/"
        echo ""
        echo "Uzak -> Yerel (sonuçlar):"
        echo "scp -r $REMOTE_USER@$REMOTE_IP:$REMOTE_DIR/seperated_images/ ./results/"
        echo ""
        read -p "Enter tuşuna basın..."
        ;;
    3)
        echo "🔄 Proje senkronizasyonu başlatılıyor..."
        rsync -avz --progress \
            --exclude="*.pyc" \
            --exclude="__pycache__" \
            --exclude=".DS_Store" \
            /Users/mustafasabanteksen/Desktop/research/resnet_frame_embedder_test_v1/ \
            $REMOTE_USER@$REMOTE_IP:$REMOTE_DIR/
        echo "✅ Senkronizasyon tamamlandı!"
        read -p "Enter tuşuna basın..."
        ;;
    4)
        echo "⚡ Hızlı kalite kontrol çalıştırılıyor..."
        ssh -t $REMOTE_USER@$REMOTE_IP "
            cd $REMOTE_DIR
            echo '🍦 UZAKTAN HIZLI KALİTE KONTROL'
            echo '==============================='
            echo ''
            
            # Klasör kontrolü
            good_count=\$(find good_images -type f 2>/dev/null | wc -l)
            test_count=\$(find test_images -type f 2>/dev/null | wc -l)
            
            echo \"📊 Mevcut görseller:\"
            echo \"   ✅ İyi görseller: \$good_count adet\"
            echo \"   🔍 Test görselleri: \$test_count adet\"
            echo ''
            
            if [ \$good_count -gt 0 ] && [ \$test_count -gt 0 ]; then
                echo '🚀 Kalite kontrol başlatılıyor...'
                bash quick_test.sh
            else
                echo '⚠️  Görseller eksik! Önce görselleri transfer edin.'
                echo '   Kullanın: bash ssh_connect.sh -> Seçenek 2'
            fi
            
            read -p 'Enter tuşuna basın...'
        "
        ;;
    5)
        echo "🔍 SSH bağlantısı test ediliyor..."
        if ssh -o ConnectTimeout=5 $REMOTE_USER@$REMOTE_IP "echo 'Bağlantı başarılı!'" 2>/dev/null; then
            echo "✅ SSH bağlantısı çalışıyor!"
            ssh $REMOTE_USER@$REMOTE_IP "
                echo '📊 Sistem Bilgileri:'
                echo '   Hostname:' \$(hostname)
                echo '   Uptime:' \$(uptime)
                echo '   Disk:' \$(df -h ~ | tail -1)
                echo '   Python:' \$(python3 --version 2>/dev/null || echo 'Python yok')
            "
        else
            echo "❌ SSH bağlantısı başarısız!"
            echo "   Kontrol edin:"
            echo "   - IP doğru mu? ($REMOTE_IP)"
            echo "   - SSH server çalışıyor mu?"
            echo "   - WiFi ağı aynı mı?"
        fi
        read -p "Enter tuşuna basın..."
        ;;
    0)
        echo "👋 Görüşmek üzere!"
        exit 0
        ;;
    *)
        echo "❌ Geçersiz seçim!"
        read -p "Enter tuşuna basın..."
        ;;
esac

echo ""
echo "🔄 Ana menüye dönmek için scripti yeniden çalıştırın:"
echo "   bash ssh_connect.sh"
