#!/bin/bash

# Magnum Quality Control - Python Virtual Environment Setup
# Docker alternatifi için Python sanal ortam kurulumu

set -e

echo "🍦 Magnum Quality Control - Python Environment Setup"
echo "=================================================="

# Renk kodları
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Python versiyonu kontrolü
echo -e "${BLUE}🐍 Checking Python version...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}✅ Python $PYTHON_VERSION found${NC}"
else
    echo -e "${RED}❌ Python3 not found. Please install Python 3.8+${NC}"
    exit 1
fi

# Virtual environment oluştur
ENV_NAME="magnum_env"
echo -e "${BLUE}📦 Creating virtual environment: $ENV_NAME${NC}"

if [ -d "$ENV_NAME" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment already exists. Removing...${NC}"
    rm -rf "$ENV_NAME"
fi

python3 -m venv "$ENV_NAME"
source "$ENV_NAME/bin/activate"

echo -e "${GREEN}✅ Virtual environment created and activated${NC}"

# Pip güncellemesi
echo -e "${BLUE}🔧 Upgrading pip...${NC}"
pip install --upgrade pip wheel setuptools

# Bağımlılıkları kur
echo -e "${BLUE}📥 Installing dependencies...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo -e "${YELLOW}⚠️  requirements.txt not found. Installing core packages...${NC}"
    pip install torch torchvision opencv-python scikit-learn numpy matplotlib pillow tqdm
fi

# Gerekli dizinleri oluştur
echo -e "${BLUE}📁 Creating required directories...${NC}"
required_dirs=("good_images" "test_images" "seperated_images" "logs" "reports")

for dir in "${required_dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo -e "${GREEN}✅ Created directory: $dir${NC}"
    else
        echo -e "${YELLOW}📁 Directory already exists: $dir${NC}"
    fi
done

# Test scriptini oluştur
echo -e "${BLUE}🧪 Creating test script...${NC}"
cat > test_installation.py << 'EOF'
#!/usr/bin/env python3
import sys
import torch
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def test_installation():
    print("🧪 Testing Magnum Quality Control Installation")
    print("=" * 50)
    
    # Python version
    print(f"🐍 Python: {sys.version}")
    
    # PyTorch
    print(f"🔥 PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"   - CUDA: Available (Version: {torch.version.cuda})")
        print(f"   - GPU Count: {torch.cuda.device_count()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("   - MPS (Apple Silicon): Available")
    else:
        print("   - Device: CPU only")
    
    # OpenCV
    print(f"📷 OpenCV: {cv2.__version__}")
    
    # Numpy
    print(f"🔢 NumPy: {np.__version__}")
    
    # Test basic functionality
    print("\n🧪 Testing basic functionality...")
    
    # PyTorch tensor test
    x = torch.randn(1, 3, 224, 224)
    print(f"✅ PyTorch tensor creation: {x.shape}")
    
    # OpenCV test
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    print(f"✅ OpenCV image creation: {img.shape}")
    
    # ResNet test
    try:
        from torchvision import models
        model = models.resnet18(pretrained=False)
        print("✅ ResNet model loading: Success")
    except Exception as e:
        print(f"❌ ResNet model loading: {e}")
    
    print("\n🎉 Installation test completed!")

if __name__ == "__main__":
    test_installation()
EOF

# Test çalıştır
echo -e "${BLUE}🧪 Running installation test...${NC}"
python test_installation.py

# Aktivasyon scripti oluştur
echo -e "${BLUE}📝 Creating activation script...${NC}"
cat > activate_magnum_env.sh << EOF
#!/bin/bash
# Magnum Quality Control Environment Activator

echo "🍦 Activating Magnum Quality Control Environment..."
source $ENV_NAME/bin/activate
echo "✅ Environment activated!"
echo ""
echo "🚀 Quick start commands:"
echo "  python ice_cream_quality_control_v3.py --help"
echo "  python ice_cream_quality_control_v3.py --good_dir good_images --test_dir test_images"
echo ""
echo "📁 Directory structure:"
echo "  good_images/     - Reference good images"
echo "  test_images/     - Images to test"
echo "  seperated_images/ - Detected anomalies"
echo ""
echo "To deactivate: deactivate"
EOF

chmod +x activate_magnum_env.sh

echo ""
echo -e "${GREEN}🎉 Setup completed successfully!${NC}"
echo ""
echo -e "${BLUE}📋 Next steps:${NC}"
echo "1. Activate environment:  source activate_magnum_env.sh"
echo "2. Add good images to:    good_images/"
echo "3. Add test images to:    test_images/"
echo "4. Run system:           python ice_cream_quality_control_v3.py --good_dir good_images --test_dir test_images"
echo ""
echo -e "${YELLOW}💡 Tips:${NC}"
echo "• Use --device cpu for CPU-only processing"
echo "• Use --device mps for Apple Silicon GPU acceleration"
echo "• Check DOCKER_KURULUM.md for Docker installation alternative"
