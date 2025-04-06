#!/bin/bash
set -e

# 🔐 تحميل التوكن من .env
export $(grep -v '^#' .env | xargs)

# 🛠️ تثبيت mise
chmod +x "$0"
mise trust
mise install
eval "$(mise activate bash)"

# ✅ إنشاء البيئة الافتراضية لو مش موجودة
if [ ! -d ".venv" ]; then
  echo "🔧 Creating virtual environment..."
  python -m venv .venv
fi

# ✅ تفعيل البيئة
source .venv/bin/activate

# ✅ تثبيت الباكجات
echo "📦 Installing python requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Environment setup complete!"
