#!/bin/bash

# Voice Assistant Setup Script

echo "🚀 Setting up Voice Assistant..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "✅ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads
mkdir -p logs

# Copy environment file
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your configuration"
fi

echo "✨ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your model paths"
echo "2. Run: python main.py"
echo "3. Open http://localhost:8000 in your browser"
