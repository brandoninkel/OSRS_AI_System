#!/bin/bash
###############################################################################
# Create macOS Application Bundle for OSRS AI Admin GUI
# Creates a proper .app that can be launched from Finder with an icon
###############################################################################

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Creating OSRS AI Admin GUI Application Bundle...${NC}"

# Paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
APP_NAME="OSRS AI Admin"
APP_BUNDLE="$SCRIPT_DIR/$APP_NAME.app"
CONTENTS_DIR="$APP_BUNDLE/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"

# Clean up old bundle if it exists
if [ -d "$APP_BUNDLE" ]; then
    echo -e "${YELLOW}🗑️  Removing old app bundle...${NC}"
    rm -rf "$APP_BUNDLE"
fi

# Create app bundle structure
echo -e "${BLUE}📁 Creating app bundle structure...${NC}"
mkdir -p "$MACOS_DIR"
mkdir -p "$RESOURCES_DIR"

# Create Info.plist
echo -e "${BLUE}📝 Creating Info.plist...${NC}"
cat > "$CONTENTS_DIR/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>launch</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleIdentifier</key>
    <string>com.osrs-ai.admin</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>OSRS AI Admin</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>2.0.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSRequiresAquaSystemAppearance</key>
    <false/>
</dict>
</plist>
EOF

# Create launch script
echo -e "${BLUE}🚀 Creating launch script...${NC}"
cat > "$MACOS_DIR/launch" << 'EOF'
#!/bin/bash
# Launch script for OSRS AI Admin GUI

# Get the directory where the app bundle is located
APP_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
ADMIN_DIR="$(dirname "$APP_DIR")"
PROJECT_ROOT="$(dirname "$ADMIN_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Set up PATH to include common Python locations
export PATH="/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin:$HOME/.pyenv/shims:$PATH"

# Enable user site-packages (where PyQt6 is installed for Python 3.9)
export PYTHONUSERBASE="$HOME/Library/Python/3.9"

# Function to test if Python has PyQt6
test_python_pyqt6() {
    local python_cmd="$1"
    if command -v "$python_cmd" &> /dev/null; then
        # Test with user site-packages enabled
        if "$python_cmd" -c "import PyQt6" 2>/dev/null; then
            echo "$python_cmd"
            return 0
        fi
    fi
    return 1
}

# Try to find Python with PyQt6 installed
PYTHON_CMD=""

# Try common Python commands in order (including Xcode Python)
for cmd in python3 /Applications/Xcode.app/Contents/Developer/usr/bin/python3 /usr/bin/python3 /usr/local/bin/python3 /opt/homebrew/bin/python3 python; do
    if PYTHON_CMD=$(test_python_pyqt6 "$cmd"); then
        break
    fi
done

# If no Python with PyQt6 found, show error with details
if [ -z "$PYTHON_CMD" ]; then
    # Get Python version for error message
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1)
        PYTHON_PATH=$(which python3)

        osascript -e "display dialog \"PyQt6 not found in system Python.\n\nPython: $PYTHON_PATH\nVersion: $PYTHON_VERSION\n\nPlease install PyQt6:\n  python3 -m pip install --user PyQt6 psutil\n\nOr run from terminal:\n  cd $ADMIN_DIR\n  python3 pyqt6_admin_gui.py\" buttons {\"OK\"} default button \"OK\" with icon stop"
    else
        osascript -e 'display dialog "Python 3 is required but not found. Please install Python 3." buttons {"OK"} default button "OK" with icon stop'
    fi
    exit 1
fi

# Launch the GUI
exec "$PYTHON_CMD" "$ADMIN_DIR/pyqt6_admin_gui.py" 2>&1 | logger -t "OSRS-AI-Admin"
EOF

chmod +x "$MACOS_DIR/launch"

# Create app icon using Python (generates a simple icon)
echo -e "${BLUE}🎨 Creating app icon...${NC}"
python3 << 'PYTHON_SCRIPT'
import os
from pathlib import Path

# Try to create a simple icon using PIL if available
try:
    from PIL import Image, ImageDraw, ImageFont
    
    # Create icon at multiple sizes for .icns
    sizes = [16, 32, 64, 128, 256, 512, 1024]
    icon_dir = Path(os.environ['RESOURCES_DIR'])
    
    for size in sizes:
        img = Image.new('RGB', (size, size), color='#1e1e2e')
        draw = ImageDraw.Draw(img)
        
        # Draw a simple OSRS-themed icon (gold border, dark background)
        border = max(2, size // 32)
        draw.rectangle(
            [border, border, size-border, size-border],
            outline='#f9e2af',
            width=border
        )
        
        # Draw "OSRS" text if size is large enough
        if size >= 64:
            try:
                font_size = size // 4
                # Try to use a system font
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
                except:
                    font = ImageFont.load_default()
                
                text = "OSRS\nAI"
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                x = (size - text_width) // 2
                y = (size - text_height) // 2
                draw.text((x, y), text, fill='#a6e3a1', font=font, align='center')
            except:
                pass
        
        # Save PNG
        img.save(icon_dir / f'icon_{size}x{size}.png')
    
    # Create .icns file using iconutil (macOS only)
    iconset_dir = icon_dir / 'AppIcon.iconset'
    iconset_dir.mkdir(exist_ok=True)
    
    # Copy PNGs to iconset with proper naming
    size_map = {
        16: ['icon_16x16.png'],
        32: ['icon_16x16@2x.png', 'icon_32x32.png'],
        64: ['icon_32x32@2x.png'],
        128: ['icon_128x128.png'],
        256: ['icon_128x128@2x.png', 'icon_256x256.png'],
        512: ['icon_256x256@2x.png', 'icon_512x512.png'],
        1024: ['icon_512x512@2x.png']
    }
    
    for size, names in size_map.items():
        src = icon_dir / f'icon_{size}x{size}.png'
        if src.exists():
            for name in names:
                dst = iconset_dir / name
                import shutil
                shutil.copy(src, dst)
    
    # Convert to .icns
    import subprocess
    subprocess.run(['iconutil', '-c', 'icns', str(iconset_dir), '-o', str(icon_dir / 'AppIcon.icns')], check=True)
    
    # Clean up
    import shutil
    shutil.rmtree(iconset_dir)
    for size in sizes:
        (icon_dir / f'icon_{size}x{size}.png').unlink()
    
    print("✅ Icon created successfully")

except ImportError:
    print("⚠️  PIL not available, creating simple icon...")
    # Create a very basic icon without PIL
    icon_dir = Path(os.environ['RESOURCES_DIR'])
    
    # Create a simple 512x512 PNG with basic drawing
    import subprocess
    
    # Use sips to create a basic colored icon (macOS built-in tool)
    icon_path = icon_dir / 'AppIcon.icns'
    
    # Create a simple colored square using sips
    subprocess.run([
        'sips', '-s', 'format', 'png',
        '--resampleWidth', '512',
        '--padColor', '1e1e2e',
        '/System/Library/CoreServices/CoreTypes.bundle/Contents/Resources/GenericApplicationIcon.icns',
        '--out', str(icon_path)
    ], capture_output=True)
    
    print("✅ Basic icon created")

except Exception as e:
    print(f"⚠️  Could not create icon: {e}")
    print("   App will use default icon")
PYTHON_SCRIPT

# Make the app bundle executable
chmod -R 755 "$APP_BUNDLE"

echo -e "${GREEN}✅ App bundle created successfully!${NC}"
echo -e "${BLUE}📍 Location: $APP_BUNDLE${NC}"
echo ""
echo -e "${YELLOW}📋 Next steps:${NC}"
echo -e "  1. Double-click '${APP_NAME}.app' to launch"
echo -e "  2. Drag to Applications folder for permanent installation"
echo -e "  3. Drag to Dock for quick access"
echo ""
echo -e "${GREEN}🎉 Done!${NC}"

