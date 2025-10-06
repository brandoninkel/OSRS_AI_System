#!/usr/bin/env python3
"""
Create OSRS AI Admin GUI Icon
Generates a professional icon for the macOS app bundle
"""

import os
import sys
import subprocess
from pathlib import Path

def create_icon_with_pil():
    """Create icon using PIL/Pillow"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        print("🎨 Creating icon with PIL...")
        
        # Icon sizes for macOS .icns
        sizes = [16, 32, 64, 128, 256, 512, 1024]
        
        # Get resources directory
        script_dir = Path(__file__).parent
        app_bundle = script_dir / "OSRS AI Admin.app"
        resources_dir = app_bundle / "Contents" / "Resources"
        resources_dir.mkdir(parents=True, exist_ok=True)
        
        # Create icons at each size
        for size in sizes:
            # Create image with dark background
            img = Image.new('RGB', (size, size), color='#1e1e2e')
            draw = ImageDraw.Draw(img)
            
            # Draw gold border
            border = max(2, size // 32)
            draw.rectangle(
                [border, border, size-border-1, size-border-1],
                outline='#f9e2af',
                width=border
            )
            
            # Draw inner accent
            inner_border = border * 2
            draw.rectangle(
                [inner_border, inner_border, size-inner_border-1, size-inner_border-1],
                outline='#89b4fa',
                width=max(1, border // 2)
            )
            
            # Draw text if size is large enough
            if size >= 64:
                try:
                    font_size = size // 5
                    # Try to use a bold system font
                    font_paths = [
                        "/System/Library/Fonts/Helvetica.ttc",
                        "/System/Library/Fonts/SFNSDisplay.ttf",
                        "/Library/Fonts/Arial Bold.ttf"
                    ]
                    
                    font = None
                    for font_path in font_paths:
                        if os.path.exists(font_path):
                            try:
                                font = ImageFont.truetype(font_path, font_size)
                                break
                            except:
                                continue
                    
                    if font is None:
                        font = ImageFont.load_default()
                    
                    # Draw "OSRS" and "AI" text
                    text1 = "OSRS"
                    text2 = "AI"
                    
                    # Calculate positions
                    bbox1 = draw.textbbox((0, 0), text1, font=font)
                    bbox2 = draw.textbbox((0, 0), text2, font=font)
                    
                    text1_width = bbox1[2] - bbox1[0]
                    text2_width = bbox2[2] - bbox2[0]
                    text_height = bbox1[3] - bbox1[1]
                    
                    # Center text
                    x1 = (size - text1_width) // 2
                    y1 = (size - text_height * 2) // 2
                    x2 = (size - text2_width) // 2
                    y2 = y1 + text_height
                    
                    # Draw text with shadow
                    shadow_offset = max(1, size // 128)
                    draw.text((x1 + shadow_offset, y1 + shadow_offset), text1, fill='#000000', font=font)
                    draw.text((x1, y1), text1, fill='#a6e3a1', font=font)
                    
                    draw.text((x2 + shadow_offset, y2 + shadow_offset), text2, fill='#000000', font=font)
                    draw.text((x2, y2), text2, fill='#cba6f7', font=font)
                    
                except Exception as e:
                    print(f"   ⚠️  Could not add text to {size}x{size} icon: {e}")
            
            # Save PNG
            png_path = resources_dir / f'icon_{size}x{size}.png'
            img.save(png_path)
            print(f"   ✅ Created {size}x{size} icon")
        
        # Create .iconset directory
        iconset_dir = resources_dir / 'AppIcon.iconset'
        iconset_dir.mkdir(exist_ok=True)
        
        # Copy PNGs to iconset with proper naming for macOS
        size_map = {
            16: ['icon_16x16.png'],
            32: ['icon_16x16@2x.png', 'icon_32x32.png'],
            64: ['icon_32x32@2x.png'],
            128: ['icon_128x128.png'],
            256: ['icon_128x128@2x.png', 'icon_256x256.png'],
            512: ['icon_256x256@2x.png', 'icon_512x512.png'],
            1024: ['icon_512x512@2x.png']
        }
        
        import shutil
        for size, names in size_map.items():
            src = resources_dir / f'icon_{size}x{size}.png'
            if src.exists():
                for name in names:
                    dst = iconset_dir / name
                    shutil.copy(src, dst)
        
        # Convert to .icns using iconutil
        print("🔄 Converting to .icns format...")
        icns_path = resources_dir / 'AppIcon.icns'
        result = subprocess.run(
            ['iconutil', '-c', 'icns', str(iconset_dir), '-o', str(icns_path)],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ Created AppIcon.icns")
            
            # Clean up temporary files
            shutil.rmtree(iconset_dir)
            for size in sizes:
                png_path = resources_dir / f'icon_{size}x{size}.png'
                if png_path.exists():
                    png_path.unlink()
            
            return True
        else:
            print(f"❌ iconutil failed: {result.stderr}")
            return False
            
    except ImportError:
        print("❌ PIL/Pillow not installed")
        return False
    except Exception as e:
        print(f"❌ Error creating icon: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_simple_icon():
    """Create a simple icon without PIL"""
    print("🎨 Creating simple icon...")
    
    script_dir = Path(__file__).parent
    app_bundle = script_dir / "OSRS AI Admin.app"
    resources_dir = app_bundle / "Contents" / "Resources"
    resources_dir.mkdir(parents=True, exist_ok=True)
    
    # Use macOS built-in icon as base
    generic_icon = "/System/Library/CoreServices/CoreTypes.bundle/Contents/Resources/GenericApplicationIcon.icns"
    dest_icon = resources_dir / "AppIcon.icns"
    
    import shutil
    shutil.copy(generic_icon, dest_icon)
    
    print("✅ Created basic icon")
    return True

def main():
    print("🚀 OSRS AI Admin Icon Creator")
    print("=" * 50)
    
    # Try PIL first
    if create_icon_with_pil():
        print("\n✅ Icon created successfully with PIL!")
        return 0
    
    # Fall back to simple icon
    print("\n⚠️  Falling back to simple icon...")
    if create_simple_icon():
        print("\n✅ Simple icon created!")
        print("   💡 Install Pillow for custom icon: pip install Pillow")
        return 0
    
    print("\n❌ Failed to create icon")
    return 1

if __name__ == "__main__":
    sys.exit(main())

