# OSRS AI Admin - macOS Application Bundle

**Version**: 2.0.0  
**Created**: October 5, 2025  
**Type**: Native macOS Application (.app)

---

## 🎯 Overview

The OSRS AI Admin GUI is now available as a **native macOS application** with a proper icon and app bundle structure. You can launch it like any other Mac app!

---

## 📦 What's Included

### **Application Bundle**: `OSRS AI Admin.app`
```
OSRS AI Admin.app/
├── Contents/
│   ├── Info.plist          # App metadata
│   ├── MacOS/
│   │   └── launch          # Launch script
│   └── Resources/
│       └── AppIcon.icns    # Custom OSRS-themed icon
```

### **Icon Design**
- **Theme**: OSRS-inspired with dark background
- **Colors**: 
  - Background: Dark blue-gray (#1e1e2e)
  - Border: Gold (#f9e2af)
  - Accent: Blue (#89b4fa)
  - Text: Green "OSRS" (#a6e3a1) + Purple "AI" (#cba6f7)
- **Sizes**: 16x16 to 1024x1024 (Retina-ready)

---

## 🚀 How to Use

### **Option 1: Launch from Finder** (Recommended)
1. Open Finder
2. Navigate to: `/Users/brandon/Documents/projects/GE/admin/`
3. Double-click **"OSRS AI Admin.app"**
4. The GUI will launch automatically

### **Option 2: Add to Applications**
1. Drag **"OSRS AI Admin.app"** to `/Applications/`
2. Launch from Launchpad or Spotlight
3. Search for "OSRS AI" and press Enter

### **Option 3: Add to Dock**
1. Launch the app (Option 1 or 2)
2. Right-click the icon in the Dock
3. Select: **Options → Keep in Dock**
4. Now you can launch with one click!

### **Option 4: Create Desktop Alias**
1. Right-click **"OSRS AI Admin.app"**
2. Select **"Make Alias"**
3. Drag the alias to your Desktop
4. Rename to "OSRS AI Admin" (removes "alias" suffix)

---

## 🔧 How It Works

### **Launch Process**
1. User double-clicks the app icon
2. macOS executes `Contents/MacOS/launch` script
3. Script finds Python 3 on your system
4. Script checks for PyQt6 (installs if missing)
5. Script launches `pyqt6_admin_gui.py`
6. GUI appears with full functionality

### **Python Detection**
The launch script automatically finds Python:
- Checks for `python3` command
- Falls back to `python` command
- Shows error dialog if Python not found

### **Dependency Check**
The launch script checks for PyQt6:
- If installed: Launches immediately
- If missing: Shows dialog and installs automatically
- Uses `pip install PyQt6 psutil`

---

## 🛠️ Rebuilding the App Bundle

If you need to rebuild the app (e.g., after updating the GUI):

### **Quick Rebuild**
```bash
cd /Users/brandon/Documents/projects/GE/admin
./create_app_bundle.sh
```

### **Rebuild with New Icon**
```bash
cd /Users/brandon/Documents/projects/GE/admin
./create_app_bundle.sh
python3 create_icon.py
```

### **What Gets Rebuilt**
- ✅ App bundle structure
- ✅ Info.plist metadata
- ✅ Launch script
- ✅ App icon (if you run create_icon.py)

---

## 📋 Files Created

### **App Bundle**
- `OSRS AI Admin.app` - The macOS application (104 KB icon + structure)

### **Build Scripts**
- `create_app_bundle.sh` - Creates the .app bundle structure
- `create_icon.py` - Generates the custom OSRS icon

### **Documentation**
- `APP_BUNDLE_README.md` - This file

---

## 🔍 Troubleshooting

### **Issue: "App can't be opened because it is from an unidentified developer"**

**Solution 1: Allow in System Preferences**
1. Go to **System Preferences → Security & Privacy**
2. Click **"Open Anyway"** next to the blocked app message
3. Confirm by clicking **"Open"**

**Solution 2: Remove Quarantine Attribute**
```bash
cd /Users/brandon/Documents/projects/GE/admin
xattr -cr "OSRS AI Admin.app"
```

---

### **Issue: "Python 3 is required but not found"**

**Solution**: Install Python 3
```bash
# Using Homebrew
brew install python3

# Or download from python.org
open https://www.python.org/downloads/
```

---

### **Issue: "PyQt6 is not installed"**

**Solution**: The app should auto-install, but you can manually install:
```bash
python3 -m pip install PyQt6 psutil
```

---

### **Issue: App icon doesn't show**

**Solution 1: Rebuild icon**
```bash
cd /Users/brandon/Documents/projects/GE/admin
python3 create_icon.py
```

**Solution 2: Clear icon cache**
```bash
# Force Finder to reload icons
sudo rm -rf /Library/Caches/com.apple.iconservices.store
killall Finder
```

---

### **Issue: App launches but GUI doesn't appear**

**Solution**: Check logs
```bash
# View system logs for the app
log show --predicate 'process == "OSRS-AI-Admin"' --last 5m

# Or check if Python process is running
ps aux | grep pyqt6_admin_gui
```

---

## 🎨 Customizing the Icon

### **Edit Icon Colors**
Edit `create_icon.py` and change these values:
```python
# Background color
img = Image.new('RGB', (size, size), color='#1e1e2e')

# Border color
draw.rectangle(..., outline='#f9e2af', ...)

# Text colors
draw.text(..., fill='#a6e3a1', ...)  # OSRS text
draw.text(..., fill='#cba6f7', ...)  # AI text
```

### **Rebuild Icon**
```bash
python3 admin/create_icon.py
```

### **Use Custom Image**
Replace `AppIcon.icns` with your own:
```bash
# Convert PNG to ICNS
mkdir MyIcon.iconset
# Add icon_512x512.png, icon_256x256.png, etc.
iconutil -c icns MyIcon.iconset -o "admin/OSRS AI Admin.app/Contents/Resources/AppIcon.icns"
```

---

## 📊 Technical Details

### **App Bundle Structure**
- **Type**: APPL (Application)
- **Identifier**: com.osrs-ai.admin
- **Version**: 2.0.0
- **Minimum macOS**: 10.13 (High Sierra)
- **High Resolution**: Yes (Retina-ready)
- **Dark Mode**: Supported

### **Launch Script**
- **Shell**: Bash
- **Python**: Auto-detected (python3 or python)
- **Dependencies**: PyQt6, psutil (auto-installed)
- **Logging**: System log (tag: OSRS-AI-Admin)

### **Icon Format**
- **Format**: ICNS (Apple Icon Image)
- **Sizes**: 16, 32, 64, 128, 256, 512, 1024 pixels
- **Retina**: @2x variants included
- **File Size**: ~104 KB

---

## 🎉 Benefits

### **User Experience**
- ✅ **Native macOS app** - Looks and feels like a real Mac app
- ✅ **Custom icon** - OSRS-themed, professional appearance
- ✅ **Dock integration** - Pin to Dock for quick access
- ✅ **Spotlight search** - Find with Cmd+Space → "OSRS AI"
- ✅ **Launchpad** - Appears in Launchpad with other apps

### **Developer Experience**
- ✅ **Easy distribution** - Just share the .app file
- ✅ **Auto-dependency check** - Installs PyQt6 if needed
- ✅ **Proper logging** - Logs to system log
- ✅ **Rebuild scripts** - Easy to update

### **Security**
- ✅ **No API exposure** - Uses shell scripts directly
- ✅ **Local-only** - Requires filesystem access
- ✅ **Sandboxed** - Runs in user context

---

## 📝 Summary

**What You Get**:
- 🎨 Native macOS app with custom OSRS icon
- 🚀 One-click launch from Finder/Dock/Launchpad
- 🔧 Auto-installs dependencies (PyQt6)
- 📊 Professional appearance and integration
- 🔒 Secure (no API exposure)

**How to Use**:
1. Double-click **"OSRS AI Admin.app"**
2. Drag to Applications or Dock
3. Launch like any other Mac app

**How to Rebuild**:
```bash
cd admin
./create_app_bundle.sh
python3 create_icon.py
```

---

**Enjoy your native OSRS AI Admin app!** 🎉

