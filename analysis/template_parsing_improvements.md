# OSRS Wiki Template Parsing Improvements

## Overview
Enhanced the OSRS wiki template parser to provide comprehensive labeling for all parameters and numeric values, ensuring that embeddings and LLMs can properly understand the structured data.

## Key Improvements Made

### 1. Enhanced Infobox Monster Template Processing
**Before:**
```
hitpoints = 500
att = 1
def = 300
dmagic1 = -45
id1 = 2042
```

**After:**
```
Hitpoints: 500
Attack Level: 1
Defence Level: 300
Magic Defence (Form 1): -45
NPC ID (Form 1): 2042
```

### 2. Comprehensive Parameter Mapping
Added 100+ parameter mappings covering:
- **Combat Stats**: hitpoints, att, str, def, mage, range
- **Attack/Defence Bonuses**: attbns, strbns, amagic, mbns, dstab, dslash, etc.
- **Form-Specific Stats**: dmagic1/2/3, id1/2/3, examine1/2/3
- **Slayer Info**: slayxp, cat, assignedby
- **Immunities**: immunepoison, immunevenom, immunecannon
- **Elemental Weaknesses**: elementalweaknesstype1, elementalweaknesspercent1

### 3. Enhanced Drop Table Processing
**Before:**
```
{{DropsLine|name=Tanzanite fang|quantity=1|rarity=1/1024|rolls=2}}
```

**After:**
```
Item Name: Tanzanite fang | Quantity: 1 | Drop Rate: 1/1024 | Rolls Per Kill: 2
```

### 4. Generic Template Enhancement
Added comprehensive fallback mappings for common parameters:
- Numeric parameters (1, 2, 3, 4, 5) → "Parameter 1", "Parameter 2", etc.
- Common fields: name, id, value, cost, level, xp, skill, location
- Combat fields: damage, accuracy, speed, range, style
- Drop fields: drop, loot, reward, chance, rate, table, tier
- Coordinate fields: x, y, z, plane, region, area
- Time fields: time, duration, cooldown, respawn, timer

### 5. HTML Entity Cleaning
**Before:**
```
?&#160;?&#160;?&#160;? NPC ID 1182 ?&#160;?&#160;?&#160;?
```

**After:**
```
? ? ? ? NPC ID 1182 ? ? ? ?
```

Properly decodes:
- `&#160;` → space
- `&nbsp;` → space  
- `&amp;` → &
- `&lt;` → <
- `&gt;` → >
- `&quot;` → "

### 6. Apple Metal GPU Detection
Successfully detects Apple Silicon M4 Pro + Metal GPU for parallel processing acceleration:
```
🍎 Apple Silicon M4 Pro + Metal GPU acceleration detected!
GPU Acceleration Available: true
```

## Impact on AI Training

### For Embeddings (mxbai)
- Every numeric value now has semantic context
- Parameter relationships are clear
- No more unlabeled strings of numbers
- Structured data is properly formatted for vector embedding

### For LLM (Llama 3.1)
- Clear, readable parameter labels
- Contextual understanding of game mechanics
- Proper stat relationships (e.g., "Magic Defence (Form 1): -45")
- Enhanced comprehension of drop rates, combat stats, and game data

## Example Transformation

### Raw Wikitext:
```
{{Infobox Monster|combat=725|hitpoints=500|att=1|def=300|dmagic1=-45|id1=2042}}
```

### Old Processing:
```
Combat: 725, Hitpoints: 500, Att: 1, Def: 300, Dmagic1: -45, Id1: 2042
```

### New Enhanced Processing:
```
=== Monster Information ===
Combat Level: 725
Hitpoints: 500
Attack Level: 1
Defence Level: 300
Magic Defence (Form 1): -45
NPC ID (Form 1): 2042
```

## Technical Implementation
- Enhanced `format_infobox()` with 80+ parameter mappings
- Improved `format_drops_line()` with drop-specific labels
- Upgraded `format_generic_template()` with 60+ common parameter mappings
- Enhanced `clean_wiki_markup()` with HTML entity decoding
- Apple Metal GPU detection for parallel processing acceleration

## Result
Every single numeric value and parameter in OSRS wiki templates now has proper, descriptive labeling that provides full context for both embedding models and LLMs to understand the game mechanics and data relationships.
