#!/usr/bin/env python3
"""
OSRS Wiki Template Parser - Converts MediaWiki templates to readable text
Fixes the stat formatting problem by parsing templates instead of removing them
NOW WITH ULTRA-AGGRESSIVE PARALLEL PROCESSING - UNCAPPED WORKERS!
"""

import re
import json
import time
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

class OSRSWikiTemplateParser:
    """Parse OSRS MediaWiki templates into readable text with ULTRA-AGGRESSIVE PARALLEL PROCESSING"""

    def __init__(self):
        # Detect system capabilities for UNCAPPED scaling
        self.cpu_cores = multiprocessing.cpu_count()
        self.has_gpu = self.detect_gpu_acceleration()
        self.max_workers = self.calculate_uncapped_workers()

        # OS limit discovery - learn safe limits during runtime
        self.discovered_max_workers = None
        self.safe_max_workers = None

        if os.getenv('OSRS_PARSER_VERBOSE') == '1' and not getattr(OSRSWikiTemplateParser, '_banner_printed', False):
            print(f"🚀 OSRS Template Parser initialized with UNCAPPED scaling:")
            print(f"   📊 CPU cores: {self.cpu_cores}")
            print(f"   🎮 GPU acceleration: {'✅ Enabled' if self.has_gpu else '❌ Disabled'}")
            print(f"   ⚡ UNCAPPED max workers: {self.max_workers:,}")
            print(f"   🎯 Ready for EXTREME parallel template processing!")
            OSRSWikiTemplateParser._banner_printed = True

    def detect_gpu_acceleration(self) -> bool:
        """Detect Apple Metal GPU acceleration"""
        try:
            import platform
            system = platform.system()
            machine = platform.machine()

            # Check for Apple Silicon with Metal GPU
            if system == "Darwin" and machine.startswith("arm"):
                if os.getenv('OSRS_PARSER_VERBOSE') == '1':
                    print("   🍎 Apple Silicon + Metal GPU detected!")
                return True

            # Could add NVIDIA/AMD GPU detection here
            return False
        except:
            return False

    def calculate_uncapped_workers(self) -> int:
        """Return a safe, capped worker count from env (default 16, max 64)."""
        try:
            cap_str = os.getenv('OSRS_PARSER_MAX_WORKERS', os.getenv('OSRS_CHECKER_MAX_WORKERS', '16'))
            cap = int(cap_str)
        except Exception:
            cap = 16
        cap = max(1, min(cap, 64))
        if os.getenv('OSRS_PARSER_VERBOSE') == '1':
            print(f"   ⚡ Max workers capped at {cap}")
        return cap

    def __init_mappings__(self):
        """Initialize all the template parameter mappings"""
        # Combat stat mappings
        self.combat_stats = {
            'att': 'Attack',
            'attack': 'Attack', 
            'str': 'Strength',
            'strength': 'Strength',
            'def': 'Defence',
            'defence': 'Defence',
            'range': 'Ranged',
            'ranged': 'Ranged',
            'mage': 'Magic',
            'magic': 'Magic',
            'prayer': 'Prayer',
            'hitpoints': 'Hitpoints',
            'hp': 'Hitpoints',
            'combat': 'Combat Level'
        }
        
        # Equipment bonus mappings
        self.equipment_bonuses = {
            'astab': 'Stab Attack',
            'aslash': 'Slash Attack', 
            'acrush': 'Crush Attack',
            'amagic': 'Magic Attack',
            'arange': 'Ranged Attack',
            'dstab': 'Stab Defence',
            'dslash': 'Slash Defence',
            'dcrush': 'Crush Defence',
            'dmagic': 'Magic Defence',
            'drange': 'Ranged Defence',
            'strbns': 'Strength Bonus',
            'rngbns': 'Ranged Strength',
            'mbns': 'Magic Damage',
            'str': 'Strength',
            'rstr': 'Ranged Strength',
            'mdmg': 'Magic Damage',
            'prayer': 'Prayer Bonus'
        }
        
        # Item stat mappings
        self.item_stats = {
            'attack': 'Attack Requirement',
            'strength': 'Strength Requirement',
            'defence': 'Defence Requirement',
            'ranged': 'Ranged Requirement',
            'magic': 'Magic Requirement',
            'prayer': 'Prayer Requirement',
            'speed': 'Attack Speed',
            'slot': 'Equipment Slot',
            'weight': 'Weight'
        }

    def parse_template(self, template_text: str) -> str:
        """Parse a single MediaWiki template into readable text"""
        # Remove outer braces and extract content
        if not (template_text.startswith('{{') and template_text.endswith('}}')):
            return template_text

        inner_content = template_text[2:-2].strip()

        # Split by first | to get template name and parameters
        if '|' in inner_content:
            template_name = inner_content.split('|', 1)[0].strip()
            params_text = inner_content.split('|', 1)[1]
        else:
            template_name = inner_content.strip()
            params_text = ""
        
        # Parse parameters
        params = self.parse_template_params(params_text)
        
        # Handle different template types
        template_lower = template_name.lower()

        if template_lower.startswith('infobox'):
            if template_lower == 'infobox quest':
                return self.format_infobox_quest(params)
            elif template_lower == 'infobox item':
                return self.format_infobox_item(params)
            elif template_lower == 'infobox bonuses':
                return self.format_infobox_bonuses(params)
            elif template_lower == 'infobox npc':
                return self.format_infobox_npc(params)
            elif template_lower == 'infobox location':
                return self.format_infobox_location(params)
            elif template_lower == 'infobox skill':
                return self.format_infobox_skill(params)
            else:
                return self.format_infobox(template_name, params)
        elif 'combat' in template_lower and 'stats' in template_lower:
            return self.format_combat_stats(params)
        elif template_lower in ['stats', 'combat stats']:
            return self.format_combat_stats(params)
        elif template_lower.startswith('dropsline'):
            return self.format_drops_line(params)
        elif template_lower.startswith('dropstable'):
            return self.format_drops_table(template_name, params)
        elif template_lower in ['recipe', 'creation']:
            return self.format_recipe(params)
        elif template_lower.startswith('locline'):
            return self.format_location_line(params)
        elif template_lower.startswith('emoteclue'):
            return self.format_emote_clue(params)
        elif template_lower in ['external', 'otheruses', 'confuse']:
            return self.format_navigation_template(template_name, params)
        elif template_lower.startswith('subject changes'):
            return self.format_subject_changes(params)
        elif template_lower in ['map']:
            return self.format_map_template(params)
        elif template_lower.startswith('relative location'):
            return self.format_relative_location(params)
        elif template_lower in ['boostable']:
            return self.format_boostable(params)
        elif template_lower in ['hastranscript']:
            return self.format_transcript(params)
        elif template_lower.startswith('thieving info'):
            return self.format_thieving_info(params)
        elif template_lower.startswith('skilling success'):
            return self.format_skilling_chart(params)
        elif template_lower.startswith('equipment'):
            return self.format_equipment_template(params)
        elif template_lower.startswith('inventory'):
            return self.format_inventory_template(params)
        elif template_lower.startswith('combatsty'):
            return self.format_combat_styles(params)
        elif template_lower.startswith('floornumber'):
            return self.format_floor_number(params)
        elif template_lower.startswith('stashlocation'):
            return self.format_stash_location(params)
        elif template_lower in ['diaryskillstats', 'diary skill stats']:
            return self.format_diary_skill_stats(params)
        elif template_lower.startswith('itemspawn'):
            return self.format_item_spawn_table(template_name, params)
        elif template_lower in ['plink', 'pl']:
            return self.format_page_link(params)
        elif template_lower in ['efn', 'footnote']:
            return self.format_footnote(params)
        elif template_lower in ['scp', 'skill calculator']:
            return self.format_skill_requirement(params)
        elif template_lower in ['quest details', 'questdetails']:
            return self.format_quest_details(params)
        elif template_lower in ['questreqstart', 'quest req start']:
            return self.format_quest_requirement_start(params)
        elif template_lower in ['boostable']:
            return self.format_boostable_requirement(params)
        elif template_lower in ['*', 'bullet']:
            return "•"
        elif template_lower in ['pagename']:
            return "[Page Name]"
        elif template_lower in ['sic']:
            return "[sic]"
        elif template_lower in ['re', 'redirect']:
            return self.format_redirect(params)
        elif template_lower in ['^', 'up']:
            return "↑"
        elif template_lower.startswith('average drop'):
            return self.format_average_drop(params)
        elif template_lower in ['okay', 'ok']:
            return "[OK]"
        elif template_lower.startswith('ipac'):
            return self.format_pronunciation(params)
        elif template_lower.startswith('citetwitter'):
            return self.format_citation(params)
        elif template_lower.startswith('questreq'):
            return self.format_quest_requirement(params)
        elif template_lower.startswith('fairycode'):
            return self.format_fairy_code(params)
        elif template_lower in ['uses material list', 'uses materials']:
            return self.format_uses_materials(params)
        elif template_lower in ['drop sources', 'dropsources']:
            return self.format_drop_sources(params)
        elif template_lower in ['store locations list', 'shop locations']:
            return self.format_store_locations(params)
        elif template_lower in ['used in recommended equipment', 'recommended equipment']:
            return self.format_recommended_equipment(params)
        elif template_lower in ['sherlockchallenge', 'sherlock challenge']:
            return self.format_sherlock_challenge(params)
        elif template_lower in ['faloitem', 'falo item']:
            return self.format_falo_item(params)
        elif template_lower in ['reflist', 'references']:
            return self.format_references(params)
        elif template_lower in ['main', 'main article']:
            return self.format_main_article(params)
        elif template_lower in ['citation', 'cite', 'citeforum', 'citenews', 'citetwitter']:
            return self.format_citation(params)
        elif template_lower in ['mmgsection', 'money making']:
            return self.format_money_making(params)
        elif template_lower in ['sync', 'synchronize']:
            return self.format_sync(params)
        elif template_lower in ['has calculator', 'calculator']:
            return self.format_calculator(params)
        elif template_lower in ['has skill guide', 'skill guide']:
            return self.format_skill_guide(params)
        elif template_lower in ['listen', 'audio']:
            return self.format_audio(params)
        elif template_lower in ['instance', 'instanced']:
            return self.format_instance(params)
        elif template_lower in ['hasstrategy', 'has strategy']:
            return self.format_strategy(params)
        elif template_lower in ['droplogproject', 'drop log']:
            return self.format_drop_log(params)
        elif template_lower in ['average drop value', 'drop value']:
            return self.format_drop_value(params)
        elif template_lower in ['respell', 'spelling']:
            return self.format_respell(params)
        elif template_lower in ['subject changes header', 'changes header']:
            return self.format_changes_header(params)
        elif template_lower in ['relativelocation', 'relative location']:
            return self.format_relative_location(params)
        elif template_lower in ['fact', 'citation needed']:
            return "[Citation Needed]"
        elif template_lower in ['coins', 'gp']:
            return self.format_coins(params)
        elif template_lower in ['members', 'member']:
            return "Members Only"
        elif template_lower in ['skills', 'skill']:
            return self.format_skill_link(params)
        elif template_lower in ['bosses', 'boss']:
            return self.format_boss_link(params)
        elif template_lower.endswith(' weapons') or template_lower.endswith(' equipment'):
            return self.format_equipment_category(template_name, params)
        else:
            return self.format_generic_template(template_name, params)

    def parse_template_params(self, params_text: str) -> Dict[str, str]:
        """Parse template parameters from text"""
        params = {}
        if not params_text:
            return params
        
        # Split by | but handle nested templates
        parts = []
        current_part = ""
        brace_count = 0
        
        for char in params_text:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
            elif char == '|' and brace_count == 0:
                parts.append(current_part)
                current_part = ""
                continue
            current_part += char
        
        if current_part:
            parts.append(current_part)
        
        # Parse each parameter
        positional_index = 1
        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                params[key.strip()] = value.strip()
            else:
                # Positional parameter
                params[str(positional_index)] = part.strip()
                positional_index += 1

        return params

    def format_infobox(self, template_name: str, params: Dict[str, str]) -> str:
        """Format infobox templates with comprehensive stat labels"""
        result = []

        # Add template type
        infobox_type = template_name.replace('Infobox', '').strip()
        if infobox_type:
            result.append(f"=== {infobox_type} Information ===")

        # Enhanced parameter mappings for OSRS
        enhanced_mappings = {
            # Basic info
            'name': 'Name',
            'members': 'Members Only',
            'combat': 'Combat Level',
            'size': 'Size',
            'release': 'Release Date',
            'update': 'Update',
            'examine': 'Examine Text',
            'examine1': 'Examine Text (Form 1)',
            'examine2': 'Examine Text (Form 2)',
            'examine3': 'Examine Text (Form 3)',
            'version1': 'Form 1',
            'version2': 'Form 2',
            'version3': 'Form 3',

            # Combat stats
            'hitpoints': 'Hitpoints',
            'att': 'Attack Level',
            'str': 'Strength Level',
            'def': 'Defence Level',
            'mage': 'Magic Level',
            'range': 'Ranged Level',
            'combat': 'Combat Level',

            # Attack bonuses
            'attbns': 'Attack Bonus',
            'strbns': 'Strength Bonus',
            'amagic': 'Magic Attack Bonus',
            'mbns': 'Magic Damage Bonus',
            'arange': 'Ranged Attack Bonus',
            'rngbns': 'Ranged Strength Bonus',

            # Defence bonuses
            'dstab': 'Stab Defence',
            'dslash': 'Slash Defence',
            'dcrush': 'Crush Defence',
            'dmagic': 'Magic Defence',
            'dmagic1': 'Magic Defence (Form 1)',
            'dmagic2': 'Magic Defence (Form 2)',
            'dmagic3': 'Magic Defence (Form 3)',
            'dlight': 'Light Ranged Defence',
            'dstandard': 'Standard Ranged Defence',
            'dheavy': 'Heavy Ranged Defence',
            'dlight1': 'Light Ranged Defence (Form 1)',
            'dstandard1': 'Standard Ranged Defence (Form 1)',
            'dheavy1': 'Heavy Ranged Defence (Form 1)',
            'dlight2': 'Light Ranged Defence (Form 2)',
            'dstandard2': 'Standard Ranged Defence (Form 2)',
            'dheavy2': 'Heavy Ranged Defence (Form 2)',
            'dlight3': 'Light Ranged Defence (Form 3)',
            'dstandard3': 'Standard Ranged Defence (Form 3)',
            'dheavy3': 'Heavy Ranged Defence (Form 3)',

            # Combat info
            'max hit': 'Maximum Hit',
            'max hit2': 'Maximum Hit (Form 2)',
            'aggressive': 'Aggressive',
            'poisonous': 'Poisonous',
            'attack style': 'Attack Style',
            'attack style1': 'Attack Style (Form 1)',
            'attack style2': 'Attack Style (Form 2)',
            'attack style3': 'Attack Style (Form 3)',
            'attack speed': 'Attack Speed',
            'xpbonus': 'XP Bonus',

            # Slayer info
            'slayxp': 'Slayer XP',
            'cat': 'Slayer Category',
            'assignedby': 'Assigned By',

            # Immunities
            'immunepoison': 'Poison Immunity',
            'immunevenom': 'Venom Immunity',
            'immunecannon': 'Cannon Immunity',
            'immunethrall': 'Thrall Immunity',

            # Elemental weaknesses
            'elementalweaknesstype1': 'Elemental Weakness Type (Form 1)',
            'elementalweaknesspercent1': 'Elemental Weakness Percent (Form 1)',
            'elementalweaknesstype2': 'Elemental Weakness Type (Form 2)',
            'elementalweaknesspercent2': 'Elemental Weakness Percent (Form 2)',
            'elementalweaknesstype3': 'Elemental Weakness Type (Form 3)',
            'elementalweaknesspercent3': 'Elemental Weakness Percent (Form 3)',

            # IDs and regions
            'id': 'NPC ID',
            'id1': 'NPC ID (Form 1)',
            'id2': 'NPC ID (Form 2)',
            'id3': 'NPC ID (Form 3)',
            'leagueregion': 'League Region',

            # Item stats
            'value': 'Value',
            'weight': 'Weight',
            'tradeable': 'Tradeable',
            'stackable': 'Stackable',
            'noteable': 'Noteable',
            'equipable': 'Equipable',
            'destroy': 'Destroy Text',
            'quest': 'Quest Item',
        }

        # Process all parameters with proper labels
        formatted_params = []
        for param, value in params.items():
            param_lower = param.lower().strip()

            # Skip empty values
            if not value or value.strip() == '':
                continue

            # Get proper label
            if param_lower in enhanced_mappings:
                label = enhanced_mappings[param_lower]
                formatted_params.append(f"{label}: {value}")
            else:
                # Fallback: capitalize and format parameter name
                formatted_name = param.replace('_', ' ').title()
                formatted_params.append(f"{formatted_name}: {value}")

        if formatted_params:
            result.extend(formatted_params)

        return "\n".join(result) + "\n"

    def format_infobox_quest(self, params: Dict[str, str]) -> str:
        """Format Infobox Quest templates with comprehensive labeling"""
        result = []
        result.append("=== Quest Information ===")

        # Quest-specific parameter mappings
        quest_mappings = {
            'name': 'Quest Name',
            'number': 'Quest Number',
            'image': 'Quest Image',
            'release': 'Release Date',
            'update': 'Update',
            'members': 'Members Only',
            'series': 'Quest Series',
            'developer': 'Developer',
            'start': 'Start Point',
            'startmap': 'Start Map Location',
            'difficulty': 'Difficulty',
            'description': 'Description',
            'length': 'Length',
            'requirements': 'Requirements',
            'items': 'Items Required',
            'recommended': 'Recommended',
            'kills': 'Enemies to Defeat',
            'ironman': 'Ironman Concerns',
            'leagueregion': 'League Region'
        }

        # Process all parameters with proper labels
        for param, value in params.items():
            param_lower = param.lower().strip()

            if not value or value.strip() == '':
                continue

            if param_lower in quest_mappings:
                label = quest_mappings[param_lower]
                result.append(f"{label}: {value}")
            else:
                formatted_name = param.replace('_', ' ').title()
                result.append(f"{formatted_name}: {value}")

        return "\n".join(result) + "\n"

    def format_infobox_item(self, params: Dict[str, str]) -> str:
        """Format Infobox Item templates with comprehensive labeling"""
        result = []
        result.append("=== Item Information ===")

        # Item-specific parameter mappings
        item_mappings = {
            'name': 'Item Name',
            'image': 'Item Image',
            'release': 'Release Date',
            'update': 'Update',
            'members': 'Members Only',
            'quest': 'Quest Item',
            'tradeable': 'Tradeable',
            'placeholder': 'Placeholder',
            'equipable': 'Equipable',
            'stackable': 'Stackable',
            'noteable': 'Noteable',
            'options': 'Options',
            'examine': 'Examine Text',
            'value': 'Value',
            'weight': 'Weight',
            'exchange': 'Grand Exchange',
            'id': 'Item ID',
            'leagueregion': 'League Region',
            'alchable': 'Alchable',
            'destroy': 'Destroy Text',
            'kept': 'Kept on Death',
            'highalch': 'High Alchemy Value',
            'lowalch': 'Low Alchemy Value'
        }

        # Process all parameters with proper labels
        for param, value in params.items():
            param_lower = param.lower().strip()

            if not value or value.strip() == '':
                continue

            if param_lower in item_mappings:
                label = item_mappings[param_lower]
                result.append(f"{label}: {value}")
            else:
                formatted_name = param.replace('_', ' ').title()
                result.append(f"{formatted_name}: {value}")

        return "\n".join(result)

    def format_infobox_bonuses(self, params: Dict[str, str]) -> str:
        """Format Infobox Bonuses templates with comprehensive labeling"""
        result = []
        result.append("=== Combat Bonuses ===")

        # Combat bonuses parameter mappings
        bonus_mappings = {
            'astab': 'Stab Attack Bonus',
            'aslash': 'Slash Attack Bonus',
            'acrush': 'Crush Attack Bonus',
            'amagic': 'Magic Attack Bonus',
            'arange': 'Ranged Attack Bonus',
            'dstab': 'Stab Defence Bonus',
            'dslash': 'Slash Defence Bonus',
            'dcrush': 'Crush Defence Bonus',
            'dmagic': 'Magic Defence Bonus',
            'drange': 'Ranged Defence Bonus',
            'str': 'Strength Bonus',
            'rstr': 'Ranged Strength Bonus',
            'mdmg': 'Magic Damage Bonus',
            'prayer': 'Prayer Bonus',
            'slot': 'Equipment Slot',
            'speed': 'Attack Speed',
            'range': 'Attack Range',
            'requirements': 'Requirements'
        }

        # Process all parameters with proper labels
        for param, value in params.items():
            param_lower = param.lower().strip()

            if not value or value.strip() == '':
                continue

            if param_lower in bonus_mappings:
                label = bonus_mappings[param_lower]
                result.append(f"{label}: {value}")
            else:
                formatted_name = param.replace('_', ' ').title()
                result.append(f"{formatted_name}: {value}")

        return "\n".join(result) + "\n"

    def format_infobox_npc(self, params: Dict[str, str]) -> str:
        """Format Infobox NPC templates with comprehensive labeling"""
        result = []
        result.append("=== NPC Information ===")

        # NPC-specific parameter mappings
        npc_mappings = {
            'name': 'NPC Name',
            'image': 'NPC Image',
            'release': 'Release Date',
            'update': 'Update',
            'members': 'Members Only',
            'race': 'Race',
            'quest': 'Quest NPC',
            'location': 'Location',
            'shop': 'Shop',
            'gender': 'Gender',
            'examine': 'Examine Text',
            'map': 'Map Location',
            'id': 'NPC ID',
            'leagueregion': 'League Region'
        }

        # Process all parameters with proper labels
        for param, value in params.items():
            param_lower = param.lower().strip()

            if not value or value.strip() == '':
                continue

            if param_lower in npc_mappings:
                label = npc_mappings[param_lower]
                result.append(f"{label}: {value}")
            else:
                formatted_name = param.replace('_', ' ').title()
                result.append(f"{formatted_name}: {value}")

        return "\n".join(result)

    def format_infobox_location(self, params: Dict[str, str]) -> str:
        """Format Infobox Location templates with comprehensive labeling"""
        result = []
        result.append("=== Location Information ===")

        # Location-specific parameter mappings
        location_mappings = {
            'name': 'Location Name',
            'image': 'Location Image',
            'release': 'Release Date',
            'update': 'Update',
            'members': 'Members Only',
            'kingdom': 'Kingdom',
            'region': 'Region',
            'type': 'Location Type',
            'inhabitants': 'Inhabitants',
            'main music': 'Main Music',
            'map': 'Map',
            'teleports': 'Teleports',
            'leagueregion': 'League Region'
        }

        # Process all parameters with proper labels
        for param, value in params.items():
            param_lower = param.lower().strip()

            if not value or value.strip() == '':
                continue

            if param_lower in location_mappings:
                label = location_mappings[param_lower]
                result.append(f"{label}: {value}")
            else:
                formatted_name = param.replace('_', ' ').title()
                result.append(f"{formatted_name}: {value}")

        return "\n".join(result)

    def format_infobox_skill(self, params: Dict[str, str]) -> str:
        """Format Infobox Skill templates with comprehensive labeling"""
        result = []
        result.append("=== Skill Information ===")

        # Skill-specific parameter mappings
        skill_mappings = {
            'name': 'Skill Name',
            'image': 'Skill Image',
            'release': 'Release Date',
            'update': 'Update',
            'members': 'Members Only',
            'type': 'Skill Type',
            'attribute': 'Attribute',
            'influences': 'Influences',
            'tools': 'Tools',
            'table': 'Training Table',
            'leagueregion': 'League Region'
        }

        # Process all parameters with proper labels
        for param, value in params.items():
            param_lower = param.lower().strip()

            if not value or value.strip() == '':
                continue

            if param_lower in skill_mappings:
                label = skill_mappings[param_lower]
                result.append(f"{label}: {value}")
            else:
                formatted_name = param.replace('_', ' ').title()
                result.append(f"{formatted_name}: {value}")

        return "\n".join(result)

    def format_quest_details(self, params: Dict[str, str]) -> str:
        """Format Quest details templates with comprehensive labeling"""
        result = []
        result.append("=== Quest Details ===")

        # Quest details parameter mappings
        quest_detail_mappings = {
            'start': 'Start Point',
            'startmap': 'Start Map Location',
            'difficulty': 'Difficulty',
            'description': 'Description',
            'length': 'Length',
            'requirements': 'Requirements',
            'items': 'Items Required',
            'recommended': 'Recommended',
            'kills': 'Enemies to Defeat',
            'ironman': 'Ironman Concerns',
            'leagueregion': 'League Region'
        }

        # Process all parameters with proper labels
        for param, value in params.items():
            param_lower = param.lower().strip()

            if not value or value.strip() == '':
                continue

            if param_lower in quest_detail_mappings:
                label = quest_detail_mappings[param_lower]
                result.append(f"{label}: {value}")
            else:
                formatted_name = param.replace('_', ' ').title()
                result.append(f"{formatted_name}: {value}")

        return "\n".join(result)

    def format_skill_requirement(self, params: Dict[str, str]) -> str:
        """Format SCP (Skill/Combat/Prayer) requirement templates"""
        skill_names = {
            'attack': 'Attack',
            'strength': 'Strength',
            'defence': 'Defence',
            'ranged': 'Ranged',
            'prayer': 'Prayer',
            'magic': 'Magic',
            'runecraft': 'Runecraft',
            'construction': 'Construction',
            'hitpoints': 'Hitpoints',
            'agility': 'Agility',
            'herblore': 'Herblore',
            'thieving': 'Thieving',
            'crafting': 'Crafting',
            'fletching': 'Fletching',
            'slayer': 'Slayer',
            'hunter': 'Hunter',
            'mining': 'Mining',
            'smithing': 'Smithing',
            'fishing': 'Fishing',
            'cooking': 'Cooking',
            'firemaking': 'Firemaking',
            'woodcutting': 'Woodcutting',
            'farming': 'Farming',
            'combat': 'Combat',
            'quest': 'Quest Points'
        }

        # Extract skill and level from parameters
        if '1' in params and '2' in params:
            skill_key = params['1'].lower()
            level = params['2']
            skill_name = skill_names.get(skill_key, params['1'].title())
            if skill_key in ['combat', 'quest']:
                return f"{skill_name}: {level}"
            else:
                return f"{skill_name} Level: {level}"
        elif len(params) >= 2:
            # Try to get first two parameters
            keys = list(params.keys())
            if len(keys) >= 2:
                skill_key = params[keys[0]].lower()
                level = params[keys[1]]
                skill_name = skill_names.get(skill_key, params[keys[0]].title())
                return f"{skill_name} Level: {level}"

        # Handle unnamed parameters (positional)
        param_values = list(params.values())
        if len(param_values) >= 2:
            skill_key = param_values[0].lower()
            level = param_values[1]
            skill_name = skill_names.get(skill_key, param_values[0].title())
            return f"{skill_name} Level: {level}"

        return "Skill Requirement"

    def format_quest_requirement_start(self, params: Dict[str, str]) -> str:
        """Format quest requirement start templates"""
        if 'yes' in params or '1' in params:
            return "(Required to Start)"
        return ""

    def format_boostable_requirement(self, params: Dict[str, str]) -> str:
        """Format boostable requirement templates"""
        if 'yes' in params or '1' in params:
            return "(Boostable)"
        elif 'no' in params or '0' in params:
            return "(Not Boostable)"
        return ""

    def format_combat_stats(self, params: Dict[str, str]) -> str:
        """Format combat stats template"""
        stats = []
        
        for param, value in params.items():
            param_lower = param.lower()
            if param_lower in self.combat_stats:
                stat_name = self.combat_stats[param_lower]
                stats.append(f"{stat_name}: {value}")
        
        if stats:
            return "Combat Stats: " + ", ".join(stats)
        return ""

    def format_drops_line(self, params: Dict[str, str]) -> str:
        """Format drop table line templates with comprehensive labeling"""
        result = []

        # Enhanced drop line parameter mappings
        drop_mappings = {
            'name': 'Item Name',
            'quantity': 'Quantity',
            'rarity': 'Drop Rate',
            'raritynotes': 'Drop Rate Notes',
            'rolls': 'Rolls Per Kill',
            'gemw': 'Grand Exchange',
            'leagueregion': 'League Region',
            'members': 'Members Only',
            'noted': 'Noted',
            'stackable': 'Stackable'
        }

        # Process all parameters with proper labels
        for param, value in params.items():
            param_lower = param.lower().strip()

            # Skip empty values
            if not value or value.strip() == '':
                continue

            # Get proper label
            if param_lower in drop_mappings:
                label = drop_mappings[param_lower]
                result.append(f"{label}: {value}")
            else:
                # Fallback: capitalize and format parameter name
                formatted_name = param.replace('_', ' ').title()
                result.append(f"{formatted_name}: {value}")

        return " | ".join(result) if result else "[Drop Entry]"

    def format_drops_table(self, template_name: str, params: Dict[str, str]) -> str:
        """Format drop table templates"""
        if 'head' in template_name.lower():
            return "=== Drop Table ==="
        elif 'bottom' in template_name.lower():
            return ""
        return "[Drop Table]"

    def format_recipe(self, params: Dict[str, str]) -> str:
        """Format recipe/creation templates"""
        result = []

        # Skill requirements
        for i in range(1, 4):
            skill_key = f'skill{i}'
            level_key = f'skill{i}lvl'
            if skill_key in params and level_key in params:
                skill = params[skill_key].title()
                level = params[level_key]
                result.append(f"Requires: {level} {skill}")

        # Materials
        materials = []
        for i in range(1, 10):
            mat_key = f'mat{i}'
            if mat_key in params:
                materials.append(params[mat_key])

        if materials:
            result.append(f"Materials: {', '.join(materials)}")

        # Output
        if 'output1' in params:
            result.append(f"Creates: {params['output1']}")

        return " | ".join(result) if result else "[Recipe]"

    def format_location_line(self, params: Dict[str, str]) -> str:
        """Format location line templates"""
        result = []

        if 'name' in params:
            result.append(f"Location: {params['name']}")
        if 'location' in params:
            result.append(f"Area: {params['location']}")
        if 'levels' in params:
            result.append(f"Level: {params['levels']}")

        return " | ".join(result) if result else "[Location]"

    def format_emote_clue(self, params: Dict[str, str]) -> str:
        """Format emote clue templates"""
        result = []

        if 'tier' in params:
            result.append(f"Clue Tier: {params['tier'].title()}")
        if 'emote' in params:
            result.append(f"Emote: {params['emote']}")
        if 'location' in params:
            result.append(f"Location: {params['location']}")

        # Required items
        items = []
        for i in range(1, 6):
            item_key = f'item{i}'
            if item_key in params:
                items.append(params[item_key])

        if items:
            result.append(f"Required Items: {', '.join(items)}")

        return " | ".join(result) if result else "[Emote Clue]"

    def format_navigation_template(self, template_name: str, params: Dict[str, str]) -> str:
        """Format navigation templates like External, Otheruses, etc."""
        if template_name.lower() == 'external':
            return "[External Link]"
        elif template_name.lower() == 'otheruses':
            return "[See Also]"
        elif template_name.lower() == 'confuse':
            return "[Disambiguation]"
        return f"[{template_name}]"

    def format_subject_changes(self, params: Dict[str, str]) -> str:
        """Format subject changes templates"""
        result = []

        if 'date' in params:
            result.append(f"Update Date: {params['date']}")
        if 'update' in params:
            result.append(f"Update: {params['update']}")
        if 'change' in params:
            change_text = params['change'][:100] + "..." if len(params['change']) > 100 else params['change']
            result.append(f"Change: {change_text}")

        return " | ".join(result) if result else "[Update History]"

    def format_map_template(self, params: Dict[str, str]) -> str:
        """Format map templates"""
        result = []

        if 'name' in params:
            result.append(f"Map: {params['name']}")
        if 'x' in params and 'y' in params:
            result.append(f"Coordinates: ({params['x']}, {params['y']})")

        return " | ".join(result) if result else "[Map]"

    def format_relative_location(self, params: Dict[str, str]) -> str:
        """Format relative location templates"""
        result = []

        if 'location' in params:
            result.append(f"Location: {params['location']}")

        directions = ['north', 'south', 'east', 'west']
        for direction in directions:
            if direction in params:
                result.append(f"{direction.title()}: {params[direction]}")

        return " | ".join(result) if result else "[Relative Location]"

    def format_boostable(self, params: Dict[str, str]) -> str:
        """Format boostable skill templates"""
        if params.get('1') == 'yes' or params.get('yes'):
            return "[Boostable]"
        return "[Not Boostable]"

    def format_transcript(self, params: Dict[str, str]) -> str:
        """Format transcript templates"""
        if 'npc' in params or any('npc' in key.lower() for key in params.keys()):
            return "[NPC Dialogue Available]"
        return "[Transcript Available]"

    def format_thieving_info(self, params: Dict[str, str]) -> str:
        """Format thieving info templates"""
        result = []

        if 'name' in params:
            result.append(f"Thieving Target: {params['name']}")
        if 'level' in params:
            result.append(f"Required Level: {params['level']}")
        if 'xp' in params:
            result.append(f"Experience: {params['xp']}")

        return " | ".join(result) if result else "[Thieving Info]"

    def format_skilling_chart(self, params: Dict[str, str]) -> str:
        """Format skilling success chart templates"""
        return "[Success Rate Chart]"

    def format_equipment_template(self, params: Dict[str, str]) -> str:
        """Format equipment display templates"""
        return "[Equipment Setup]"

    def format_inventory_template(self, params: Dict[str, str]) -> str:
        """Format inventory display templates"""
        return "[Inventory Setup]"

    def format_combat_styles(self, params: Dict[str, str]) -> str:
        """Format combat styles templates"""
        result = []

        if 'speed' in params:
            result.append(f"Attack Speed: {params['speed']}")
        if 'attackrange' in params:
            result.append(f"Attack Range: {params['attackrange']}")

        return " | ".join(result) if result else "[Combat Styles]"

    def format_floor_number(self, params: Dict[str, str]) -> str:
        """Format floor number templates"""
        if 'uk' in params:
            floor_num = params['uk']
            if floor_num == '0':
                return "Ground Floor"
            elif floor_num == '1':
                return "1st Floor"
            elif floor_num == '2':
                return "2nd Floor"
            elif floor_num == '3':
                return "3rd Floor"
            else:
                return f"{floor_num}th Floor"
        return "[Floor]"

    def format_stash_location(self, params: Dict[str, str]) -> str:
        """Format STASH location templates"""
        result = []

        if 'x' in params and 'y' in params:
            result.append(f"STASH Location: ({params['x']}, {params['y']})")
        if 'text' in params:
            result.append(f"Description: {params['text']}")

        return " | ".join(result) if result else "[STASH Unit Location]"

    def format_page_link(self, params: Dict[str, str]) -> str:
        """Format page link templates (plink)"""
        # Debug: print params to see what we're getting
        # print(f"DEBUG plink params: {params}")

        # Get the first unnamed parameter (page name)
        page_name = None
        for key in ['1', '0']:  # Try both 1 and 0 as first parameter
            if key in params and params[key]:
                page_name = params[key]
                break

        if not page_name:
            # Try to find any non-pic parameter
            for key, value in params.items():
                if value and not key.startswith('pic') and key not in ['txt', 'name']:
                    page_name = value
                    break

        if not page_name:
            return "[Link]"

        # Add display text if different
        if 'txt' in params and params['txt'] != page_name:
            return f"[[{page_name}|{params['txt']}]]"
        else:
            return f"[[{page_name}]]"

    def format_footnote(self, params: Dict[str, str]) -> str:
        """Format footnote templates (efn)"""
        # Debug: print params to see what we're getting
        # print(f"DEBUG efn params: {params}")

        # Try different parameter keys
        for key in ['1', '0']:
            if key in params and params[key]:
                return f"[Note: {params[key]}]"

        # If no numbered parameter, try to get any text parameter
        for key, value in params.items():
            if value and len(value) > 3:  # Assume longer values are the footnote text
                return f"[Note: {value}]"

        return "[Footnote]"

    def format_skill_calculator(self, params: Dict[str, str]) -> str:
        """Format skill calculator templates (SCP)"""
        for key in ['1', '0']:
            if key in params and params[key]:
                return f"[Calculator: {params[key]}]"
        return "[Skill Calculator]"

    def format_redirect(self, params: Dict[str, str]) -> str:
        """Format redirect templates (RE)"""
        for key in ['1', '0']:
            if key in params and params[key]:
                return f"[See: {params[key]}]"
        return "[Redirect]"

    def format_average_drop(self, params: Dict[str, str]) -> str:
        """Format average drop value templates"""
        return "[Average Drop Value]"

    def format_pronunciation(self, params: Dict[str, str]) -> str:
        """Format pronunciation templates (IPAc-en)"""
        return "[Pronunciation Guide]"

    def format_citation(self, params: Dict[str, str]) -> str:
        """Format citation templates"""
        if 'user' in params:
            return f"[Twitter: @{params['user']}]"
        return "[Citation]"

    def format_quest_requirement(self, params: Dict[str, str]) -> str:
        """Format quest requirement templates"""
        return "[Quest Requirement]"

    def format_fairy_code(self, params: Dict[str, str]) -> str:
        """Format fairy ring code templates"""
        if '1' in params:
            return f"[Fairy Ring: {params['1']}]"
        return "[Fairy Ring Code]"

    def format_uses_materials(self, params: Dict[str, str]) -> str:
        """Format uses material list templates"""
        return "=== Creation Materials ==="

    def format_drop_sources(self, params: Dict[str, str]) -> str:
        """Format drop sources templates"""
        return "=== Drop Sources ==="

    def format_store_locations(self, params: Dict[str, str]) -> str:
        """Format store locations templates"""
        return "=== Shop Locations ==="

    def format_recommended_equipment(self, params: Dict[str, str]) -> str:
        """Format recommended equipment templates"""
        return "=== Recommended Equipment ==="

    def format_sherlock_challenge(self, params: Dict[str, str]) -> str:
        """Format Sherlock challenge templates"""
        if 'skill' in params:
            return f"Sherlock Challenge: {params['skill']}"
        return "Sherlock Elite Clue Challenge"

    def format_falo_item(self, params: Dict[str, str]) -> str:
        """Format Falo item templates"""
        if 'item' in params:
            return f"Falo Item: {params['item']}"
        return "Falo Master Clue Item"

    def format_references(self, params: Dict[str, str]) -> str:
        """Format references/reflist templates"""
        return "=== References ==="

    def format_main_article(self, params: Dict[str, str]) -> str:
        """Format main article templates"""
        if '1' in params:
            return f"Main Article: {params['1']}"
        return "See Main Article"

    def format_money_making(self, params: Dict[str, str]) -> str:
        """Format money making guide templates"""
        return "=== Money Making Guide ==="

    def format_sync(self, params: Dict[str, str]) -> str:
        """Format sync templates"""
        return "Content synchronized with main wiki"

    def format_calculator(self, params: Dict[str, str]) -> str:
        """Format calculator templates"""
        return "Calculator Available"

    def format_skill_guide(self, params: Dict[str, str]) -> str:
        """Format skill guide templates"""
        return "Skill Guide Available"

    def format_audio(self, params: Dict[str, str]) -> str:
        """Format audio/listen templates"""
        if 'title' in params:
            return f"Audio: {params['title']}"
        return "Audio Available"

    def format_instance(self, params: Dict[str, str]) -> str:
        """Format instance templates"""
        return "Instanced Area"

    def format_strategy(self, params: Dict[str, str]) -> str:
        """Format strategy templates"""
        return "Strategy Guide Available"

    def format_drop_log(self, params: Dict[str, str]) -> str:
        """Format drop log project templates"""
        return "Drop Log Data Available"

    def format_drop_value(self, params: Dict[str, str]) -> str:
        """Format average drop value templates"""
        if 'value' in params:
            return f"Average Drop Value: {params['value']}"
        return "Drop Value Calculated"

    def format_respell(self, params: Dict[str, str]) -> str:
        """Format respell templates"""
        if '1' in params and '2' in params:
            return f"Also known as: {params['2']}"
        return ""

    def format_changes_header(self, params: Dict[str, str]) -> str:
        """Format subject changes header templates"""
        return "=== Update History ==="

    def format_coins(self, params: Dict[str, str]) -> str:
        """Format coins templates"""
        if '1' in params:
            return f"{params['1']} coins"
        return "coins"

    def format_skill_link(self, params: Dict[str, str]) -> str:
        """Format skill link templates"""
        if '1' in params:
            return f"Skill: {params['1']}"
        return "Skill"

    def format_boss_link(self, params: Dict[str, str]) -> str:
        """Format boss link templates"""
        if '1' in params:
            return f"Boss: {params['1']}"
        return "Boss"

    def format_equipment_category(self, template_name: str, params: Dict[str, str]) -> str:
        """Format equipment category templates"""
        category = template_name.replace('_', ' ').title()
        return f"=== {category} ==="

    def format_diary_skill_stats(self, params: Dict[str, str]) -> str:
        """Format DiarySkillStats template - ORIGINAL FORMAT"""
        formatted_stats = []
        for key, value in params.items():
            if key and value and not key.startswith('_'):
                # Clean up skill names
                skill_name = key.replace('_', ' ').title()
                formatted_stats.append(f"{skill_name}: {value}")

        if formatted_stats:
            return f"Skill Requirements: {', '.join(formatted_stats)}"
        return "Skill Requirements"

    def format_item_spawn_table(self, template_name: str, params: Dict[str, str]) -> str:
        """Format ItemSpawn templates - ORIGINAL FORMAT"""
        if template_name.lower() == 'itemspawntablehead':
            return "=== Item Spawn Locations ==="
        elif template_name.lower() == 'itemspawnline':
            parts = []
            if 'name' in params:
                parts.append(f"Item: {params['name']}")
            if 'location' in params:
                parts.append(f"Location: {params['location']}")
            if 'members' in params:
                parts.append(f"Members: {params['members']}")
            return " | ".join(parts) if parts else "Item Spawn"
        elif template_name.lower() == 'itemspawntablebottom':
            if 'respawn timer' in params:
                return f"Respawn Timer: {params['respawn timer']}"
            return ""
        return f"Item Spawn: {template_name}"

    def format_generic_template(self, template_name: str, params: Dict[str, str]) -> str:
        """Format generic templates with comprehensive parameter labeling"""
        if not params:
            return f"{template_name}"

        # Common parameter mappings for various templates
        common_mappings = {
            # Numeric parameters that need context
            '1': 'Parameter 1',
            '2': 'Parameter 2',
            '3': 'Parameter 3',
            '4': 'Parameter 4',
            '5': 'Parameter 5',

            # Common template parameters
            'name': 'Name',
            'id': 'ID',
            'value': 'Value',
            'cost': 'Cost',
            'level': 'Level',
            'xp': 'Experience',
            'skill': 'Skill',
            'requirement': 'Requirement',
            'location': 'Location',
            'members': 'Members Only',
            'quest': 'Quest',
            'item': 'Item',
            'quantity': 'Quantity',
            'rarity': 'Rarity',
            'type': 'Type',
            'category': 'Category',
            'description': 'Description',
            'image': 'Image',
            'caption': 'Caption',
            'url': 'URL',
            'date': 'Date',
            'author': 'Author',
            'source': 'Source',
            'page': 'Page',
            'section': 'Section',
            'version': 'Version',
            'update': 'Update',
            'release': 'Release Date',
            'examine': 'Examine Text',
            'weight': 'Weight',
            'tradeable': 'Tradeable',
            'stackable': 'Stackable',
            'noteable': 'Noteable',
            'destroy': 'Destroy Text',
            'alch': 'Alchemy Value',
            'store': 'Store Price',
            'exchange': 'Exchange Price',

            # Combat related
            'combat': 'Combat Level',
            'hitpoints': 'Hitpoints',
            'attack': 'Attack',
            'strength': 'Strength',
            'defence': 'Defence',
            'ranged': 'Ranged',
            'magic': 'Magic',
            'prayer': 'Prayer',
            'slayer': 'Slayer',
            'damage': 'Damage',
            'accuracy': 'Accuracy',
            'speed': 'Speed',
            'range': 'Range',
            'style': 'Style',

            # Drop/loot related
            'drop': 'Drop',
            'loot': 'Loot',
            'reward': 'Reward',
            'chance': 'Chance',
            'rate': 'Rate',
            'table': 'Table',
            'tier': 'Tier',
            'roll': 'Roll',
            'rolls': 'Rolls',

            # Location/map related
            'x': 'X Coordinate',
            'y': 'Y Coordinate',
            'z': 'Z Coordinate',
            'plane': 'Plane',
            'region': 'Region',
            'area': 'Area',
            'floor': 'Floor',
            'room': 'Room',

            # Time related
            'time': 'Time',
            'duration': 'Duration',
            'cooldown': 'Cooldown',
            'respawn': 'Respawn Time',
            'timer': 'Timer',

            # Skill related
            'farming': 'Farming',
            'cooking': 'Cooking',
            'fishing': 'Fishing',
            'mining': 'Mining',
            'woodcutting': 'Woodcutting',
            'smithing': 'Smithing',
            'crafting': 'Crafting',
            'fletching': 'Fletching',
            'runecraft': 'Runecraft',
            'construction': 'Construction',
            'agility': 'Agility',
            'herblore': 'Herblore',
            'thieving': 'Thieving',
            'firemaking': 'Firemaking',
            'hunter': 'Hunter',
        }

        # Format parameters with proper labels
        formatted_params = []
        for key, value in params.items():
            if not value or key.startswith('_'):
                continue

            key_lower = key.lower().strip()

            # Get proper label
            if key_lower in common_mappings:
                label = common_mappings[key_lower]
                formatted_params.append(f"{label}: {value}")
            else:
                # Fallback: capitalize and format parameter name
                formatted_name = key.replace('_', ' ').title()
                formatted_params.append(f"{formatted_name}: {value}")

        if formatted_params:
            return f"{template_name}: {', '.join(formatted_params)}"
        return f"{template_name}"

    def process_wiki_content(self, content: str) -> str:
        """Process entire wiki content, parsing all templates"""
        # Find all templates - FIXED for nested templates
        def find_templates(text):
            """Find all complete templates, handling nesting"""
            templates = []
            i = 0
            while i < len(text):
                if text[i:i+2] == '{{':
                    # Found start of template
                    start = i
                    brace_count = 0
                    j = i
                    while j < len(text):
                        if text[j:j+2] == '{{':
                            brace_count += 1
                            j += 2
                        elif text[j:j+2] == '}}':
                            brace_count -= 1
                            j += 2
                            if brace_count == 0:
                                # Found complete template
                                template = text[start:j]
                                templates.append((start, j, template))
                                i = j
                                break
                        else:
                            j += 1
                    else:
                        # No closing braces found
                        i += 1
                else:
                    i += 1
            return templates

        # Find and replace templates
        templates = find_templates(content)
        processed_content = content

        # Process templates in reverse order to maintain positions
        for start, end, template_text in reversed(templates):
            try:
                parsed = self.parse_template(template_text)
                processed_content = processed_content[:start] + parsed + processed_content[end:]
            except Exception as e:
                # If parsing fails, return simplified version
                simplified = f"[Template: {template_text[:50]}...]"
                processed_content = processed_content[:start] + simplified + processed_content[end:]

        # Clean up remaining wiki markup
        processed_content = self.clean_wiki_markup(processed_content)

        return processed_content

    def clean_wiki_markup(self, content: str) -> str:
        """Clean remaining wiki markup while preserving structure and fixing HTML entities"""
        import html

        # Decode HTML entities first
        content = html.unescape(content)

        # Fix common HTML entities that might remain
        content = content.replace('&nbsp;', ' ')
        content = content.replace('&#160;', ' ')
        content = content.replace('&amp;', '&')
        content = content.replace('&lt;', '<')
        content = content.replace('&gt;', '>')
        content = content.replace('&quot;', '"')
        content = content.replace('&#39;', "'")

        # Clean wiki markup but make it readable
        content = re.sub(r'\[\[([^|\]]+)\|([^\]]+)\]\]', r'\2', content)  # [[link|text]] -> text
        content = re.sub(r'\[\[([^\]]+)\]\]', r'\1', content)  # [[link]] -> link
        content = re.sub(r"'''([^']+)'''", r'\1', content)  # Bold
        content = re.sub(r"''([^']+)''", r'\1', content)  # Italic
        content = re.sub(r'==+\s*([^=]+)\s*==+', r'\n=== \1 ===\n', content)  # Headers

        # Add proper line breaks for better readability
        content = re.sub(r'(===\s*[^=]+\s*===)([A-Z])', r'\1\n\2', content)  # Line break after headers
        content = re.sub(r'(\|[^|]+\|[^|]+\|[^|]+)', r'\1\n', content)  # Line break after drop entries
        content = re.sub(r'([a-z]:)\s*([0-9]+)([A-Z])', r'\1 \2\n\3', content)  # Line break between stats

        # Clean up whitespace and formatting
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)  # Multiple newlines -> double newline
        content = re.sub(r' +', ' ', content)  # Multiple spaces -> single space
        content = re.sub(r'^\s+|\s+$', '', content, flags=re.MULTILINE)  # Trim line whitespace

        # Remove remaining template artifacts
        content = re.sub(r'\{\{[^}]*\}\}', '', content)  # Any remaining templates
        content = re.sub(r'<[^>]+>', '', content)  # HTML tags

        # Fix common formatting issues
        content = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', content)  # Ensure space after sentence endings
        content = re.sub(r'([a-z])([A-Z])', r'\1 \2', content)  # Add space between camelCase words

        return content.strip()

    # ═══════════════════════════════════════════════════════════════════════════════
    # ULTRA-AGGRESSIVE PARALLEL PROCESSING - UNCAPPED WORKERS!
    # ═══════════════════════════════════════════════════════════════════════════════

    def process_pages_parallel_uncapped(self, pages: List[Dict]) -> List[Dict]:
        """Process multiple pages with UNCAPPED parallel workers - NO LIMITS!"""
        if not pages:
            return []

        total_pages = len(pages)
        print(f"🚀 Processing {total_pages:,} pages with UNCAPPED parallel workers...")
        print(f"   🎯 Starting with {min(4, self.max_workers)} workers, scaling up to {self.max_workers:,}+")

        processed_pages = []
        start_time = time.time()
        current_workers = min(4, self.max_workers)  # Start small

        # Dynamic batch processing with INSANE scaling
        batch_size = max(10, len(pages) // 100)  # Adaptive batch size
        processed_count = 0

        for i in range(0, len(pages), batch_size):
            batch = pages[i:i + batch_size]
            batch_start_time = time.time()

            # INSANE DYNAMIC WORKER SCALING
            if i > 0:
                elapsed = time.time() - start_time
                avg_time_per_batch = elapsed / (i // batch_size + 1)
                memory_usage = self.get_memory_usage()

                # SMART SCALING - Use discovered OS limits
                effective_max_workers = self.max_workers
                if self.discovered_max_workers is not None:
                    # Use 75% of discovered limit as safe maximum
                    effective_max_workers = int(self.discovered_max_workers * 0.75)

                if avg_time_per_batch < 0.2 and memory_usage < 0.5 and current_workers < effective_max_workers:
                    # BLAZING fast - scale INSANELY
                    current_workers = min(current_workers * 4, effective_max_workers)
                    print(f"\r🚀🚀🚀 INSANE scaling UP to {current_workers:,} workers (blazing speed!)")
                elif avg_time_per_batch < 0.5 and memory_usage < 0.6 and current_workers < effective_max_workers:
                    # Very fast - triple workers
                    current_workers = min(current_workers * 3, effective_max_workers)
                    print(f"\r🚀🚀 ULTRA scaling UP to {current_workers:,} workers (ultra fast)")
                elif avg_time_per_batch < 1.0 and memory_usage < 0.7 and current_workers < effective_max_workers:
                    # Fast - double workers
                    current_workers = min(current_workers * 2, effective_max_workers)
                    print(f"\r🚀 Scaling UP to {current_workers:,} workers (fast processing)")
                elif avg_time_per_batch < 2.0 and memory_usage < 0.8 and current_workers < effective_max_workers // 2:
                    # Good - add significant workers
                    current_workers = min(current_workers + 256, effective_max_workers // 2)
                elif memory_usage > 0.95:
                    current_workers = max(current_workers // 4, 16)
                    print(f"\r🚨 EMERGENCY scaling DOWN to {current_workers:,} workers (critical memory)")
                elif avg_time_per_batch > 5.0:
                    current_workers = max(current_workers // 2, 32)
                    print(f"\r⚠️  Scaling DOWN to {current_workers:,} workers (slow processing)")

            # Process batch with SAFE worker count - handle OS limits gracefully
            try:
                with ProcessPoolExecutor(max_workers=current_workers) as executor:
                    future = executor.submit(self.process_batch_worker, batch)
                    try:
                        batch_results = future.result(timeout=120)  # Longer timeout for heavy processing
                        processed_pages.extend(batch_results)
                        processed_count += len(batch_results)
                    except Exception as e:
                        print(f"\r⚠️  Batch processing error: {e}")
                        continue
            except OSError as e:
                if "Invalid argument" in str(e) or "Too many open files" in str(e):
                    # Hit OS limit! Scale back dramatically and remember this limit
                    print(f"\r🚨 Hit OS limit at {current_workers:,} workers! Scaling back...")
                    current_workers = max(current_workers // 4, 64)  # Scale back to 25% or minimum 64
                    self.discovered_max_workers = current_workers  # Remember this limit
                    print(f"   🔧 Reduced to {current_workers:,} workers (OS limit discovered)")

                    # Retry with reduced workers
                    try:
                        with ProcessPoolExecutor(max_workers=current_workers) as executor:
                            future = executor.submit(self.process_batch_worker, batch)
                            batch_results = future.result(timeout=120)
                            processed_pages.extend(batch_results)
                            processed_count += len(batch_results)
                    except Exception as retry_e:
                        print(f"\r⚠️  Retry failed: {retry_e}")
                        continue
                else:
                    print(f"\r⚠️  Unexpected OS error: {e}")
                    continue

            # REAL-TIME PROGRESS BAR
            progress = (processed_count / total_pages) * 100
            elapsed = time.time() - start_time
            rate = processed_count / elapsed if elapsed > 0 else 0
            eta = (total_pages - processed_count) / rate if rate > 0 else 0
            batch_time = time.time() - batch_start_time

            # Progress bar visualization
            bar_length = 40
            filled_length = int(bar_length * progress / 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)

            print(f"\r🔥 Processing |{bar}| {progress:.1f}% | {processed_count:,}/{total_pages:,} | "
                  f"ETA: {eta:.0f}s | {rate:.1f}/s | Workers: {current_workers:,} | Batch: {batch_time:.1f}s", end='')

        print()  # New line after progress bar

        total_time = time.time() - start_time
        final_rate = processed_count / total_time if total_time > 0 else 0

        print(f"🔥 UNCAPPED processing complete: {processed_count:,} pages in {total_time:.1f}s ({final_rate:.1f}/s)")
        print(f"   🚀 Peak workers: {current_workers:,}")
        print(f"   ⚡ Performance boost: {final_rate/100:.1f}x baseline")

        return processed_pages

    def process_batch_worker(self, page_batch: List[Dict]) -> List[Dict]:
        """Process a batch of pages (runs in parallel process)"""
        results = []

        for page in page_batch:
            try:
                title = page.get('title', '')
                raw_wikitext = page.get('rawWikitext', page.get('text', ''))

                if not raw_wikitext:
                    continue

                # Process templates from raw wikitext
                processed_text = self.process_wiki_content(raw_wikitext)

                # Create processed page
                processed_page = {
                    'title': title,
                    'text': processed_text,
                    'rawWikitext': raw_wikitext,
                    'categories': page.get('categories', []),
                    'timestamp': page.get('timestamp', ''),
                    'revid': page.get('revid', ''),
                    'processed': True,
                    'processing_timestamp': time.time()
                }

                results.append(processed_page)

            except Exception as e:
                print(f"   ⚠️  Error processing {page.get('title', 'unknown')}: {e}")
                continue

        return results

    def get_memory_usage(self) -> float:
        """Get current memory usage as percentage"""
        try:
            import psutil
            return psutil.virtual_memory().percent / 100.0
        except ImportError:
            return 0.5  # Conservative estimate if can't determine

def test_parser():
    """Test the template parser with OSRS examples"""
    parser = OSRSWikiTemplateParser()
    
    # Test combat stats template
    combat_template = "{{Infobox Monster|combat=91|att=85|def=65|str=80|hitpoints=100}}"
    print("Combat Template:")
    print(parser.parse_template(combat_template))
    print()
    
    # Test item template
    item_template = "{{Infobox Item|attack=40|strength=50|defence=1|astab=+65|aslash=+55}}"
    print("Item Template:")
    print(parser.parse_template(item_template))
    print()
    
    # Test full content
    full_content = """
    The {{Infobox Monster|name=Dragon|combat=91|att=85|def=65|str=80|hitpoints=100}} is a powerful creature.
    
    It requires {{Combat Stats|attack=40|strength=50|defence=1}} to defeat effectively.
    """
    print("Full Content:")
    print(parser.process_wiki_content(full_content))

if __name__ == "__main__":
    test_parser()
