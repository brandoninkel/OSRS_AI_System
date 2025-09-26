#!/usr/bin/env python3
"""
OSRS Knowledge Graph Query Service

Provides interface for exploring OSRS knowledge graph relationships,
entity connections, and semantic queries.
"""

import json
import logging
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import sqlite3

logger = logging.getLogger(__name__)

class OSRSKGQueryService:
    def __init__(self):
        """Initialize KG query service with trained artifacts"""
        self.repo_root = Path(__file__).resolve().parents[2]
        self.kg_model_dir = self.repo_root / "data" / "kg_model"
        
        # Load KG artifacts
        self.entity_to_id = None
        self.id_to_entity = None
        self.relation_to_id = None
        self.id_to_relation = None
        self.entity_embeddings = None
        self.triples_data = []
        
        self._load_kg_artifacts()
        self._load_triples_data()
        
    def _load_kg_artifacts(self):
        """Load trained KG model artifacts"""
        try:
            # Load entity mappings
            entity_map_path = self.kg_model_dir / "entity_to_id.json"
            if entity_map_path.exists():
                with open(entity_map_path, 'r', encoding='utf-8') as f:
                    self.entity_to_id = json.load(f)
                    
                # Create reverse mapping
                self.id_to_entity = [None] * (max(self.entity_to_id.values()) + 1)
                for entity, idx in self.entity_to_id.items():
                    if 0 <= idx < len(self.id_to_entity):
                        self.id_to_entity[idx] = entity
                        
            # Load relation mappings
            relation_map_path = self.kg_model_dir / "relation_to_id.json"
            if relation_map_path.exists():
                with open(relation_map_path, 'r', encoding='utf-8') as f:
                    self.relation_to_id = json.load(f)
                    
                # Create reverse mapping
                self.id_to_relation = [None] * (max(self.relation_to_id.values()) + 1)
                for relation, idx in self.relation_to_id.items():
                    if 0 <= idx < len(self.id_to_relation):
                        self.id_to_relation[idx] = relation
                        
            # Load entity embeddings
            emb_path = self.kg_model_dir / "entity_embeddings.npy"
            if emb_path.exists():
                self.entity_embeddings = np.load(emb_path)
                logger.info(f"Loaded KG embeddings: {self.entity_embeddings.shape}")
            else:
                logger.warning("No KG embeddings found")
                
        except Exception as e:
            logger.error(f"Failed to load KG artifacts: {e}")
            
    def _load_triples_data(self):
        """Load raw triples data for relationship queries"""
        try:
            triples_path = self.repo_root / "data" / "osrs_kg_triples.csv"
            if triples_path.exists():
                import csv
                with open(triples_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    self.triples_data = list(reader)
                logger.info(f"Loaded {len(self.triples_data)} triples")
            else:
                logger.warning("No triples data found")
        except Exception as e:
            logger.error(f"Failed to load triples data: {e}")
            
    def find_similar_entities(self, entity_name: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find entities similar to the given entity using embeddings"""
        if not self.entity_embeddings is not None or not self.entity_to_id:
            return []
            
        # Find entity ID
        entity_id = self.entity_to_id.get(entity_name)
        if entity_id is None:
            # Try fuzzy matching
            entity_name_lower = entity_name.lower()
            matches = [(name, idx) for name, idx in self.entity_to_id.items() 
                      if entity_name_lower in name.lower()]
            if matches:
                entity_name, entity_id = matches[0]
            else:
                return []
                
        if entity_id >= len(self.entity_embeddings):
            return []
            
        # Get entity embedding
        entity_emb = self.entity_embeddings[entity_id].reshape(1, -1)
        
        # Normalize embeddings
        entity_norm = entity_emb / np.linalg.norm(entity_emb)
        all_norms = self.entity_embeddings / np.linalg.norm(self.entity_embeddings, axis=1, keepdims=True)
        
        # Compute similarities
        similarities = np.dot(entity_norm, all_norms.T)[0]
        
        # Get top similar entities
        top_indices = np.argsort(similarities)[::-1][:top_k + 1]  # +1 to exclude self
        
        results = []
        for idx in top_indices:
            if idx != entity_id and idx < len(self.id_to_entity) and self.id_to_entity[idx]:
                similar_entity = self.id_to_entity[idx]
                score = float(similarities[idx])
                results.append((similar_entity, score))
                
        return results[:top_k]
        
    def find_entity_relationships(self, entity_name: str, relation_type: Optional[str] = None) -> List[Dict[str, str]]:
        """Find all relationships for an entity"""
        if not self.triples_data:
            return []
            
        relationships = []
        entity_name_lower = entity_name.lower()
        
        for triple in self.triples_data:
            head = triple.get('head', '')
            relation = triple.get('relation', '')
            tail = triple.get('tail', '')
            
            # Check if entity is head or tail
            if entity_name_lower in head.lower():
                if not relation_type or relation_type.lower() in relation.lower():
                    relationships.append({
                        'subject': head,
                        'predicate': relation,
                        'object': tail,
                        'direction': 'outgoing'
                    })
                    
            elif entity_name_lower in tail.lower():
                if not relation_type or relation_type.lower() in relation.lower():
                    relationships.append({
                        'subject': head,
                        'predicate': relation,
                        'object': tail,
                        'direction': 'incoming'
                    })
                    
        return relationships
        
    def explore_entity_neighborhood(self, entity_name: str, max_hops: int = 2) -> Dict[str, Any]:
        """Explore entity's neighborhood in the knowledge graph"""
        if not self.triples_data:
            return {}
            
        visited = set()
        current_level = {entity_name}
        neighborhood = {
            'center_entity': entity_name,
            'levels': [],
            'total_entities': 0,
            'total_relationships': 0
        }
        
        for hop in range(max_hops):
            next_level = set()
            level_relationships = []
            
            for entity in current_level:
                if entity in visited:
                    continue
                    
                visited.add(entity)
                relationships = self.find_entity_relationships(entity)
                
                for rel in relationships:
                    level_relationships.append(rel)
                    # Add connected entities to next level
                    if rel['direction'] == 'outgoing':
                        next_level.add(rel['object'])
                    else:
                        next_level.add(rel['subject'])
                        
            neighborhood['levels'].append({
                'hop': hop + 1,
                'entities': list(current_level),
                'relationships': level_relationships,
                'entity_count': len(current_level),
                'relationship_count': len(level_relationships)
            })
            
            current_level = next_level - visited
            if not current_level:
                break
                
        neighborhood['total_entities'] = len(visited)
        neighborhood['total_relationships'] = sum(len(level['relationships']) for level in neighborhood['levels'])
        
        return neighborhood
        
    def query_by_relation(self, relation_type: str, limit: int = 50) -> List[Dict[str, str]]:
        """Find all triples with a specific relation type"""
        if not self.triples_data:
            return []
            
        results = []
        relation_lower = relation_type.lower()
        
        for triple in self.triples_data:
            relation = triple.get('relation', '')
            if relation_lower in relation.lower():
                results.append({
                    'subject': triple.get('head', ''),
                    'predicate': relation,
                    'object': triple.get('tail', '')
                })
                
                if len(results) >= limit:
                    break
                    
        return results
        
    def get_kg_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        stats = {
            'entities_count': len(self.entity_to_id) if self.entity_to_id else 0,
            'relations_count': len(self.relation_to_id) if self.relation_to_id else 0,
            'triples_count': len(self.triples_data),
            'embeddings_loaded': self.entity_embeddings is not None,
            'embedding_dimension': self.entity_embeddings.shape[1] if self.entity_embeddings is not None else 0
        }
        
        if self.relation_to_id:
            stats['relation_types'] = list(self.relation_to_id.keys())[:20]  # Top 20 relations
            
        return stats
