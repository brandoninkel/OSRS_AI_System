#!/usr/bin/env python3
"""
OSRS RAG Embedding Service - Hybrid RAG/RAFT System
Target Directory: /final/OSRS_AI/AI/hybrid-system/rag-pipeline/embeddings/

This service handles text embedding using Ollama with optimized models for OSRS wiki content.
Supports both mxbai-embed-large and nomic-embed-text models.

Usage:
    from embedding_service import EmbeddingService
    service = EmbeddingService()
    embeddings = service.embed_texts(["General Graardor is a boss", "Bandos armor drops"])
"""

import requests
import json
import numpy as np
from typing import List, Dict, Optional, Union
import logging
import time
from dataclasses import dataclass
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    model_name: str = "mxbai-embed-large:latest"  # Optimal for OSRS wiki content
    ollama_url: str = "http://localhost:11434"
    batch_size: int = 64  # Optimized for Mac Mini M4 Pro 24GB RAM
    max_retries: int = 3
    timeout: int = 45  # Increased for larger batches
    cache_embeddings: bool = True
    # Mac M4 Pro optimizations
    max_concurrent_requests: int = 8
    memory_efficient_batching: bool = True

class EmbeddingService:
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.embedding_cache = {}
        self.session = None
        
        # Verify Ollama is running and model is available
        self._verify_setup()
    
    def _verify_setup(self):
        """Verify Ollama is running and embedding model is available"""
        try:
            response = requests.get(f"{self.config.ollama_url}/api/tags", timeout=10)
            if response.status_code != 200:
                raise ConnectionError(f"Cannot connect to Ollama at {self.config.ollama_url}")
            
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            if self.config.model_name not in model_names:
                logger.warning(f"Model {self.config.model_name} not found. Available models: {model_names}")
                logger.info(f"Attempting to pull {self.config.model_name}...")
                self._pull_model()
            
            logger.info(f"✅ Embedding service initialized with {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to verify Ollama setup: {e}")
            raise
    
    def _pull_model(self):
        """Pull the embedding model if not available"""
        try:
            response = requests.post(
                f"{self.config.ollama_url}/api/pull",
                json={"name": self.config.model_name},
                timeout=300  # 5 minutes for model download
            )
            
            if response.status_code == 200:
                logger.info(f"✅ Successfully pulled {self.config.model_name}")
            else:
                raise Exception(f"Failed to pull model: {response.text}")
                
        except Exception as e:
            logger.error(f"❌ Failed to pull model {self.config.model_name}: {e}")
            raise
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(f"{self.config.model_name}:{text}".encode()).hexdigest()
    
    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text using Ollama
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        if not text.strip():
            return []
        
        # Check cache first
        cache_key = self._get_cache_key(text)
        if self.config.cache_embeddings and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    f"{self.config.ollama_url}/api/embeddings",
                    json={
                        "model": self.config.model_name,
                        "prompt": text
                    },
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    embedding = response.json()["embedding"]
                    
                    # Cache the result
                    if self.config.cache_embeddings:
                        self.embedding_cache[cache_key] = embedding
                    
                    return embedding
                else:
                    logger.warning(f"Embedding request failed (attempt {attempt + 1}): {response.text}")
                    
            except Exception as e:
                logger.warning(f"Embedding error (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        logger.error(f"Failed to embed text after {self.config.max_retries} attempts")
        return []
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts with batching
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            batch_embeddings = []
            
            logger.info(f"Processing embedding batch {i//self.config.batch_size + 1}/{(len(texts)-1)//self.config.batch_size + 1}")
            
            for text in batch:
                embedding = self.embed_text(text)
                batch_embeddings.append(embedding)
            
            embeddings.extend(batch_embeddings)
            
            # Small delay between batches to avoid overwhelming Ollama
            if i + self.config.batch_size < len(texts):
                time.sleep(0.1)
        
        return embeddings
    
    async def embed_texts_async(self, texts: List[str], use_batch_api: bool = True) -> List[List[float]]:
        """
        Asynchronously embed multiple texts for better performance

        Args:
            texts: List of texts to embed
            use_batch_api: If True, use Ollama's batch /api/embed endpoint (MUCH faster)
                          If False, use individual requests with concurrency

        Returns:
            List of embeddings
        """
        if not texts:
            return []

        async with aiohttp.ClientSession() as session:
            self.session = session

            if use_batch_api:
                # Use Ollama's native batch embedding API - sends multiple texts in one request
                # This is MUCH faster than individual requests (10-50x speedup)
                return await self._embed_texts_batch(texts, session)
            else:
                # Original approach: individual requests with concurrency control
                max_conc = max(1, int(getattr(self.config, 'max_concurrent_requests', 8)))
                semaphore = asyncio.Semaphore(max_conc)
                tasks = [self._embed_text_async(text, semaphore) for text in texts]
                embeddings = await asyncio.gather(*tasks)
                return embeddings

    async def _embed_texts_batch(self, texts: List[str], session: aiohttp.ClientSession) -> List[List[float]]:
        """Use Ollama's batch embedding API to embed multiple texts in one request

        This is MUCH faster than sending individual requests because:
        1. Single HTTP request instead of N requests
        2. Ollama can optimize batch processing internally
        3. Reduces network overhead
        4. Can achieve 10-50x speedup depending on batch size
        """
        if not texts:
            return []

        # Filter out empty texts and track their positions
        text_to_index = {}
        filtered_texts = []
        for i, text in enumerate(texts):
            if text.strip():
                text_to_index[len(filtered_texts)] = i
                filtered_texts.append(text)

        if not filtered_texts:
            return [[] for _ in texts]

        # Check cache for all texts
        uncached_indices = []
        uncached_texts = []
        results = [None] * len(filtered_texts)

        for i, text in enumerate(filtered_texts):
            cache_key = self._get_cache_key(text)
            if self.config.cache_embeddings and cache_key in self.embedding_cache:
                results[i] = self.embedding_cache[cache_key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # If all cached, return immediately
        if not uncached_texts:
            # Map back to original positions
            final_results = [[] for _ in texts]
            for filtered_idx, original_idx in text_to_index.items():
                final_results[original_idx] = results[filtered_idx]
            return final_results

        # Send batch request to Ollama
        for attempt in range(self.config.max_retries):
            try:
                async with session.post(
                    f"{self.config.ollama_url}/api/embed",  # Note: /api/embed not /api/embeddings
                    json={
                        "model": self.config.model_name,
                        "input": uncached_texts  # Send array of texts
                    },
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout * 2)  # Longer timeout for batches
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        embeddings = data["embeddings"]

                        # Cache and store results
                        for i, embedding in zip(uncached_indices, embeddings):
                            results[i] = embedding
                            if self.config.cache_embeddings:
                                cache_key = self._get_cache_key(filtered_texts[i])
                                self.embedding_cache[cache_key] = embedding

                        # Map back to original positions
                        final_results = [[] for _ in texts]
                        for filtered_idx, original_idx in text_to_index.items():
                            final_results[original_idx] = results[filtered_idx]

                        return final_results
                    else:
                        error_text = await response.text()
                        logger.warning(f"Batch embedding attempt {attempt + 1} failed: {response.status} - {error_text}")

            except Exception as e:
                logger.warning(f"Batch embedding attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

        # If all retries failed, fall back to individual requests
        logger.warning("Batch embedding failed, falling back to individual requests")
        return await self.embed_texts_async(texts, use_batch_api=False)

    async def _embed_text_async(self, text: str, semaphore: asyncio.Semaphore) -> List[float]:
        """Async version of embed_text"""
        if not text.strip():
            return []
        
        # Check cache first
        cache_key = self._get_cache_key(text)
        if self.config.cache_embeddings and cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        async with semaphore:
            for attempt in range(self.config.max_retries):
                try:
                    async with self.session.post(
                        f"{self.config.ollama_url}/api/embeddings",
                        json={
                            "model": self.config.model_name,
                            "prompt": text
                        },
                        timeout=self.config.timeout
                    ) as response:
                        
                        if response.status == 200:
                            data = await response.json()
                            embedding = data["embedding"]
                            
                            # Cache the result
                            if self.config.cache_embeddings:
                                self.embedding_cache[cache_key] = embedding
                            
                            return embedding
                        else:
                            error_text = await response.text()
                            logger.warning(f"Async embedding request failed (attempt {attempt + 1}): {error_text}")
                            
                except Exception as e:
                    logger.warning(f"Async embedding error (attempt {attempt + 1}): {e}")
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
            logger.error(f"Failed to embed text after {self.config.max_retries} attempts")
            return []
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings from this model"""
        test_embedding = self.embed_text("test")
        return len(test_embedding) if test_embedding else 0
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self.embedding_cache.clear()
        logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.embedding_cache),
            "model": self.config.model_name
        }

# Test function
def test_embedding_service():
    """Test the embedding service"""
    service = EmbeddingService()
    
    test_texts = [
        "General Graardor is the leader of Bandos' forces in the God Wars Dungeon",
        "Bandos chestplate is a piece of armor that provides excellent defence bonuses",
        "Money making guide for killing dragons involves high combat stats"
    ]
    
    print("Testing single embedding...")
    embedding = service.embed_text(test_texts[0])
    print(f"Embedding dimension: {len(embedding)}")
    
    print("Testing batch embeddings...")
    embeddings = service.embed_texts(test_texts)
    print(f"Generated {len(embeddings)} embeddings")
    
    print("Cache stats:", service.get_cache_stats())

if __name__ == "__main__":
    test_embedding_service()
