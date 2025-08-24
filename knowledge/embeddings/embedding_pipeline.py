"""
Complete embedding generation and storage pipeline.
"""

import time
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from ..extraction.entity_types import CodeEntity
from ..extraction.storage_integration import EntityStorage
from .embedding_generator import EmbeddingGenerator, EmbeddingResult
from .embedding_storage import EmbeddingStorage


class EmbeddingPipeline:
    """
    Complete pipeline for generating and storing embeddings for code entities.
    
    Combines entity retrieval, embedding generation, and storage in a unified interface.
    """
    
    def __init__(self, 
                 db_path: str = "knowledge/chromadb/entities.duckdb",
                 embedding_dim: int = 384,
                 model_path: Optional[str] = None):
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self.model_path = model_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.entity_storage = EntityStorage(db_path)
        self.embedding_generator = EmbeddingGenerator(
            embedding_dim=embedding_dim,
            model_path=model_path
        )
        self.embedding_storage = EmbeddingStorage(db_path)
        
        # Pipeline metrics
        self.metrics = {
            'entities_processed': 0,
            'embeddings_generated': 0,
            'embeddings_stored': 0,
            'total_pipeline_time': 0.0,
            'errors': 0
        }
    
    def process_entities_by_file(self, file_path: str, 
                               force_regenerate: bool = False) -> Dict[str, Any]:
        """Process all entities from a specific file."""
        start_time = time.time()
        
        try:
            # Get entities from file
            entities = self.entity_storage.get_entities_by_file(file_path)
            
            if not entities:
                return {
                    'file_path': file_path,
                    'entities_processed': 0,
                    'embeddings_generated': 0,
                    'embeddings_stored': 0,
                    'processing_time': time.time() - start_time,
                    'success': False,
                    'message': 'No entities found for file'
                }
            
            # Process entities
            result = self._process_entities(entities, force_regenerate)
            
            # Update file-specific metrics
            result.update({
                'file_path': file_path,
                'processing_time': time.time() - start_time,
                'success': True
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing entities for file {file_path}: {e}")
            self.metrics['errors'] += 1
            
            return {
                'file_path': file_path,
                'entities_processed': 0,
                'embeddings_generated': 0,
                'embeddings_stored': 0,
                'processing_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def process_entities_by_type(self, entity_type: str,
                               limit: int = 1000,
                               force_regenerate: bool = False) -> Dict[str, Any]:
        """Process entities of a specific type."""
        start_time = time.time()
        
        try:
            # Get entities by type
            entities = self.entity_storage.get_entities_by_type(entity_type, limit)
            
            if not entities:
                return {
                    'entity_type': entity_type,
                    'entities_processed': 0,
                    'embeddings_generated': 0,
                    'embeddings_stored': 0,
                    'processing_time': time.time() - start_time,
                    'success': False,
                    'message': f'No entities found for type {entity_type}'
                }
            
            # Process entities
            result = self._process_entities(entities, force_regenerate)
            
            # Update type-specific metrics
            result.update({
                'entity_type': entity_type,
                'processing_time': time.time() - start_time,
                'success': True
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing entities for type {entity_type}: {e}")
            self.metrics['errors'] += 1
            
            return {
                'entity_type': entity_type,
                'entities_processed': 0,
                'embeddings_generated': 0,
                'embeddings_stored': 0,
                'processing_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def process_all_entities(self, batch_size: int = 100,
                           force_regenerate: bool = False) -> Dict[str, Any]:
        """Process all entities in the database."""
        start_time = time.time()
        
        try:
            # Get all entities with basic statistics
            stats = self.entity_storage.get_file_statistics()
            total_entities = stats.get('total_entities', 0)
            
            if total_entities == 0:
                return {
                    'entities_processed': 0,
                    'embeddings_generated': 0,
                    'embeddings_stored': 0,
                    'processing_time': time.time() - start_time,
                    'success': False,
                    'message': 'No entities found in database'
                }
            
            self.logger.info(f"Processing {total_entities} entities in batches of {batch_size}")
            
            # Process by file to maintain context
            file_results = []
            total_processed = 0
            total_generated = 0
            total_stored = 0
            
            # Get unique file paths
            # This is a simplified approach - in production, we'd want pagination
            try:
                conn = self.entity_storage._get_connection()
                file_paths = conn.execute("""
                    SELECT DISTINCT file_path FROM entities 
                    ORDER BY file_path LIMIT 1000
                """).fetchall()
                
                for (file_path,) in file_paths:
                    file_result = self.process_entities_by_file(file_path, force_regenerate)
                    file_results.append(file_result)
                    
                    if file_result['success']:
                        total_processed += file_result['entities_processed']
                        total_generated += file_result['embeddings_generated']
                        total_stored += file_result['embeddings_stored']
                    
                    # Progress logging
                    if len(file_results) % 10 == 0:
                        self.logger.info(f"Processed {len(file_results)} files, "
                                       f"{total_processed} entities so far")
            
            except Exception as e:
                self.logger.error(f"Error during batch processing: {e}")
                raise
            
            processing_time = time.time() - start_time
            
            return {
                'entities_processed': total_processed,
                'embeddings_generated': total_generated,
                'embeddings_stored': total_stored,
                'files_processed': len([r for r in file_results if r['success']]),
                'files_failed': len([r for r in file_results if not r['success']]),
                'processing_time': processing_time,
                'avg_time_per_entity': processing_time / total_processed if total_processed > 0 else 0,
                'success': True,
                'file_results': file_results
            }
            
        except Exception as e:
            self.logger.error(f"Error processing all entities: {e}")
            self.metrics['errors'] += 1
            
            return {
                'entities_processed': 0,
                'embeddings_generated': 0,
                'embeddings_stored': 0,
                'processing_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    def _process_entities(self, entities: List[CodeEntity], 
                         force_regenerate: bool = False) -> Dict[str, Any]:
        """Internal method to process a list of entities."""
        
        # Filter entities that need processing
        entities_to_process = []
        
        if force_regenerate:
            entities_to_process = entities
        else:
            # Check which entities already have embeddings
            for entity in entities:
                existing_embedding = self.embedding_storage.get_entity_embedding(str(entity.id))
                if existing_embedding is None:
                    entities_to_process.append(entity)
        
        if not entities_to_process:
            return {
                'entities_processed': len(entities),
                'embeddings_generated': 0,
                'embeddings_stored': 0,
                'message': 'All entities already have embeddings'
            }
        
        # Generate embeddings
        self.logger.info(f"Generating embeddings for {len(entities_to_process)} entities")
        
        embedding_results = self.embedding_generator.generate_embeddings_batch(
            entities_to_process,
            batch_size=50,  # Smaller batches for memory management
            use_cache=True
        )
        
        # Store embeddings
        storage_result = self.embedding_storage.store_embeddings(
            embedding_results,
            update_mode='upsert'
        )
        
        # Update metrics
        self.metrics['entities_processed'] += len(entities)
        self.metrics['embeddings_generated'] += len(embedding_results)
        self.metrics['embeddings_stored'] += storage_result['inserted'] + storage_result['updated']
        
        return {
            'entities_processed': len(entities),
            'embeddings_generated': len(embedding_results),
            'embeddings_stored': storage_result['inserted'] + storage_result['updated'],
            'storage_details': storage_result
        }
    
    def fit_embedding_model(self, sample_size: int = 1000, 
                           save_model: bool = True) -> Dict[str, Any]:
        """
        Fit the embedding model on a sample of entities from the database.
        
        Args:
            sample_size: Number of entities to use for fitting
            save_model: Whether to save the fitted model
            
        Returns:
            Dictionary with fitting results
        """
        start_time = time.time()
        
        try:
            # Get a diverse sample of entities for training
            sample_entities = self._get_training_sample(sample_size)
            
            if not sample_entities:
                return {
                    'success': False,
                    'message': 'No entities available for model fitting',
                    'fitting_time': time.time() - start_time
                }
            
            # Fit the model
            model_save_path = None
            if save_model:
                model_save_path = f"knowledge/models/embedding_model_{int(time.time())}.json"
                Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
            
            self.embedding_generator.fit_model(sample_entities, model_save_path)
            
            fitting_time = time.time() - start_time
            
            return {
                'success': True,
                'entities_used_for_fitting': len(sample_entities),
                'model_info': self.embedding_generator.model.get_model_info(),
                'model_save_path': model_save_path,
                'fitting_time': fitting_time
            }
            
        except Exception as e:
            self.logger.error(f"Error fitting embedding model: {e}")
            return {
                'success': False,
                'error': str(e),
                'fitting_time': time.time() - start_time
            }
    
    def _get_training_sample(self, sample_size: int) -> List[CodeEntity]:
        """Get a diverse sample of entities for model training."""
        
        sample_entities = []
        
        try:
            # Get entities from different types for diversity
            entity_types = ['function', 'class', 'module', 'variable', 'constant']
            entities_per_type = max(1, sample_size // len(entity_types))
            
            for entity_type in entity_types:
                type_entities = self.entity_storage.get_entities_by_type(
                    entity_type, 
                    limit=entities_per_type
                )
                sample_entities.extend(type_entities)
                
                if len(sample_entities) >= sample_size:
                    break
            
            # Limit to requested sample size
            return sample_entities[:sample_size]
            
        except Exception as e:
            self.logger.error(f"Error getting training sample: {e}")
            return []
    
    def search_similar_entities(self, query_entity_id: str,
                               limit: int = 10,
                               threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for entities similar to a given entity."""
        
        try:
            # Get the query entity's embedding
            query_embedding = self.embedding_storage.get_entity_embedding(query_entity_id)
            
            if query_embedding is None:
                return []
            
            # Find similar entities
            similar_entities = self.embedding_storage.find_similar_entities(
                query_embedding=query_embedding,
                limit=limit,
                threshold=threshold,
                exclude_entity_id=query_entity_id
            )
            
            return similar_entities
            
        except Exception as e:
            self.logger.error(f"Error searching similar entities: {e}")
            return []
    
    def search_by_text(self, query_text: str,
                      limit: int = 10,
                      entity_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search entities by text query using embeddings."""
        
        try:
            # Generate embedding for query text
            from ..extraction.entity_types import CodeEntity, EntityType
            from uuid import uuid4
            
            # Create a temporary entity for the query
            temp_entity = CodeEntity(
                id=uuid4(),
                type=EntityType.FUNCTION,  # Default type
                name="query",
                metadata={'query_text': query_text}
            )
            
            # Generate embedding for query
            result = self.embedding_generator.generate_embedding(temp_entity)
            query_embedding = result.embedding
            
            # Search for similar entities
            similar_entities = self.embedding_storage.find_similar_entities(
                query_embedding=query_embedding,
                limit=limit,
                threshold=0.5,  # Lower threshold for text queries
                entity_types=entity_types
            )
            
            return similar_entities
            
        except Exception as e:
            self.logger.error(f"Error searching by text: {e}")
            return []
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for the entire pipeline."""
        
        # Get component metrics
        generator_metrics = self.embedding_generator.get_metrics()
        storage_stats = self.embedding_storage.get_embedding_statistics()
        
        # Combine with pipeline metrics
        combined_metrics = {
            'pipeline': dict(self.metrics),
            'embedding_generation': generator_metrics,
            'embedding_storage': storage_stats,
            'database_path': self.db_path,
            'embedding_dimension': self.embedding_dim
        }
        
        # Calculate derived metrics
        if self.metrics['entities_processed'] > 0:
            combined_metrics['pipeline']['avg_processing_time'] = (
                self.metrics['total_pipeline_time'] / self.metrics['entities_processed']
            )
            combined_metrics['pipeline']['success_rate'] = 1.0 - (
                self.metrics['errors'] / self.metrics['entities_processed']
            )
        
        return combined_metrics
    
    def reset_metrics(self):
        """Reset pipeline metrics."""
        self.metrics = {
            'entities_processed': 0,
            'embeddings_generated': 0,
            'embeddings_stored': 0,
            'total_pipeline_time': 0.0,
            'errors': 0
        }
        
        self.embedding_generator.reset_metrics()
    
    def close(self):
        """Clean shutdown of pipeline."""
        self.entity_storage.close()
        self.embedding_storage.close()