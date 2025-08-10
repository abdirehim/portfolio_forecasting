"""Data caching functionality for the portfolio forecasting system."""

import pandas as pd
import numpy as np
import pickle
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timedelta
import logging

# Set up logging
logger = logging.getLogger(__name__)


class CacheManager:
    """Manages local caching of fetched data to reduce API calls."""
    
    def __init__(self, cache_dir: str = "data/cache", expiry_hours: int = 24):
        """Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            expiry_hours: Hours after which cache expires
        """
        self.cache_dir = Path(cache_dir)
        self.expiry_hours = expiry_hours
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different types of cache
        (self.cache_dir / "data").mkdir(exist_ok=True)
        (self.cache_dir / "metadata").mkdir(exist_ok=True)
        (self.cache_dir / "models").mkdir(exist_ok=True)
        
        logger.info(f"Cache manager initialized with directory: {self.cache_dir}")
    
    def _generate_cache_key(self, symbols: List[str], start_date: str, end_date: str) -> str:
        """Generate a unique cache key for the given parameters.
        
        Args:
            symbols: List of asset symbols
            start_date: Start date string
            end_date: End date string
            
        Returns:
            Unique cache key string
        """
        # Sort symbols to ensure consistent key generation
        sorted_symbols = sorted(symbols)
        key_string = f"{'-'.join(sorted_symbols)}_{start_date}_{end_date}"
        
        # Create hash for shorter, consistent key
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return f"data_{key_hash}"
    
    def _get_cache_filepath(self, cache_key: str, cache_type: str = "data") -> Path:
        """Get the full path for a cache file.
        
        Args:
            cache_key: Cache key
            cache_type: Type of cache (data, metadata, models)
            
        Returns:
            Path to cache file
        """
        return self.cache_dir / cache_type / f"{cache_key}.pkl"
    
    def _get_metadata_filepath(self, cache_key: str) -> Path:
        """Get the metadata file path for a cache key.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Path to metadata file
        """
        return self.cache_dir / "metadata" / f"{cache_key}_meta.json"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache is still valid (not expired).
        
        Args:
            cache_key: Cache key to check
            
        Returns:
            True if cache is valid, False otherwise
        """
        metadata_path = self._get_metadata_filepath(cache_key)
        
        if not metadata_path.exists():
            return False
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            cached_time = datetime.fromisoformat(metadata['cached_at'])
            expiry_time = cached_time + timedelta(hours=self.expiry_hours)
            
            is_valid = datetime.now() < expiry_time
            
            if not is_valid:
                logger.info(f"Cache {cache_key} has expired")
            
            return is_valid
            
        except Exception as e:
            logger.warning(f"Error checking cache validity for {cache_key}: {e}")
            return False
    
    def _save_metadata(self, cache_key: str, symbols: List[str], start_date: str, end_date: str, data_shape: tuple) -> None:
        """Save metadata for cached data.
        
        Args:
            cache_key: Cache key
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            data_shape: Shape of cached data
        """
        metadata = {
            'cache_key': cache_key,
            'symbols': symbols,
            'start_date': start_date,
            'end_date': end_date,
            'cached_at': datetime.now().isoformat(),
            'expiry_hours': self.expiry_hours,
            'data_shape': data_shape,
            'data_size_mb': self._get_cache_size_mb(cache_key)
        }
        
        metadata_path = self._get_metadata_filepath(cache_key)
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving metadata for {cache_key}: {e}")
    
    def _get_cache_size_mb(self, cache_key: str) -> float:
        """Get cache file size in MB.
        
        Args:
            cache_key: Cache key
            
        Returns:
            File size in MB
        """
        cache_path = self._get_cache_filepath(cache_key)
        
        try:
            if cache_path.exists():
                size_bytes = cache_path.stat().st_size
                return round(size_bytes / (1024 * 1024), 2)
        except Exception:
            pass
        
        return 0.0
    
    def get_cached_data(self, symbols: List[str], start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Retrieve cached data if available and valid.
        
        Args:
            symbols: List of asset symbols
            start_date: Start date string
            end_date: End date string
            
        Returns:
            Cached DataFrame if available, None otherwise
        """
        cache_key = self._generate_cache_key(symbols, start_date, end_date)
        cache_path = self._get_cache_filepath(cache_key)
        
        if not cache_path.exists():
            logger.debug(f"No cache found for key: {cache_key}")
            return None
        
        if not self._is_cache_valid(cache_key):
            logger.info(f"Cache expired for key: {cache_key}")
            self._remove_cache(cache_key)
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            
            logger.info(f"Successfully loaded cached data for {symbols} ({data.shape[0]} rows)")
            return data
            
        except Exception as e:
            logger.error(f"Error loading cached data for {cache_key}: {e}")
            self._remove_cache(cache_key)
            return None
    
    def save_data_to_cache(self, data: pd.DataFrame, symbols: List[str], start_date: str, end_date: str) -> bool:
        """Save data to cache.
        
        Args:
            data: DataFrame to cache
            symbols: List of asset symbols
            start_date: Start date string
            end_date: End date string
            
        Returns:
            True if successfully cached, False otherwise
        """
        if data.empty:
            logger.warning("Cannot cache empty DataFrame")
            return False
        
        cache_key = self._generate_cache_key(symbols, start_date, end_date)
        cache_path = self._get_cache_filepath(cache_key)
        
        try:
            # Save data
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Save metadata
            self._save_metadata(cache_key, symbols, start_date, end_date, data.shape)
            
            logger.info(f"Successfully cached data for {symbols} ({data.shape[0]} rows, {self._get_cache_size_mb(cache_key)} MB)")
            return True
            
        except Exception as e:
            logger.error(f"Error caching data for {cache_key}: {e}")
            return False
    
    def _remove_cache(self, cache_key: str) -> None:
        """Remove cache files for a given key.
        
        Args:
            cache_key: Cache key to remove
        """
        try:
            cache_path = self._get_cache_filepath(cache_key)
            metadata_path = self._get_metadata_filepath(cache_key)
            
            if cache_path.exists():
                cache_path.unlink()
            
            if metadata_path.exists():
                metadata_path.unlink()
                
            logger.debug(f"Removed cache for key: {cache_key}")
            
        except Exception as e:
            logger.warning(f"Error removing cache for {cache_key}: {e}")
    
    def clear_expired_cache(self) -> int:
        """Clear all expired cache files.
        
        Returns:
            Number of cache entries removed
        """
        removed_count = 0
        
        try:
            metadata_files = list((self.cache_dir / "metadata").glob("*_meta.json"))
            
            for metadata_file in metadata_files:
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    cache_key = metadata['cache_key']
                    
                    if not self._is_cache_valid(cache_key):
                        self._remove_cache(cache_key)
                        removed_count += 1
                        
                except Exception as e:
                    logger.warning(f"Error processing metadata file {metadata_file}: {e}")
                    continue
            
            if removed_count > 0:
                logger.info(f"Cleared {removed_count} expired cache entries")
            
        except Exception as e:
            logger.error(f"Error clearing expired cache: {e}")
        
        return removed_count
    
    def clear_all_cache(self) -> int:
        """Clear all cache files.
        
        Returns:
            Number of cache entries removed
        """
        removed_count = 0
        
        try:
            # Remove all data files
            data_files = list((self.cache_dir / "data").glob("*.pkl"))
            for file in data_files:
                file.unlink()
                removed_count += 1
            
            # Remove all metadata files
            metadata_files = list((self.cache_dir / "metadata").glob("*.json"))
            for file in metadata_files:
                file.unlink()
            
            logger.info(f"Cleared all cache ({removed_count} entries)")
            
        except Exception as e:
            logger.error(f"Error clearing all cache: {e}")
        
        return removed_count
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about current cache status.
        
        Returns:
            Dictionary with cache information
        """
        info = {
            'cache_dir': str(self.cache_dir),
            'expiry_hours': self.expiry_hours,
            'total_entries': 0,
            'valid_entries': 0,
            'expired_entries': 0,
            'total_size_mb': 0.0,
            'entries': []
        }
        
        try:
            metadata_files = list((self.cache_dir / "metadata").glob("*_meta.json"))
            info['total_entries'] = len(metadata_files)
            
            for metadata_file in metadata_files:
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    cache_key = metadata['cache_key']
                    is_valid = self._is_cache_valid(cache_key)
                    
                    if is_valid:
                        info['valid_entries'] += 1
                    else:
                        info['expired_entries'] += 1
                    
                    info['total_size_mb'] += metadata.get('data_size_mb', 0)
                    
                    entry_info = {
                        'cache_key': cache_key,
                        'symbols': metadata.get('symbols', []),
                        'date_range': f"{metadata.get('start_date')} to {metadata.get('end_date')}",
                        'cached_at': metadata.get('cached_at'),
                        'size_mb': metadata.get('data_size_mb', 0),
                        'is_valid': is_valid
                    }
                    
                    info['entries'].append(entry_info)
                    
                except Exception as e:
                    logger.warning(f"Error processing metadata file {metadata_file}: {e}")
                    continue
            
            info['total_size_mb'] = round(info['total_size_mb'], 2)
            
        except Exception as e:
            logger.error(f"Error getting cache info: {e}")
        
        return info
    
    def save_model_to_cache(self, model: Any, model_name: str, metadata: Optional[Dict] = None) -> bool:
        """Save a trained model to cache.
        
        Args:
            model: Model object to cache
            model_name: Name for the cached model
            metadata: Optional metadata dictionary
            
        Returns:
            True if successfully cached, False otherwise
        """
        cache_path = self.cache_dir / "models" / f"{model_name}.pkl"
        metadata_path = self.cache_dir / "models" / f"{model_name}_meta.json"
        
        try:
            # Save model
            with open(cache_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Save metadata
            model_metadata = {
                'model_name': model_name,
                'cached_at': datetime.now().isoformat(),
                'model_type': type(model).__name__,
                'size_mb': round(cache_path.stat().st_size / (1024 * 1024), 2)
            }
            
            if metadata:
                model_metadata.update(metadata)
            
            with open(metadata_path, 'w') as f:
                json.dump(model_metadata, f, indent=2)
            
            logger.info(f"Successfully cached model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error caching model {model_name}: {e}")
            return False
    
    def load_model_from_cache(self, model_name: str) -> Optional[Any]:
        """Load a cached model.
        
        Args:
            model_name: Name of the cached model
            
        Returns:
            Loaded model if available, None otherwise
        """
        cache_path = self.cache_dir / "models" / f"{model_name}.pkl"
        
        if not cache_path.exists():
            logger.debug(f"No cached model found: {model_name}")
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                model = pickle.load(f)
            
            logger.info(f"Successfully loaded cached model: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading cached model {model_name}: {e}")
            return None