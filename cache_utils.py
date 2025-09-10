"""
Cache utilities and performance monitoring for the AI Study Companion
"""
import time
import streamlit as st
import psutil
import os
from typing import Dict, Any, Optional
from functools import wraps
import pickle
import hashlib

class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """End timing and return duration"""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            self.metrics[operation] = duration
            del self.start_times[operation]
            return duration
        return 0.0
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system performance stats"""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "available_memory_gb": psutil.virtual_memory().available / (1024**3),
                "process_memory_mb": psutil.Process().memory_info().rss / (1024**2)
            }
        except:
            return {"error": "Unable to get system stats"}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        cache_info = {
            "streamlit_cache_size": 0,
            "session_state_size": len(st.session_state),
            "cached_operations": list(self.metrics.keys()),
            "last_operation_times": self.metrics
        }
        
        # Try to get streamlit cache info
        try:
            if hasattr(st, '_cache_resource_api'):
                cache_info["resource_cache_enabled"] = True
            if hasattr(st, '_cache_data_api'):
                cache_info["data_cache_enabled"] = True
        except:
            pass
        
        return cache_info

class SmartCache:
    """Enhanced caching with TTL and memory management"""
    
    def __init__(self, max_size: int = 100, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = {}
        self.access_times = {}
        self.creation_times = {}
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate a cache key from function name and arguments"""
        key_data = (func_name, args, sorted(kwargs.items()))
        key_string = str(key_data)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_expired(self, key: str, ttl: Optional[int] = None) -> bool:
        """Check if a cache entry has expired"""
        if key not in self.creation_times:
            return True
        
        age = time.time() - self.creation_times[key]
        max_age = ttl if ttl is not None else self.default_ttl
        return age > max_age
    
    def _evict_lru(self):
        """Remove least recently used item"""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), 
                     key=lambda k: self.access_times[k])
        self._remove_key(lru_key)
    
    def _remove_key(self, key: str):
        """Remove a key from all cache structures"""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.creation_times.pop(key, None)
    
    def get(self, key: str, ttl: Optional[int] = None):
        """Get an item from cache"""
        if key in self.cache and not self._is_expired(key, ttl):
            self.access_times[key] = time.time()
            return self.cache[key]
        elif key in self.cache:
            # Expired, remove it
            self._remove_key(key)
        return None
    
    def set(self, key: str, value: Any):
        """Set an item in cache"""
        # Manage cache size
        while len(self.cache) >= self.max_size:
            self._evict_lru()
        
        current_time = time.time()
        self.cache[key] = value
        self.access_times[key] = current_time
        self.creation_times[key] = current_time
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.access_times.clear()
        self.creation_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        current_time = time.time()
        expired_count = sum(1 for key in self.creation_times.keys() 
                          if self._is_expired(key))
        
        return {
            "total_items": len(self.cache),
            "expired_items": expired_count,
            "max_size": self.max_size,
            "cache_hit_potential": len(self.cache) - expired_count,
            "memory_usage_estimate": sum(len(str(v)) for v in self.cache.values())
        }

def cache_with_ttl(ttl: int = 300):
    """Decorator for caching function results with TTL"""
    def decorator(func):
        cache = SmartCache(default_ttl=ttl)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = cache._generate_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            result = cache.get(key, ttl)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result
        
        # Add cache management methods
        wrapper.cache_clear = cache.clear
        wrapper.cache_stats = cache.get_stats
        wrapper._cache = cache
        
        return wrapper
    return decorator

def performance_timer(operation_name: str = None):
    """Decorator to time function execution"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            
            if 'performance_monitor' not in st.session_state:
                st.session_state.performance_monitor = PerformanceMonitor()
            
            monitor = st.session_state.performance_monitor
            monitor.start_timer(name)
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = monitor.end_timer(name)
                if st.checkbox("Show Performance Debug", key=f"perf_{name}"):
                    st.caption(f"⏱️ {name}: {duration:.3f}s")
        
        return wrapper
    return decorator

def memory_efficient_operation(max_memory_mb: int = 500):
    """Decorator to monitor memory usage during operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check memory before operation
            try:
                process = psutil.Process()
                memory_before = process.memory_info().rss / (1024**2)
                
                if memory_before > max_memory_mb:
                    st.warning(f"⚠️ High memory usage: {memory_before:.1f}MB")
                
                result = func(*args, **kwargs)
                
                memory_after = process.memory_info().rss / (1024**2)
                memory_diff = memory_after - memory_before
                
                if memory_diff > 50:  # More than 50MB increase
                    st.info(f"📊 Memory used: +{memory_diff:.1f}MB")
                
                return result
                
            except Exception as e:
                st.warning(f"Memory monitoring failed: {e}")
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

def create_performance_dashboard():
    """Create a performance monitoring dashboard"""
    if 'performance_monitor' not in st.session_state:
        st.session_state.performance_monitor = PerformanceMonitor()
    
    monitor = st.session_state.performance_monitor
    
    with st.expander("🔧 Performance Dashboard", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**System Stats**")
            sys_stats = monitor.get_system_stats()
            
            if "error" not in sys_stats:
                st.metric("CPU Usage", f"{sys_stats['cpu_percent']:.1f}%")
                st.metric("Memory Usage", f"{sys_stats['memory_percent']:.1f}%")
                st.metric("Process Memory", f"{sys_stats['process_memory_mb']:.1f}MB")
            else:
                st.warning("Unable to get system stats")
        
        with col2:
            st.markdown("**Cache Stats**")
            cache_stats = monitor.get_cache_stats()
            
            st.metric("Session State Items", cache_stats['session_state_size'])
            
            if cache_stats['last_operation_times']:
                st.markdown("**Recent Operations:**")
                for op, duration in cache_stats['last_operation_times'].items():
                    st.caption(f"• {op}: {duration:.3f}s")
        
        # Cache management buttons
        st.markdown("**Cache Management**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear Data Cache"):
                st.cache_data.clear()
                st.success("Data cache cleared!")
        
        with col2:
            if st.button("🗑️ Clear Resource Cache"):
                st.cache_resource.clear()
                st.success("Resource cache cleared!")
        
        with col3:
            if st.button("🔄 Reset Session"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.success("Session reset!")
                st.rerun()

# Context manager for timing operations
class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        
        # Store in session state for monitoring
        if 'operation_times' not in st.session_state:
            st.session_state.operation_times = {}
        
        st.session_state.operation_times[self.operation_name] = self.duration

# Utility functions for cache warming
def warm_up_caches(course_name: str):
    """Pre-load commonly used data into cache"""
    try:
        from rag_chain import check_course_status, _get_relevant_documents
        
        with Timer("cache_warmup"):
            # Check course status (this loads vectorstore)
            status = check_course_status(course_name)
            
            if status["is_ready"]:
                # Pre-load some common queries
                common_queries = [
                    "overview", "main concepts", "important topics",
                    "key points", "summary"
                ]
                
                for query in common_queries:
                    try:
                        _get_relevant_documents(course_name, query, k=3)
                    except:
                        pass  # Continue with other queries
        
        st.success("🚀 Caches warmed up successfully!")
        
    except Exception as e:
        st.warning(f"Cache warmup failed: {e}")

def optimize_for_performance():
    """Apply performance optimizations"""
    # Set streamlit config for better performance
    try:
        st.set_page_config(
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except:
        pass  # Config already set
    
    # Add performance tips
    if st.sidebar.checkbox("💡 Performance Tips", key="perf_tips"):
        st.sidebar.markdown("""
        **Performance Tips:**
        - 🚀 Caching speeds up repeated operations
        - 📚 Upload smaller, focused files
        - 🔄 Clear cache if memory is low
        - ⚡ Use specific topics for faster searches
        - 💾 Keep browser tabs to minimum
        """)

# Export commonly used functions
__all__ = [
    'PerformanceMonitor',
    'SmartCache', 
    'cache_with_ttl',
    'performance_timer',
    'memory_efficient_operation',
    'create_performance_dashboard',
    'Timer',
    'warm_up_caches',
    'optimize_for_performance'
]