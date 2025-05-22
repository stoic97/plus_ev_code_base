"""
Hardware Abstraction Layer for Trading System

This module provides hardware abstraction to support both GPU and CPU processing,
with graceful fallback mechanisms when GPU operations fail.
"""

import logging
import numpy as np
import time
import warnings
from typing import Union, Any, Dict, Optional, Tuple

# Set up logging
logger = logging.getLogger(__name__)

class HardwareManager:
    """
    Hardware abstraction layer that manages GPU/CPU execution and provides
    graceful fallback mechanisms.
    
    Attributes:
        force_cpu (bool): Force CPU execution even if GPU is available
        gpu_available (bool): Whether GPU is detected and usable
        current_device (str): Current computation device ('gpu' or 'cpu')
        array_lib: Reference to array library (numpy or cupy)
        memory_pool: Reference to GPU memory pool if using GPU
        device_id (int): GPU device ID to use
        _peak_memory_usage (int): Track peak GPU memory usage
        _memory_threshold (float): Fraction of GPU memory to use before cleanup
    """
    
    def __init__(self, force_cpu: bool = False, device_id: int = 0, 
                 memory_threshold: float = 0.8):
        """
        Initialize the hardware manager.
        
        Args:
            force_cpu: If True, use CPU even if GPU is available
            device_id: GPU device ID to use
            memory_threshold: Fraction of GPU memory that can be used before cleanup
        """
        self.force_cpu = force_cpu
        self.device_id = device_id
        self.gpu_available = not force_cpu and self._check_gpu_availability()
        self.current_device = 'cpu' if force_cpu or not self.gpu_available else 'gpu'
        self.array_lib = self._get_array_lib()
        self._peak_memory_usage = 0
        self._memory_threshold = memory_threshold
        
        # Set up memory pool if using GPU
        self.memory_pool = None
        self.pinned_pool = None
        if self.current_device == 'gpu':
            self._setup_memory_pools()
            logger.info(f"Using GPU (device {device_id}) with memory threshold {memory_threshold}")
        else:
            logger.info("Using CPU for computation")
    
    def _check_gpu_availability(self) -> bool:
        """
        Check if GPU is available and usable.
        
        Returns:
            bool: True if GPU is available, False otherwise
        """
        try:
            import cupy as cp
            # Try a simple operation to verify GPU works
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Try to use the specified device
                with cp.cuda.Device(self.device_id):
                    test_array = cp.array([1, 2, 3])
                    _ = cp.sum(test_array)
                    logger.info(f"GPU is available (device {self.device_id})")
                    # Get device info
                    device_props = cp.cuda.runtime.getDeviceProperties(self.device_id)
                    logger.info(f"GPU: {device_props['name'].decode()}")
                    mem_info = cp.cuda.runtime.memGetInfo()
                    logger.info(f"GPU Memory: {mem_info[1]/1024**3:.2f} GB total, "
                                f"{mem_info[0]/1024**3:.2f} GB free")
                    return True
        except ImportError:
            logger.warning("CuPy not installed. GPU acceleration unavailable.")
            return False
        except Exception as e:
            logger.warning(f"GPU initialization failed: {str(e)}")
            return False
    
    def _get_array_lib(self):
        """
        Get the appropriate array library (cupy or numpy).
        
        Returns:
            module: cupy if using GPU, numpy if using CPU
        """
        if self.current_device == 'gpu':
            import cupy as cp
            return cp
        else:
            return np
    
    def _setup_memory_pools(self):
        """Set up memory pools for efficient GPU memory management"""
        if self.current_device == 'gpu':
            import cupy as cp
            self.memory_pool = cp.get_default_memory_pool()
            self.pinned_pool = cp.get_default_pinned_memory_pool()
            logger.debug("GPU memory pools initialized")
    
    def to_device(self, array: Any) -> Any:
        """
        Move array to the current computation device.
        
        Args:
            array: Input array (numpy or cupy)
            
        Returns:
            array on the appropriate device
        """
        if array is None:
            return None
            
        # If we're on CPU, ensure it's a numpy array
        if self.current_device == 'cpu':
            if hasattr(array, 'get'):  # It's a cupy array
                return array.get()
            return array  # Already a numpy array or other type
        
        # We're on GPU
        xp = self.array_lib
        try:
            # Handle numpy arrays -> move to GPU
            if isinstance(array, np.ndarray):
                return xp.asarray(array)
            # Handle lists and scalars
            elif isinstance(array, (list, float, int)):
                return xp.asarray(array)
            # Handle dictionaries of arrays
            elif isinstance(array, dict):
                return {k: self.to_device(v) for k, v in array.items()}
            # Already a cupy array or unsupported type
            return array
        except Exception as e:
            logger.warning(f"Error moving data to GPU: {str(e)}. Falling back to CPU.")
            self.fallback_to_cpu()
            return array if isinstance(array, np.ndarray) else np.asarray(array)
    
    def to_cpu(self, array: Any) -> Any:
        """
        Move array to CPU if it's on GPU.
        
        Args:
            array: Input array (numpy or cupy)
            
        Returns:
            numpy array on CPU
        """
        if array is None:
            return None
            
        # Already on CPU or not an array type
        if not hasattr(array, 'get'):
            return array
            
        # Convert from GPU to CPU
        try:
            return array.get()
        except Exception as e:
            logger.warning(f"Error moving data to CPU: {str(e)}")
            return array  # Return as is if conversion fails
    
    def fallback_to_cpu(self) -> bool:
        """
        Force fallback to CPU mode when GPU operations fail.
        
        Returns:
            bool: True if fallback occurred, False if already on CPU
        """
        if self.current_device == 'gpu':
            logger.warning("Falling back to CPU processing")
            self.current_device = 'cpu'
            self.array_lib = np
            self._cleanup_gpu_memory()
            return True
        return False
    
    def _cleanup_gpu_memory(self):
        """
        Clean up GPU memory by releasing memory pools.
        """
        if hasattr(self, 'memory_pool') and self.memory_pool:
            try:
                self.memory_pool.free_all_blocks()
                if self.pinned_pool:
                    self.pinned_pool.free_all_blocks()
                logger.debug("GPU memory cleaned up")
            except Exception as e:
                logger.warning(f"Error cleaning up GPU memory: {str(e)}")
    
    def monitor_memory_usage(self) -> Dict[str, float]:
        """
        Monitor current GPU memory usage and perform cleanup if needed.
        
        Returns:
            Dict with memory usage information
        """
        if self.current_device != 'gpu':
            return {'device': 'cpu', 'memory_used': 0, 'memory_total': 0}
            
        try:
            import cupy as cp
            free, total = cp.cuda.runtime.memGetInfo()
            used = total - free
            used_mb = used / (1024 * 1024)
            total_mb = total / (1024 * 1024)
            
            # Update peak memory usage
            self._peak_memory_usage = max(self._peak_memory_usage, used)
            
            # Check if we need to clean up
            if used > self._memory_threshold * total:
                logger.warning(f"GPU memory usage high ({used_mb:.2f}/{total_mb:.2f} MB). Cleaning up.")
                self._cleanup_gpu_memory()
                
            return {
                'device': 'gpu',
                'memory_used': used_mb,
                'memory_total': total_mb,
                'memory_utilization': used / total,
                'peak_memory_mb': self._peak_memory_usage / (1024 * 1024)
            }
        except Exception as e:
            logger.warning(f"Error monitoring GPU memory: {str(e)}")
            return {'device': 'gpu', 'error': str(e)}
    
    def with_device_context(self, func, *args, **kwargs):
        """
        Execute a function within the appropriate device context, 
        with fallback to CPU if GPU fails.
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of function execution
        """
        if self.current_device == 'cpu':
            return func(*args, **kwargs)
        
        try:
            # Execute on GPU
            import cupy as cp
            with cp.cuda.Device(self.device_id):
                return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"GPU execution failed: {str(e)}. Falling back to CPU.")
            self.fallback_to_cpu()
            return func(*args, **kwargs)  # Retry on CPU
    
    def timed_execution(self, func, *args, **kwargs) -> Tuple[Any, float]:
        """
        Execute a function and time its execution.
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Tuple of (result, execution_time_seconds)
        """
        start_time = time.time()
        result = self.with_device_context(func, *args, **kwargs)
        execution_time = time.time() - start_time
        return result, execution_time
    
    def __str__(self) -> str:
        """String representation of the hardware manager"""
        if self.current_device == 'gpu':
            mem_info = self.monitor_memory_usage()
            return (f"HardwareManager(device='gpu', device_id={self.device_id}, "
                   f"memory_used={mem_info['memory_used']:.2f}MB, "
                   f"memory_total={mem_info['memory_total']:.2f}MB)")
        else:
            return "HardwareManager(device='cpu')"

'''
# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Test the hardware manager
    hw = HardwareManager()
    
    # Create a test array
    test_array = np.random.randn(1000, 1000)
    
    # Move to device
    device_array = hw.to_device(test_array)
    
    # Perform operations
    def matrix_operation(arr):
        xp = hw.array_lib  # Use the appropriate array library
        return xp.dot(arr, arr.T)
    
    result, exec_time = hw.timed_execution(matrix_operation, device_array)
    
    # Move result back to CPU
    cpu_result = hw.to_cpu(result)
    
    print(f"Execution time: {exec_time:.4f} seconds on {hw.current_device}")
    print(f"Result shape: {cpu_result.shape}")
    
    # Monitor memory
    if hw.current_device == 'gpu':
        print(hw.monitor_memory_usage())
        '''