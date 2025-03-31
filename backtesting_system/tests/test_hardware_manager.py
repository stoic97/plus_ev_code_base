"""
Test script for the HardwareManager to verify correct operation
with both GPU and CPU computation.
"""

import logging
import numpy as np
import time
import sys
import os

# Ensure the module can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.gpu.memory_manager import HardwareManager

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_operations():
    """Test basic array operations on both CPU and GPU"""
    logger.info("Testing basic operations")
    
    # Initialize hardware manager (auto-detect GPU)
    hw = HardwareManager()
    logger.info(f"Initialized hardware manager: {hw}")
    
    # Create test arrays
    a = np.random.randn(1000, 500).astype(np.float32)
    b = np.random.randn(500, 2000).astype(np.float32)
    
    # Test matrix multiplication
    def matrix_multiply(x, y):
        xp = hw.array_lib
        return xp.dot(x, y)
    
    # Move data to device
    logger.info("Moving data to device")
    a_device = hw.to_device(a)
    b_device = hw.to_device(b)
    
    # Execute operation with timing
    logger.info("Executing matrix multiplication")
    result_device, exec_time = hw.timed_execution(matrix_multiply, a_device, b_device)
    
    # Move result back to CPU
    result = hw.to_cpu(result_device)
    
    logger.info(f"Matrix multiplication completed in {exec_time:.4f} seconds")
    logger.info(f"Result shape: {result.shape}")
    
    # Verify result is correct by comparing with numpy
    expected = np.dot(a, b)
    error = np.abs(result - expected).max()
    logger.info(f"Maximum error: {error}")
    
    return error < 1e-5

def test_fallback_mechanism():
    """Test the fallback mechanism from GPU to CPU"""
    logger.info("Testing fallback mechanism")
    
    # Try to initialize with GPU
    hw = HardwareManager(force_cpu=False)
    initial_device = hw.current_device
    logger.info(f"Initial device: {initial_device}")
    
    # Create a function that will fail on GPU
    def failing_operation(x):
        if hw.current_device == 'gpu':
            # This will force a failure on GPU
            raise RuntimeError("Simulated GPU failure")
        return x * 2
    
    # Create test data
    test_data = np.random.rand(100)
    test_data_device = hw.to_device(test_data)
    
    # Execute with device context which should handle the fallback
    logger.info("Executing operation that will fail on GPU")
    result = hw.with_device_context(failing_operation, test_data_device)
    
    # Check if fallback occurred
    logger.info(f"Current device after fallback attempt: {hw.current_device}")
    fallback_occurred = initial_device == 'gpu' and hw.current_device == 'cpu'
    
    if fallback_occurred:
        logger.info("Fallback mechanism worked correctly")
    else:
        logger.info("No fallback occurred (either already on CPU or the error was handled differently)")
    
    # Verify result is correct
    expected = test_data * 2
    result_cpu = hw.to_cpu(result)
    error = np.abs(result_cpu - expected).max()
    logger.info(f"Maximum error: {error}")
    
    return error < 1e-5

def test_memory_management():
    """Test memory management capabilities"""
    logger.info("Testing memory management")
    
    # Initialize with lower memory threshold to trigger cleanup
    hw = HardwareManager(memory_threshold=0.5)
    
    if hw.current_device == 'cpu':
        logger.info("Skipping memory management test on CPU")
        return True
    
    # Initial memory state
    initial_mem = hw.monitor_memory_usage()
    logger.info(f"Initial memory state: {initial_mem}")
    
    # Allocate a large array to consume memory
    logger.info("Allocating large array")
    large_array = hw.to_device(np.random.rand(5000, 5000).astype(np.float32))
    
    # Check memory usage after allocation
    after_alloc_mem = hw.monitor_memory_usage()
    logger.info(f"Memory after allocation: {after_alloc_mem}")
    
    # Delete the array and force cleanup
    logger.info("Deleting array and cleaning up")
    del large_array
    hw._cleanup_gpu_memory()
    
    # Check memory after cleanup
    after_cleanup_mem = hw.monitor_memory_usage()
    logger.info(f"Memory after cleanup: {after_cleanup_mem}")
    
    # Verify memory was released
    memory_released = after_cleanup_mem['memory_used'] < after_alloc_mem['memory_used']
    
    if memory_released:
        logger.info("Memory management is working correctly")
    else:
        logger.warning("Memory might not have been properly released")
    
    return memory_released

def test_with_both_modes():
    """Test operations in both CPU and GPU modes"""
    logger.info("Testing with both CPU and GPU modes")
    
    # Test with GPU (if available)
    hw_gpu = HardwareManager(force_cpu=False)
    gpu_device = hw_gpu.current_device
    logger.info(f"Auto-detect device: {gpu_device}")
    
    # Test with forced CPU
    hw_cpu = HardwareManager(force_cpu=True)
    cpu_device = hw_cpu.current_device
    logger.info(f"Forced CPU device: {cpu_device}")
    
    assert cpu_device == 'cpu', "Forced CPU mode failed to set CPU device"
    
    # Simple operation to test
    def simple_operation(x):
        lib = hw_cpu.array_lib if hw_cpu.current_device == 'cpu' else hw_gpu.array_lib
        return lib.sum(x ** 2)
    
    # Test data
    test_data = np.random.rand(1000)
    
    # Execute on both devices
    gpu_data = hw_gpu.to_device(test_data)
    cpu_data = hw_cpu.to_device(test_data)
    
    gpu_result, gpu_time = hw_gpu.timed_execution(simple_operation, gpu_data)
    cpu_result, cpu_time = hw_cpu.timed_execution(simple_operation, cpu_data)
    
    # Convert to CPU for comparison
    gpu_result_cpu = hw_gpu.to_cpu(gpu_result)
    cpu_result_cpu = hw_cpu.to_cpu(cpu_result)
    
    logger.info(f"GPU execution time: {gpu_time:.6f} seconds")
    logger.info(f"CPU execution time: {cpu_time:.6f} seconds")
    logger.info(f"GPU result: {gpu_result_cpu}")
    logger.info(f"CPU result: {cpu_result_cpu}")
    
    # Verify results match
    results_match = abs(gpu_result_cpu - cpu_result_cpu) < 1e-5
    
    if results_match:
        logger.info("Results match between GPU and CPU execution")
    else:
        logger.warning("Results differ between GPU and CPU execution")
    
    return results_match

def run_all_tests():
    """Run all tests and report results"""
    tests = [
        ("Basic Operations", test_basic_operations),
        ("Fallback Mechanism", test_fallback_mechanism),
        ("Memory Management", test_memory_management),
        ("Both Modes", test_with_both_modes)
    ]
    
    results = []
    
    logger.info("=" * 50)
    logger.info("HARDWARE MANAGER TEST SUITE")
    logger.info("=" * 50)
    
    for name, test_func in tests:
        logger.info("\n" + "-" * 50)
        logger.info(f"Running test: {name}")
        logger.info("-" * 50)
        
        try:
            start_time = time.time()
            result = test_func()
            end_time = time.time()
            
            status = "PASSED" if result else "FAILED"
            results.append((name, status, end_time - start_time))
            
            logger.info(f"Test {name}: {status} in {end_time - start_time:.4f} seconds")
        
        except Exception as e:
            logger.error(f"Test {name} raised an exception: {str(e)}", exc_info=True)
            results.append((name, "ERROR", 0))
    
    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 50)
    
    for name, status, duration in results:
        logger.info(f"{name}: {status} ({duration:.4f}s)")
    
    # Overall status
    passed = all(status == "PASSED" for _, status, _ in results)
    logger.info("\nOverall status: " + ("PASSED" if passed else "FAILED"))
    
    return passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)