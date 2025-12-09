import pytest
import math
from VMEstimator import VMEstimator

class TestVMEstimator:
    
    @pytest.fixture
    def vm_estimator(self):
        """Fixture to create a VMEstimator instance for tests"""
        return VMEstimator()

    def test_zero_load(self, vm_estimator):
        """Test with zero load (no requests, no waiting functions, no busy VMs)"""
        # Arrange
        requests_per_second = 0
        waiting_functions = []
        busy_vms = []
        idle_vms = 5
        
        # Act
        result = vm_estimator.estimate(requests_per_second, waiting_functions, busy_vms, idle_vms)
        
        # Assert
        expected = 0  # No VMs needed when there's no load
        assert result == expected
    
    def test_only_incoming_requests(self, vm_estimator):
        """Test with only incoming requests, no waiting functions or busy VMs"""
        # Arrange
        requests_per_second = 7
        waiting_functions = []
        busy_vms = []
        idle_vms = 3
        
        # Act
        result = vm_estimator.estimate(requests_per_second, waiting_functions, busy_vms, idle_vms)
        
        # Assert
        expected = math.ceil(requests_per_second)
        assert result == expected
    
    def test_only_waiting_functions(self, vm_estimator):
        """Test with only waiting functions, no incoming requests or busy VMs"""
        # Arrange
        requests_per_second = 0
        waiting_functions = [2.0, 1.5, 3.0, 1.0]
        busy_vms = []
        idle_vms = 2
        
        # Act
        result = vm_estimator.estimate(requests_per_second, waiting_functions, busy_vms, idle_vms)
        
        # Assert
        expected = len(waiting_functions)
        assert result == expected
    
    def test_only_busy_vms(self, vm_estimator):
        """Test with only busy VMs, no incoming requests or waiting functions"""
        # Arrange
        requests_per_second = 0
        waiting_functions = []
        busy_vms = [2.5, 3.0, 1.5]
        idle_vms = 0
        
        # Act
        result = vm_estimator.estimate(requests_per_second, waiting_functions, busy_vms, idle_vms)
        
        # Assert
        # When there's no load but busy VMs, the result should be the count of busy VMs
        assert result == len(busy_vms)
    
    def test_soon_free_vms_threshold(self, vm_estimator):
        """Test the soon_free_threshold behavior with busy VMs"""
        # Arrange
        vm_estimator.soon_free_threshold = 1.0
        requests_per_second = 3
        waiting_functions = [2.0, 1.5]
        # Two VMs will be free soon (below the threshold)
        busy_vms = [0.5, 0.8, 1.5, 2.0]
        idle_vms = 1
        
        # Act
        result = vm_estimator.estimate(requests_per_second, waiting_functions, busy_vms, idle_vms)
        
        # Assert
        # Required: 3 (requests) + 2 (waiting) = 5
        # Available: 1 (idle) + 2 (soon free) = 3
        # Non-soon free VMs: 2
        # Expected: 5 + 2 = 7
        expected = math.ceil(requests_per_second) + len(waiting_functions) + (len(busy_vms) - 2)
        assert result == expected
    
    def test_enough_available_resources(self, vm_estimator):
        """Test when there are enough available resources (idle + soon-to-be-free)"""
        # Arrange
        requests_per_second = 2
        waiting_functions = [1.0]
        busy_vms = [0.5, 2.0, 3.0]  # One VM will be free soon
        idle_vms = 5
        
        # Act
        result = vm_estimator.estimate(requests_per_second, waiting_functions, busy_vms, idle_vms)
        
        # Assert
        # Required: 2 (requests) + 1 (waiting) = 3
        # Available: 5 (idle) + 1 (soon free) = 6, which is > 3
        expected = math.ceil(requests_per_second) + len(waiting_functions) + len(busy_vms)
        assert result == expected
    
    def test_not_enough_available_resources(self, vm_estimator):
        """Test when there are not enough available resources"""
        # Arrange
        requests_per_second = 10
        waiting_functions = [1.5, 2.0, 3.0, 4.0, 1.0]
        busy_vms = [0.5, 0.8, 2.0, 3.0]  # Two VMs will be free soon
        idle_vms = 2
        
        # Act
        result = vm_estimator.estimate(requests_per_second, waiting_functions, busy_vms, idle_vms)
        
        # Assert
        # Required: 10 (requests) + 5 (waiting) = 15
        # Available: 2 (idle) + 2 (soon free) = 4, which is < 15
        # Non-soon free VMs: 2
        # Expected: 15 + 2 = 17
        total_required = math.ceil(requests_per_second) + len(waiting_functions)
        soon_free_vms = sum(1 for time in busy_vms if time <= vm_estimator.soon_free_threshold)
        expected = total_required + (len(busy_vms) - soon_free_vms)
        assert result == expected
    
    def test_decimal_requests_per_second(self, vm_estimator):
        """Test with decimal requests_per_second value"""
        # Arrange
        requests_per_second = 3.7
        waiting_functions = [1.0, 2.0]
        busy_vms = [1.5, 2.5]
        idle_vms = 1
        
        # Act
        result = vm_estimator.estimate(requests_per_second, waiting_functions, busy_vms, idle_vms)
        
        # Assert
        # ceil(3.7) = 4 for requests
        expected = math.ceil(requests_per_second) + len(waiting_functions) + len(busy_vms)
        assert result == expected
    
    def test_with_example_values(self, vm_estimator):
        """Test with the example values provided in the VMEstimator.py file"""
        # Arrange - using values from the example in VMEstimator.py
        requests_per_second = 5
        waiting_functions = [2.0, 1.5, 3.0]
        busy_vms = [1.0, 2.5]
        idle_vms = 2
        
        # Act
        result = vm_estimator.estimate(requests_per_second, waiting_functions, busy_vms, idle_vms)
        
        # Assert
        # Required: 5 (requests) + 3 (waiting) = 8
        # Available: 2 (idle) + 1 (soon free) = 3, which is < 8
        # Non-soon free VMs: 1
        # Expected: 8 + 1 = 9
        assert result == 9

    def test_custom_soon_free_threshold(self, vm_estimator):
        """Test with a custom soon_free_threshold value"""
        # Arrange
        vm_estimator.soon_free_threshold = 2.0  # Change threshold
        requests_per_second = 4
        waiting_functions = [1.0, 3.0]
        busy_vms = [0.5, 1.8, 2.5, 3.0]  # Now 2 VMs will be free soon with the new threshold
        idle_vms = 1
        
        # Act
        result = vm_estimator.estimate(requests_per_second, waiting_functions, busy_vms, idle_vms)
        
        # Assert
        # Required: 4 (requests) + 2 (waiting) = 6
        # Available: 1 (idle) + 2 (soon free) = 3, which is < 6
        # Non-soon free VMs: 2
        # Expected: 6 + 2 = 8
        soon_free_vms = sum(1 for time in busy_vms if time <= vm_estimator.soon_free_threshold)
        assert soon_free_vms == 2
        expected = math.ceil(requests_per_second) + len(waiting_functions) + (len(busy_vms) - soon_free_vms)
        assert result == expected 