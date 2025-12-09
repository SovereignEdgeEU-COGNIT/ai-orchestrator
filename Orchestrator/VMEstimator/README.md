# VMEstimator

The VMEstimator is responsible for calculating the optimal number of VMs needed to handle function execution workloads.

## Features

- **Reactive VM Estimation**: Calculate required VMs based on current load metrics
- **Proactive VM Prediction**: Predict future VM needs using trained models (to be implemented)
- **Model Training**: Train prediction models using historical data (to be implemented)

## How to Use

```python
from VMEstimator import VMEstimator

# Create an instance
vm_estimator = VMEstimator()

# Calculate optimal number of VMs
optimal_vms = vm_estimator.estimate(
    requests_per_second=5,
    waiting_functions=[2.0, 1.5, 3.0],
    busy_vms=[1.0, 2.5],
    idle_vms=2
)
```

## Running Tests

Tests are written using pytest. To run the tests, install pytest and run:

```bash
# Install pytest if not already installed
pip install pytest

# Run the tests
cd orchestration-algorithms/VMEstimator
pytest test_VMEstimator.py -v
```

## Test Coverage

The tests cover the following scenarios:

- Zero load estimation
- Only incoming requests
- Only waiting functions
- Only busy VMs
- Soon-to-be-free VMs threshold behavior
- Sufficient vs. insufficient available resources
- Decimal requests per second handling
- Custom soon-free threshold values
- Example values from the main script 