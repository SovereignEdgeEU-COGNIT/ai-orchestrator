# Orchestrator Algorithms 2.0

This branch contains the second version of algorithms used by the AI-enabled orchestrator, with enhanced functionality focused on managing serverless runtimes more efficiently.

## Algorithms Included
1. **VMEstimator**: This algorithm estimates the virtual machine (VM) requirements based on the current workload and demand. It helps in scaling VMs dynamically to ensure optimal performance and cost-efficiency in serverless runtime technologies (SRTs).

1. **VMPlacer**: This algorithm determines the optimal placement of VMs across available physical hosts. It considers factors such as resource availability, usage, and VM requirements to minimize operational costs and maximize performance in environments utilizing SRTs.

1. **LoadOptimizer**: This algorithm optimizes both energy usage and interference, aiming to manage serverless functions effectively within a cloud-edge environment.

## Version Information
- **Version**: 2.0
- **Algorithms**: Energy-interference optimization for managing serverless runtimes

This version builds upon the energy-aware capabilities by adding interference management, providing better performance for serverless deployments in shared resources.
