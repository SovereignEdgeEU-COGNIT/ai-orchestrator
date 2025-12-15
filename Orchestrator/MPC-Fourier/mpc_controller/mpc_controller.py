import numpy as np
import cvxpy as cp
import matplotlib
matplotlib.use('TkAgg')  # or 'QtAgg' or 'Agg' depending on your setup
import matplotlib.pyplot as plt
# Simulation parameters
real_horizon_sec = 5         # Prediction horizon (look ahead 10 steps)
dt = 0.1        # Time step in seconds (100ms per step)
H= int(real_horizon_sec/dt)

# System parameters
cold_start_time = 10.2 #in second
cold_start_delay = int(np.ceil(cold_start_time/dt)) # number of time steps

warm_exec_time = 0.3  # in second
warm_container_timeout = 10 # second after which unsued warm container is reclaimed
warm_container_lifetime = int (warm_container_timeout/dt) # in time steps

max_total_containers = 50    # warm container
queue_capacity = 100    # max request queue size
initial_warm_containers = 3
initial_queue = 0

# Cost weights
alpha = 1.0   # Cold start penalty
beta = 0.5    # Queue delay penalty
gamma = 0.1   # Resource usage cost

# Simulate Predicted future arrivals (lambda)
predicted_arrivals = np.zeros(H)
for t in range(H):
    current_time = t * dt
    if 0 <= current_time < 3:
        predicted_arrivals[t] = 50 * dt
    elif 3 <= current_time < 6:
        predicted_arrivals[t] = 25 * dt
    elif 6 <= current_time < 9:
        predicted_arrivals[t] = 50 * dt
    elif 9 <= current_time < 12:
        predicted_arrivals[t] = 75 * dt
    elif 12 <= current_time < 15:
        predicted_arrivals[t] = 100 * dt
    else:
        predicted_arrivals[t] = 0

print("Predicted arrivals (first 20):", predicted_arrivals[:20])
print("Sum of arrivals over full horizon:", np.sum(predicted_arrivals))



# Optimization variables
dispatch_warm = cp.Variable(H) # how many request will be dispatched to warm containers at time t
cold_start_trigger = cp.Variable(H) # how many cold starts to trigger at time t
queue_length = cp.Variable(H+1) # how many reqeust are waiting in queue at time t
warm_container_count = cp.Variable(H+1)


# Model cold start readiness
cold_start_ready = [0]*H
for t in range(H):
    if t>= cold_start_delay:
        cold_start_ready[t] = cold_start_trigger[t-cold_start_delay]
    else:
        cold_start_ready[t] =0
cold_start_ready = cp.hstack(cold_start_ready)

# Constraints
constraints = [
    queue_length[0] == initial_queue,
    warm_container_count[0] == initial_warm_containers
]

for t in range(H):
    served = dispatch_warm[t] + cold_start_ready[t]
    #queue dynamic

    constraints += [
        queue_length[t+1] == queue_length[t] + predicted_arrivals[t] - served,
        queue_length[t + 1] >= 0,
        queue_length[t + 1] <= queue_capacity

    ]

    #Dispatch contraints
    constraints += [
        dispatch_warm[t] >=0,
        dispatch_warm[t] <= warm_container_count[t],
        cold_start_trigger[t] >= 0
    ]

    # Warm container evolution
    warm_decay = dispatch_warm[t - warm_container_lifetime] if t >= warm_container_lifetime else 0
    warm_added = cold_start_ready[t]
    constraints += [
        warm_container_count[t+1] == warm_container_count[t] - warm_decay + warm_added,
        warm_container_count[t+1] >=0
    ]

    #Total container cap
    constraints += [
        warm_container_count[t] + cold_start_trigger[t] <= max_total_containers
    ]

# Objective function
objective = cp.Minimize(
    alpha * cp.sum(cold_start_trigger) +
    beta * cp.sum(queue_length[1:]) +
    gamma * cp.sum(dispatch_warm + cold_start_trigger)
)


# Solve
problem = cp.Problem(objective, constraints)
problem.solve(verbose=True) #give feedback

print("Problem status:", problem.status)

# Print results
print("Dispatch to warm:", dispatch_warm.value)
print("Cold starts     :", cold_start_trigger.value)
print("Queue length    :", queue_length.value)

# Plot results
time_axis = np.arange(H) * dt
plt.figure(figsize=(12, 6))
plt.plot(time_axis, dispatch_warm.value, label="Dispatched to Warm", marker='o')
plt.plot(time_axis, cold_start_trigger.value, label="Cold Starts Triggered", marker='x')
plt.plot(time_axis, cold_start_ready.value, label="Cold Starts Ready (Used)", marker='^')
plt.plot(time_axis, warm_container_count.value[:-1], label="Warm Containers", marker='s')
plt.plot(time_axis, queue_length.value[1:], label="Queue Length", marker='d')
plt.xlabel("Time (seconds)")
plt.ylabel("Count")
plt.title("MPC Scheduling with Dynamic Warm Pool and Container Cap")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()