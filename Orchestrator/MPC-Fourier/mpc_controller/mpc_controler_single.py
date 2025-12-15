import numpy as np
import cvxpy as cp
import matplotlib

matplotlib.use('TkAgg')  # Or comment this if you use Jupyter
import matplotlib.pyplot as plt

# Simulation parameters
T = 20  # Total simulation time steps
H = 5  # MPC prediction horizon
lambda_t = np.random.poisson(10, size=T)  # Simulated request arrival rate

# Function constants
L_cold = 2  # Cold start delay (time steps)
L_warm = 1  # Warm execution time
mu = 5  # Requests per warm container per step
max_containers = 10  # Max container budget
alpha = 1.0  # Cold start penalty
beta = 1.0  # Queue delay penalty

# Initial state
q_init = 0
w_init = 2
cold_buffer = [0] * L_cold  # Buffer for cold start delay

# Tracking history
queue_history = [q_init]
warm_history = [w_init]
coldstart_history = []

for t in range(T - H):
    print(f"\nTime step {t}")
    print(f"  Incoming requests: {lambda_t[t]}")
    print(f"  Queue before: {queue_history[-1]}")
    print(f"  Warm containers before: {warm_history[-1]}")

    # Decision variables
    x = cp.Variable(H, integer=True)  # Cold starts
    q = cp.Variable(H + 1)  # Queue length
    w = cp.Variable(H + 1)  # Warm containers
    s = cp.Variable(H)  # Served requests

    # Constraints
    constraints = [
        q[0] == queue_history[-1],
        w[0] == warm_history[-1],
        x >= 0, q >= 0, w >= 0, s >= 0, w <= max_containers,
        s <= q[0:H],
        s <= w[0:H] * mu
    ]

    lam = lambda_t[t:t + H]
    cost = 0

    for k in range(H):
        q_next = q[k] + lam[k] - s[k]

        # Cold start arrivals from earlier steps
        if k >= L_cold:
            warm_gain = x[k - L_cold]
        else:
            warm_gain = 0
        w_next = w[k] + warm_gain

        constraints += [q[k + 1] == q_next, w[k + 1] == w_next]

        wait_warm = cp.pos(q[k] - w[k] * mu) * L_warm
        wait_cold = x[k] * (L_cold + L_warm)
        cost += beta * wait_warm + alpha * wait_cold

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.ECOS_BB)

    # Apply first control
    x0 = int(np.round(x.value[0]))
    served_now = min(queue_history[-1], warm_history[-1] * mu)
    q_next = queue_history[-1] + lambda_t[t] - served_now
    w_next = warm_history[-1] + cold_buffer.pop(0)
    cold_buffer.append(x0)

    # Print results for this time step
    print(f"  Cold starts invoked: {x0}")
    print(f"  Requests served: {served_now}")
    print(f"  Queue after: {q_next}")
    print(f"  Warm containers after: {w_next}")

    # Update history
    coldstart_history.append(x0)
    warm_history.append(w_next)
    queue_history.append(q_next)

# Plotting
plt.figure()
plt.plot(queue_history, label="Queue Length")
plt.plot(warm_history, label="Warm Containers")
plt.plot(coldstart_history, label="Cold Starts")
plt.plot(lambda_t[:T - H + 1], label="Request Arrival Rate")
plt.xlabel("Time")
plt.legend()
plt.title("MPC with Cold Start Delay")
plt.grid(True)
plt.tight_layout()
plt.show()
