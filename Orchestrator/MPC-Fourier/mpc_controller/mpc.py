
import cvxpy as cp
import math
import numpy as np
H = 10
L_cold = 5
L_warm = 0.28
mu = 1/L_warm
max_containers = 10
alpha = 1.0
beta = 1.0
gamma = 0.05
long_queue_penalty = 1

# Initial state
cold_buffer = [0] * L_cold

def mpc(H, q, w, max_containers, mu, lam):
    smoothness_weight = 1.0 # 0.5, 2.0
    x = cp.Variable(H, integer=True)  # number of cold start
    r = cp.Variable(H, integer=True)  # number of reclaim
    q_var = cp.Variable(H + 1, nonneg=True)  # queue length
    w_var = cp.Variable(H + 1, nonneg=True)  # number of warm pod
    s = cp.Variable(H, nonneg=True)  # number of servered requests

    constraints = [
        q_var[0] == q,
        w_var[0] == w,
        x >= 0, r >= 0, q_var >= 0, w_var >= 0, s >= 0,
        w_var <= max_containers,
        s <= q_var[0:H],
        s <= w_var[0:H] * mu
    ]

    cost = 0
    cumulative_arrivals = 0
    max_delay = L_warm + 1
    max_queue = max_delay * mu
    for k in range(H):

        q_next = q_var[k] + lam[k] - s[k]
        warm_gain = x[k - L_cold] if k >= L_cold else 0
        w_next = w_var[k] + warm_gain - r[k]

        constraints += [q_var[k + 1] == q_next, w_var[k + 1] == w_next]
        #constraints += [r[k] <= w_var[k]]
        delay_penalty = cp.pos(q_var[k] - max_queue)
        cost += 10.0 * delay_penalty
        warm_capacity = w_var[k] * mu
        cumulative_arrivals += lam[k]

        latency_cost = (L_cold + L_warm) * cp.pos(cumulative_arrivals - warm_capacity * (k + 1))
        wait_warm = q_var[k] * L_warm
        #wait_cold = x[k] * (L_cold + L_warm)
        cold_delay_penalty = cp.pos(lam[k] - w_var[k] * mu) * (L_cold + L_warm)

        #cost += alpha * cold_delay_penalty

        overprovision = (gamma / mu) * cp.maximum(w_var[k] * mu - q_var[k], 0)
        reclaim_penalty = 0.2
        cold_start_cost = 0.1 * x[k]

        # Total cost
        cost += alpha * cold_delay_penalty + beta * wait_warm + overprovision + reclaim_penalty * r[k] + cold_start_cost
        #cost += reclaim_penalty * r[k]
        #cost += reclaim_penalty * cp.minimum(r[k], w_var[k] - lam[k] / mu)
        #constraints += [r[k] <= w_var[k] - lam[k] / mu]
        max_reclaim = max(0, w - lam[k] / mu)  # upper bound as constant
        constraints += [r[k] <= max_reclaim]
        if k >=1:
            cost += long_queue_penalty * q_var[k]
        if k > 0:
            cost += smoothness_weight * cp.square(w_var[k] - w_var[k - 1])
            cost += 0.2 * cp.square(x[k] - x[k - 1])
            #cost += 0.1 * cp.square(w_var[k] - w_var[k - 1])

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.ECOS_BB)
    s0 = int(np.round(s.value[0]))  # Number of real requests MPC says should be served
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print("MPC solver failed to converge.")
        x0, r0 = 0, 0
    else:
        x0 = int(np.round(x.value[0]))  # number runtimes will be cold started
        r0 = int(np.round(r.value[0]))  # number runtimes will be reclaimed
        s0 = int(np.round(s.value[0]))  # number requests will send to openwhisk

    # elapsed = time.time() - start
    # elapsed = min(elapsed, 5.0)  # cap to 5 seconds maximum
    served = min(q, w * mu)
    return served, x0, r0, s0

q= 100
w = 1
served, x0, r0, s0 =mpc(H=20, q=100, w=1, max_containers=10, mu=1/0.28, lam=[0.3, 0.5, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.0,0.3, 0.5, 0, 0, 0, 0])

print(f'with q={q}, w={w}')
print(f'served = {served}')
print(f's0= {s0}')
print(f'x0= {x0}')
print(f'r0 = {r0}')
