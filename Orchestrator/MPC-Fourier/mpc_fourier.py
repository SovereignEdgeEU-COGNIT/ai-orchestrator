import redis
import time
import json
import requests
import base64
import cvxpy as cp
import numpy as np
from threading import Thread
from numpy.fft import fft, fftfreq
from concurrent.futures import ThreadPoolExecutor
from threading import Semaphore, BoundedSemaphore
#from termcolor import colored
#import urllib3
#urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from prometheus.monitoring_function import PrometheusClient
from Reclaim import final_clean_pod_reclaim
from utility import forecast_log
from AutoScaler import prewarm
try:
    from termcolor import colored
except ImportError:
    def colored(text, color=None, on_color=None, attrs=None):
        # Simple fallback if termcolor is not installed
        return text

# Redis setup
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Constants
whisk_url = "https://XXX.XXX.XX.XXX:XXXXX/api/v1/namespaces/guest/actions/object-detection?blocking=true&result=true"
auth_key = "AUTHENTICATION KEY"
header = {
    "Authorization": f"Basic {base64.b64encode(auth_key.encode()).decode()}",
    "Content-Type": "application/json"
}

# Load image and encode
img_path = "frame_sample/3.jpg"
with open(img_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

# Thread-safe HTTP session
session = requests.Session()

# Configuration
warm_containers = 1
max_dispatch_per_loop = 30
executor = ThreadPoolExecutor(max_workers=1)
 # Only allow N concurrent requests
service_time = 0.28
gap = service_time + 0.01  # Add buffer

recently_deleted_pods = {}  # pod_name
RECLAIM_GRACE_PERIOD = 60  # seconds

current_warm = 1  # or 1 to start
sema = BoundedSemaphore(current_warm)
def get_queue_arrival_rate():
    now = time.time()
    recent_entries = r.lrange("request_timestamps", -30, -1)
    recent_times = [float(ts) for ts in recent_entries if now - float(ts) <= 30]
    return len(recent_times) / 30.0

def update_executor_if_needed(new_worker_count):
    global executor, warm_containers
    if new_worker_count != warm_containers:
        print(f"[INFO] Updating ThreadPoolExecutor to {new_worker_count} workers")
        executor.shutdown(wait=False)  # do not wait for existing tasks
        warm_containers = new_worker_count
        executor = ThreadPoolExecutor(max_workers=warm_containers)
def update_semaphore_if_needed(new_warm_count):
    global current_warm, sema
    if new_warm_count != current_warm:
        current_warm = new_warm_count
        sema = BoundedSemaphore(current_warm)
    print(f"[INFO] Updated semaphore to {current_warm} warm containers")
def is_recently_deleted(pod_name):
    ts = recently_deleted_pods.get(pod_name)
    return ts is not None and (time.time() - ts) < RECLAIM_GRACE_PERIOD

def update_recent_deletions(pod_name):
    recently_deleted_pods[pod_name] = time.time()

def cleanup_deletion_history():
    now = time.time()
    expired = [k for k, t in recently_deleted_pods.items() if now - t > RECLAIM_GRACE_PERIOD]
    for k in expired:
        del recently_deleted_pods[k]
def forecast(trace, n_predict=5, harmonics=10, clip_std_factor=3.0):
    trace = np.array(trace)
    n = trace.size
    t = np.arange(n)

    # Quadratic trend fitting: f(t) = a*t^2 + b*t + c
    coeffs = np.polyfit(t, trace, 2)  # [a, b, c]
    trend = np.polyval(coeffs, t)
    detrended = trace - trend

    # Fourier Transform
    fft_values = fft(detrended)
    freqs = fftfreq(n)
    indexes = np.argsort(np.abs(freqs))  # low frequency to high

    # Reconstruct signal with top `harmonics`
    t_future = np.arange(n, n + n_predict)
    restored = np.zeros(n_predict)
    for i in indexes[:1 + harmonics * 2]:
        amplitude = np.abs(fft_values[i]) / n
        phase = np.angle(fft_values[i])
        restored += amplitude * np.cos(2 * np.pi * freqs[i] * t_future + phase)

    # Add back the quadratic trend for future steps
    trend_future = np.polyval(coeffs, t_future)
    forecast = restored + trend_future

    # Optional: clip negative or extreme values
    mean = np.mean(trace)
    std = np.std(trace)
    upper_bound = mean + clip_std_factor * std
    forecast = np.clip(forecast, 0, upper_bound)

    return forecast
def maybe_reclaim(r0,recently_deleted_pods):
    if r0 > 0:
        print(f"[INFO][{time.strftime('%H:%M:%S')}] calling early reclaim to reclaim {r0}")
        deleted_pods = final_clean_pod_reclaim.reclaim_pods(r0, recently_deleted_pods.keys())
        for pod in deleted_pods:
            recently_deleted_pods[pod] = time.time()

def maybe_cold_start(x0):
    if x0 > 0:
        print(f"[INFO][{time.strftime('%H:%M:%S')}] Calling cold start {x0} new pods ")
        prewarm.prewarm_with_shell(x0, action_name)


def mpc_revised_v3(H, q, w, lam):
    # Parameters & Constants 
    L_warm = 0.28
    L_cold = 10.0
    MPC_step_duration = 5.0

    cold_start_delay = int(np.round(L_cold / MPC_step_duration))
    mu = MPC_step_duration / L_warm
    max_containers = 40

    #Cost Weights
    queue_cost_beta_Lwarm_factor = 200.0
    queue_cost_linear_direct = 100.0
    queue_cost_quadratic = 10.0
    queue_cost_max_violation_coeff = 500.0
    queue_max_threshold_for_violation = 0.5

    capacity_shortfall_penalty_coeff = 2.0 * (L_cold + L_warm)
    cold_start_direct_cost = 0.6

    warm_pod_running_cost = 0.02
    idle_running_capacity_cost = 0.01 #0.1
    reclamation_reward_val = 0#-0.2
    cumulative_latency_penalty_coeff = 0.5 * (L_cold + L_warm)

    smoothness_x_coeff = 1.0
    smoothness_r_coeff = 1.0
    smoothness_w_coeff = 0.5

    #Penalty for violating the reclaim target (soft constraint)
    penalty_reclaim_target_violation = 10000.0  # Should be a large value

    EPSILON = 1e-7

    #Optimization Variables
    x = cp.Variable(H, nonneg=True, name="cold_starts")
    r = cp.Variable(H, nonneg=True, name="reclaims")
    q_var = cp.Variable(H + 1, nonneg=True, name="queue_length")
    w_var = cp.Variable(H + 1, nonneg=True, name="warm_pods")
    s = cp.Variable(H, nonneg=True, name="served_requests")

    #Slack variable for the soft reclamation constraint
    reclaim_target_violation = cp.Variable(H, nonneg=True, name="reclaim_target_slack")

    #Constraints
    constraints = [
        q_var[0] == q,
        w_var[0] == w,
        w_var <= max_containers,
    ]

    if w == 0 and q > 0:
        constraints += [x[0] >= 1]

    cost = 0

    for k in range(H):
        # Service Constraints
        constraints += [s[k] <= q_var[k]]
        constraints += [s[k] <= w_var[k] * mu]

        # System Dynamics
        constraints += [q_var[k + 1] == q_var[k] + lam[k] - s[k]]
        warm_gain_from_cold_starts = x[k - cold_start_delay] if k >= cold_start_delay else 0
        constraints += [w_var[k + 1] == w_var[k] + warm_gain_from_cold_starts - r[k]]

        # Reclamation Constraints
        constraints += [r[k] <= w_var[k]]  # Cannot reclaim more than available

        #MODIFIED RECLAMATION CONSTRAINT (SOFT CONSTRAINT)
        # This constraint tries to ensure that after reclamation, remaining pods (w_var[k] - r[k])
        # can handle the current demand (lam[k]/mu). If not possible, the slack variable
        # 'reclaim_target_violation[k]' will absorb the deficit at a high cost.
        # This is now DCP compliant: affine_expr >= affine_expr - affine_var
        reclaim_target_rhs = lam[k] / mu - EPSILON
        constraints += [w_var[k] - r[k] >= reclaim_target_rhs - reclaim_target_violation[k]]

        # Cost for using the slack variable (violating the reclaim target)
        cost += penalty_reclaim_target_violation * reclaim_target_violation[k]

        #Cost Components (rest are the same as mpc_revised)
        cost += queue_cost_beta_Lwarm_factor * q_var[k + 1] * L_warm
        cost += queue_cost_linear_direct * q_var[k + 1]
        cost += queue_cost_quadratic * cp.square(q_var[k + 1])
        cost += queue_cost_max_violation_coeff * cp.pos(q_var[k + 1] - queue_max_threshold_for_violation)

        cost += capacity_shortfall_penalty_coeff * cp.pos(lam[k] - w_var[k] * mu)
        cost += cold_start_direct_cost * x[k]

        cost += warm_pod_running_cost * w_var[k + 1]
        cost += idle_running_capacity_cost * cp.pos(w_var[k] * mu - lam[k])
        cost += reclamation_reward_val * r[k]

        current_warm_capacity_at_k_mu = w_var[k] * mu
        cumulative_lambda_up_to_k = cp.sum(lam[0:k+1])
        cost += cumulative_latency_penalty_coeff * cp.pos(
            cumulative_lambda_up_to_k - current_warm_capacity_at_k_mu * (k + 1))

        if k >= 1:
            cost += smoothness_x_coeff * cp.square(x[k] - x[k - 1])
            cost += smoothness_r_coeff * cp.square(r[k] - r[k - 1])
            cost += smoothness_w_coeff * cp.square(w_var[k + 1] - w_var[k])

    #Solver Call
    problem = cp.Problem(cp.Minimize(cost), constraints)
    solver_succeeded = False

    try:
        problem.solve(solver=cp.GUROBI, verbose=False)  # Keep verbose=True for debugging
        if problem.status in ["optimal", "optimal_inaccurate"]:
            solver_succeeded = True
        else:
            print(colored(f"MPC solver did NOT find an optimal solution. Status: {problem.status}", "red"))
            if problem.value is not None:
                print(colored(f"   Problem value: {problem.value}", "red"))
            if reclaim_target_violation.value is not None:
                print(colored(f"   Reclaim Slack values: {np.round(reclaim_target_violation.value, 3)}", "yellow"))


    except cp.error.SolverError as e:
        print(colored(f"MPC solver (CVXPY) error: {e}", "red"))
    except Exception as e:
        print(colored(f"MPC solver encountered an unexpected exception: {e}", "red"))

    # --- Result Processing ---
    if solver_succeeded:
        s_values = np.round(s.value).astype(int) if s.value is not None else np.zeros(H, dtype=int)
        x_values = np.round(x.value).astype(int) if x.value is not None else np.zeros(H, dtype=int)
        r_values = np.round(r.value).astype(int) if r.value is not None else np.zeros(H, dtype=int)

        q_debug = np.round(q_var.value, 2) if q_var.value is not None else ["N/A"] * (H + 1)
        w_debug = np.round(w_var.value, 2) if w_var.value is not None else ["N/A"] * (H + 1)
        reclaim_slack_debug = np.round(reclaim_target_violation.value,
                                       3) if reclaim_target_violation.value is not None else ["N/A"] * H

        s0_action = s_values[0]
        x0_action = x_values[0]
        r0_action = r_values[0]

        print(colored(f"[MPC] Optimal solution found. Status: {problem.status}", "green"))
        print(colored(f"  Objective Value: {problem.value}", "green"))
        print(colored(f"  Q_var: {q_debug}", "yellow"))
        print(colored(f"  W_var: {w_debug}", "yellow"))
        print(colored(f"  Reclaim Slack: {reclaim_slack_debug}", "yellow"))
        print(colored(f"  X values: {x_values}", "cyan"))
        print(colored(f"  R values: {r_values}", "magenta"))
        print(colored(f"  S values (served per step): {s_values}", "blue"))

        return s0_action, x0_action, r0_action, s0_action, s_values.tolist()

    else:
        print(colored("MPC optimization failed. Returning default zero actions for x0, r0.", "red"))
        x0_action = 0
        r0_action = 0
        s0_action = int(np.floor(min(q, w * mu if w > 0 else 0)))
        print(colored(f"  Fallback actions: s0={s0_action}, x0={x0_action}, r0={r0_action}", "red"))
        s_values_fallback = [s0_action] + [0] * (H - 1)
        return s0_action, x0_action, r0_action, s0_action, s_values_fallback


def mpc(H, q, w, lam):
    smoothness_weight = 1.0 # 0.5, 2.0
    L_warm = 0.28 #second
    L_cold = 10 # second
    MPC_step_duration = 5 #second
    cold_start_delay = int(np.round(L_cold / MPC_step_duration)) # This will be 2
    mu = MPC_step_duration * 1 / L_warm # Requests served per warm container per step (approx 17.857)
    max_containers = 40
    alpha = 1.0 # Weight for cold_delay_penalty
    beta = 1.0 # Weight for wait_warm
    gamma = 0.05 # Weight for overprovision
    long_queue_penalty = 1 # Weight for direct queue length penalty (if k >= 1)

    # Define optimization variables
    x = cp.Variable(H, nonneg=True)  # number of cold start
    r = cp.Variable(H, nonneg=True)  # number of reclaim
    q_var = cp.Variable(H + 1, nonneg=True)  # queue length
    w_var = cp.Variable(H + 1, nonneg=True)  # number of warm pod
    s = cp.Variable(H, nonneg=True)  # number of served requests

    constraints = [
        q_var[0] == q,
        w_var[0] == w,
        x >= 0, r >= 0, q_var >= 0, w_var >= 0, s >= 0,
        w_var <= max_containers # Max total containers constraint
    ]

    cost = 0
    max_delay = L_warm + 1
    max_queue = max_delay * mu # ~22.857

    # Define EPSILON once globally for clarity
    EPSILON = 1e-3

    for k in range(H):
        # Constraints on served requests (s[k])
        constraints += [s[k] <= q_var[k]]  # Served requests cannot exceed current queue length
        constraints += [s[k] <= w_var[k] * mu]  # Served requests cannot exceed warm pod capacity

        # Dynamics for queue and warm pods
        q_next = q_var[k] + lam[k] - s[k]
        # warm_gain from cold starts considering delay
        warm_gain = x[k - cold_start_delay] if k >= cold_start_delay else 0
        w_next = w_var[k] + warm_gain - r[k]

        constraints += [q_var[k + 1] == q_next]
        constraints += [w_var[k + 1] == w_next]

        # DCP-compliant reclaim constraints (FIXED from previous version)
        constraints += [r[k] <= w_var[k]]  # Cannot reclaim more than available warm containers
        constraints += [w_var[k] - r[k] >= lam[k] / mu - EPSILON]  # Ensure remaining warm containers can serve demand + EPSILON

        #Cost components -----------------------
        # Calculate warm_capacity for the current step k
        warm_capacity = w_var[k] * mu

        delay_penalty = cp.pos(q_var[k] - max_queue)
        cost += 10.0 * delay_penalty

        # Refactored latency_cost calculation
        cumulative_lambda_k = cp.sum(lam[0:k+1]) # Sum of lambda up to current step k
        latency_cost = (L_cold + L_warm) * cp.pos(cumulative_lambda_k - warm_capacity * (k + 1))
        cost += latency_cost # Add this cost, it was correctly identified as missing before.

        wait_warm = q_var[k] * L_warm # Cost for waiting in queue
        cold_delay_penalty = cp.pos(lam[k] - w_var[k] * mu) * (L_cold + L_warm) # Penalty for demand > capacity
        overprovision = (gamma / mu) * cp.pos(w_var[k] * mu - q_var[k]) # Penalty for overprovisioning

        cold_start_cost = 0.5 * x[k] # Adjusted cold_start_cost

        idle_capacity = cp.pos(w_var[k] * mu - lam[k])
        reclaim_reward = -0.3 * r[k]  # Reward for reclaiming

        # Sum up cost components for current step k
        cost += alpha * cold_delay_penalty \
              + beta * wait_warm \
              + overprovision \
              + cold_start_cost \
              + 0.05 * idle_capacity \
              + reclaim_reward

        # Smoothness penalties
        if k >= 1:
            cost += 1.0 * cp.square(x[k] - x[k - 1])  # Penalize large changes in cold starts
            cost += 1.0 * cp.square(r[k] - r[k - 1])  # Penalize large changes in reclaims
            cost += smoothness_weight * cp.square(w_var[k] - w_var[k - 1]) # Penalize large changes in warm pods

            # Direct queue length penalty for k >= 1 (as per original logic)
            cost += long_queue_penalty * q_var[k]

    # Special handling for w=0 and q>0 initial state (bootstrap phase)
    # This block *forces* a cold start if starting from zero warm containers with pending demand.
    # The MPC will then solve for the consequences.
    if w == 0 and q > 0:
        x_min = 1  # At least one cold start is required to begin
        constraints += [x[0] >= x_min]


    # --- Solver Call ---
    prob = cp.Problem(cp.Minimize(cost), constraints)
    try:
        # Using OSQP as it's often more robust for QP problems.
        # Set verbose=True for detailed solver output if debugging, otherwise False.
        prob.solve(solver=cp.GUROBI, verbose=False)
        solver_succeeded = True
    except Exception as e:
        print(f"MPC solver encountered an exception: {e}")
        solver_succeeded = False # Indicate failure if exception occurs

    # Result Processing ----------------------------
    # Check prob.status only if solver_succeeded, otherwise assume failure.
    if not solver_succeeded or prob.status not in ["optimal", "optimal_inaccurate"]:
        if solver_succeeded: # This means it ran but returned non-optimal status
            print(colored(f"MPC solver failed to converge. Status: {prob.status}", "red"))
        # Return zeros if optimization failed or was not optimal
        x0 = r0 = s0 = 0
        s_values = np.zeros(H).tolist() # Ensure s_values is a list for consistency
        # Calculate 'served' based on current state (0 if w=0 or no capacity)
        served = min(q, w * mu) # Or just 0 if w=0 is the primary failure mode
        return served, x0, r0, s0, s_values
    else:
        # Round and convert values to appropriate types
        s_values = np.round(s.value).astype(int).tolist() if s.value is not None else [0] * H
        x_values = np.round(x.value).astype(int).tolist() if x.value is not None else [0] * H
        r_values = np.round(r.value).astype(int).tolist() if r.value is not None else [0] * H

        s0 = int(s_values[0])
        x0 = int(x_values[0])
        r0 = int(r_values[0])

        print(colored(f"[MPC] Optimal status: {prob.status}", "green"))
        print(colored(f"[MPC] x values: {x_values}", "cyan"))
        print(colored(f"[MPC] r values: {r_values}", "magenta"))
        print(colored(f"[MPC] s values: {s_values}", "red", "on_blue"))
        # print(f"q_var: {np.round(q_var.value)}") # Uncomment for debugging queue state
        # print(f"w_var: {np.round(w_var.value)}") # Uncomment for debugging warm pod state

        # The 'served' value returned is the actual number of requests served at the current step (s[0])
        served = s0

        return served, x0, r0, s0, s_values

def send_batch_gradual(target_count, duration=.01, afterMPC = True):
    interval = duration / max(1, target_count)
    sent = 0
    start = time.time()
    deadline = start + duration

    while sent < target_count and time.time() < deadline:
        req_raw = r.lpop("request_queue")
        if req_raw:
            try:
                req = json.loads(req_raw)
                executor.submit(send_request, req, afterMPC)
                sent += 1
            except Exception as e:
                print(f"Dispatch error: {e}")

        #time.sleep(interval)


def send_batch (total_request_send, total_warm_containers, service_time):


    gap = service_time + 0.01  # Add buffer
    batch_size = total_warm_containers  # 3 requests per round
    print(f"[MPC] Dispatching {total_request_send} requests...")
    num_batches = total_request_send // batch_size + int(total_request_send % batch_size > 0)
    for i in range(num_batches):
        future = []
        for _ in range(batch_size):
            req_raw = r.lpop("request_queue")
            if not req_raw:
                break
            try:
                req = json.loads(req_raw)
                future.append(executor.submit(send_request, req))

            except Exception as e:
                print(f"Dispatch error: {e}")

def dispatch_per_second(s_list, afterMPC=True):
    for i, count in enumerate(s_list):
        print(f"[MPC] Second {i}: dispatching {count} requests")
        for _ in range(count):
            req_raw = r.lpop("request_queue")
            if not req_raw:
                break
            try:
                req = json.loads(req_raw)
                executor.submit(send_request, req, afterMPC)
            except Exception as e:
                print(f"Dispatch error: {e}")
        time.sleep(1)
def send_request(req, afterMPC= True):

    """Send request and measure delay"""
    with sema:
        try:
            source = "afterMPC" if afterMPC else "beforeMPC"
            print(colored(f"We are saving data for {source}", "yellow", "on_red"))
            req_id = req["req_id"]
            enqueue_time = req["enqueue_time"]
            dequeue_time = time.time()
            queue_delay = dequeue_time - enqueue_time

            payload = {
                "object_of_interest": "red",
                "frame": encoded_image
            }
            if afterMPC:
                print(f"Sending request {req_id} | Queue delay: {queue_delay:.2f}s send after MPC")
            else:
                print(f"Sending request {req_id} | Queue delay: {queue_delay:.2f}s-- send before MPC")


            start_time = time.time()
            response = session.post(whisk_url, json=payload, headers=header, verify=False)
            end_time = time.time()

            http_latency = end_time - start_time
            total_delay = end_time - enqueue_time

            status = response.status_code
            try:
                resp_json = response.json()
                center = resp_json.get("center_point")
            except Exception:
                center = None

            print(f"Request {req_id} | Status: {status} | HTTP: {http_latency:.2f}s | Total: {total_delay:.2f}s | Center: {center} |WhenToSend:{source}")
            with open("MPC_request_response_tail_1.csv", "a") as f:
                if afterMPC:
                    f.write(f"{req_id},{enqueue_time},{dequeue_time},{http_latency:.3f},{total_delay:.3f},{status},{source}\n")

        except Exception as e:
            print(f"Request failed: {e}")


action_name = "object-detection"
pod_name = "wskopenwhisk-invoker-00-.*-guest-objectdetection"
prom = PrometheusClient(ip="localhost", port=9090) # work with p06
H = 5 # 5 steps not 5 seconds
MPC_INTERNAL_STEP_DURATION = 5.0
#arrival rate history
request_history = []
trace_window = 12 # how many recent sample to keep for forecast 12*5s = 60 seconds
step_index = 0
# Limit runtime
k = 62  # k minutes
MAX_RUNTIME_SECONDS = 600 * k
start_pp = time.time()

while True:
    queue_len = r.llen("request_queue")
    print(f"[INFO][{time.strftime('%H:%M:%S')}]  Queue length: {queue_len}")
    #print('It gonna be clear queue before mpc..............................')
    #send_thread = Thread(target=send_batch_gradual, args=(queue_len, 0.05, False))
    #send_thread.start()

    start = time.time()
    w = int(prom.get_warm_container_count(pod_name))
    print(f"[INFO] [{time.strftime('%H:%M:%S')}] Observed warm pods available: {w}")
    # Step 1: Pull arrival rate from Prometheus
    #rate = prom.get_request_rate(function=action_name, window="30s")
    rate = get_queue_arrival_rate()
    print(f"[INFO] [{time.strftime('%H:%M:%S')}] Observed request rate (last 5s): {rate:.2f}")

    request_history.append(rate)
    if len(request_history) > trace_window:
        request_history.pop(0)

    # Step 2: Forecast arrival rate
    if len(request_history) < trace_window:
        print(
            f"[WARN] [{time.strftime('%H:%M:%S')}] Insufficient history ({len(request_history)} points), skipping MPC decision.")
        # Fallback: send all available requests without MPC guidance
        print(f"[Fallback] Sending all queued requests without MPC...")
        print(f'warm now ={w}')
        if w > 1:
            update_semaphore_if_needed(w)
            update_executor_if_needed(w)

        else:
            update_semaphore_if_needed(queue_len)
            update_executor_if_needed(max(1, queue_len))


        # fallback dispatch ONLY here
        send_thread = Thread(target=send_batch_gradual, args=(queue_len, 0.05, False))
        send_thread.start()
        print(f"[Fallback] Dispatched {queue_len} requests.")
        step_index += 5
        next_step_time = start + 5
        time_to_wait = next_step_time - time.time()
        if time_to_wait > 0:
            print(colored(f"Sleeping now will be awake in {time_to_wait} seconds", "blue", "on_yellow"))
            time.sleep(time_to_wait)
        continue  # skip this loop iteration
    # Perform forecast and proceed with MPC if enough history
    forecasted_rates_per_second_per_block = forecast(request_history, n_predict=H, harmonics=5).tolist()
    print(
        f"[INFO][{time.strftime('%H:%M:%S')}] Forecasted avg rates for next {H} {MPC_INTERNAL_STEP_DURATION}s blocks (req/s per block): {[round(v, 2) for v in forecasted_rates_per_second_per_block]}")

    # Convert forecasted rates to *total arrivals per 5-second MPC block*
    lam_total_arrivals_per_block = [rate * MPC_INTERNAL_STEP_DURATION for rate in forecasted_rates_per_second_per_block]
    print(
        f"[INFO][{time.strftime('%H:%M:%S')}] Forecasted total arrivals for MPC (requests per block): {[round(v, 2) for v in lam_total_arrivals_per_block]}")

    forecast_log.log_forecast_data(rate, forecasted_rates_per_second_per_block[:1],
                                   log_file="log/forecast_log_tail_1.csv", current_step=step_index,
                                   step_interval=MPC_INTERNAL_STEP_DURATION)
    queue_len = r.llen("request_queue")
    #print(f"[INFO][{time.strftime('%H:%M:%S')}]  Queue length before go to MPC: {queue_len}")
    print(f"[INFO][{time.strftime('%H:%M:%S')}]  Before go to MPC: queue length: {queue_len}, w: {w}, lam: {lam_total_arrivals_per_block}")
    served, x0, r0, s0, s_values = mpc_revised_v3(H, q=queue_len, w=w, lam=lam_total_arrivals_per_block)

    print(
        f"[MPC][{time.strftime('%H:%M:%S')}] MPC returns: s0={s0}, x0={x0}, r0={r0}, servered={served}, s= {s_values}")


    print(colored(f"[MPC] Dispatching up to {s0} requests...", "yellow", "on_blue"))

    print(f"[MPC] MPC running in {time.time() - start} seconds")

    update_semaphore_if_needed(w)

    # Launch request dispatch immediately
    send_thread = Thread(target=send_batch_gradual, args=(s0, 5, True))
    send_thread.start()
    ttt = time.time()
    # Launch background reclaim/cold start
    Thread(target=maybe_reclaim, args=(r0, recently_deleted_pods)).start()
    Thread(target=maybe_cold_start, args=(x0,)).start()

    send_thread.join()  # optional: wait for sending to finish before next loop
    print(f'After cold start and reclaim, it takes {time.time() - ttt}......................')
    print('............')
    step_index += 5
    next_step_time = start + 0.5
    time_to_wait = next_step_time - time.time()
    if time_to_wait > 0:
        print(colored(f"Sleeping now will be awake in {time_to_wait} seconds", "blue", "on_yellow"))
        time.sleep(time_to_wait)


