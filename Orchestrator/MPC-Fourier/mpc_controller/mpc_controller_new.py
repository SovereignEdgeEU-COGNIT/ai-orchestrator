# Full MPC Scheduler with Forecast Integration for Multiple Functions + Reclaim Logic

import numpy as np
import cvxpy as cp
#from prometheus_api_client import PrometheusConnect
from datetime import datetime, timedelta
import pandas as pd
import time

# === Parameters ===
H = 5  # MPC horizon
eta = 0.05
L_warm = 0.05
L_cold = 0.4
#L_cold = {"func-a": 0.4, "func-b": 0.6, "func-c": 0.3} # Example values
delta = 0.001
rho = 0.0001  # Reclaim penalty weight (tune as needed)
harmonics = 5  # for Fourier extrapolation

# === Prometheus connection ===
#prom = PrometheusConnect(url="http://localhost:9090", disable_ssl=True)

# === Function: fetch recent request rates ===
def get_request_rate(function_name):
    query = f'rate(openwhisk_user_activation_count_total{{function_name="{function_name}"}}[1m])'
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=5)
    data = prom.custom_query_range(query=query, start_time=start_time, end_time=end_time, step='15s')
    if not data or not data[0]["values"]:
        return [0] * 20
    return [float(x[1]) for x in data[0]["values"]]

# === Function: Fourier forecast ===
def fourier_forecast(trace, n_predict=5, harmonics=5):
    trace = np.array(trace)
    n = trace.size
    t = np.arange(n)
    p = np.polyfit(t, trace, 1)
    detrended = trace - p[0] * t
    freq = np.fft.fftfreq(n)
    fft_values = np.fft.fft(detrended)
    indexes = list(range(n))
    indexes.sort(key=lambda i: abs(freq[i]))
    t_future = np.arange(n + n_predict)
    future = np.zeros(t_future.size)
    for i in indexes[:1 + harmonics * 2]:
        amp = np.abs(fft_values[i]) / n
        phase = np.angle(fft_values[i])
        future += amp * np.cos(2 * np.pi * freq[i] * t_future + phase)
    return (future + p[0] * t_future)[-n_predict:]



#initial queue length q0, warm container w0, cold container provisioned at t =0, penalty coefficient for queue length
def run_mpc_multi(lambda_forecasts, q0=5, w0=3, c0=2, penalty_q=0.9):
    F = len(lambda_forecasts)
    H = len(lambda_forecasts[0])

    if not isinstance(q0, list): q0 = [q0] * F
    if not isinstance(w0, list): w0 = [w0] * F
    if not isinstance(c0, list): c0 = [c0] * F

    # === Decision Variables ===
    a_warm = cp.Variable((F, H), nonneg=True)
    a_wait = cp.Variable((F, H), nonneg=True)
    a_cold = cp.Variable((F, H), nonneg=True)
    p = cp.Variable((F, H), nonneg=True)
    r = cp.Variable((F, H), nonneg=True)

    # === Initialize symbolic states ===
    q_sym = [[cp.Constant(q0[f])] for f in range(F)]
    w_sym = [[cp.Constant(w0[f])] for f in range(F)]
    c_sym = [[cp.Constant(c0[f])] for f in range(F)]

    # === Symbolic state evolution ===
    for t in range(H):
        for f in range(F):
            #demand = lambda_forecasts[f][t] + q_sym[f][-1]
            #served = a_warm[f, t] + a_cold[f, t] + a_wait[f, t]
            q_next = q_sym[f][-1] + lambda_forecasts[f][t] - a_warm[f, t] - a_cold[f, t]
            #w_next = w_sym[f][-1] + c_sym[f][-1] - a_warm[f, t] - r[f, t]
            w_next = w_sym[f][-1] - a_warm[f, t] - r[f, t] + p[f, t]  # Provisioning increases warm count

            c_next = c_sym[f][-1] - p[f, t]  # Provisioning decreases cold availability
            #c_next = p[f, t]
            q_sym[f].append(q_next)
            w_sym[f].append(w_next)
            c_sym[f].append(c_next)

    # === Objective and Constraints ===
    total_cost = 0
    constraints = []
    idle_penalty_weight = 0.5
    for f in range(F):
        for t in range(H):
            demand = lambda_forecasts[f][t] + q_sym[f][t]
            expected_min_demand = min(lambda_forecasts[f])
            warm_overage = cp.maximum(w_sym[f][t] - expected_min_demand, 0)
            if t == 0:
                print(f"[Debug] func_{f} warm_overage={warm_overage}")
            total_cost += penalty_q * q_sym[f][t] \
                        + cp.multiply(a_cold[f, t], (L_cold + L_warm)) \
                        + cp.multiply(p[f, t], delta) \
                        + cp.multiply(r[f, t], rho)\
                        + idle_penalty_weight * warm_overage

            constraints += [
                a_warm[f, t] + a_cold[f, t] + a_wait[f, t] == demand,
                a_warm[f, t] <= w_sym[f][t],
                a_cold[f, t] <= c_sym[f][t],
                r[f, t] <= w_sym[f][t],
                a_wait[f, t] >= 0
            ]

    # === Solve ===
    problem = cp.Problem(cp.Minimize(total_cost), constraints)
    problem.solve(solver=cp.ECOS)

    results = []
    if problem.status not in [cp.OPTIMAL_INACCURATE, cp.INFEASIBLE]:
        a_warm_val = a_warm.value
        a_wait_val = a_wait.value
        a_cold_val = a_cold.value
        p_val = p.value
        r_val = r.value
        q_values = [[q0[f]] for f in range(F)]
        for t in range(H):
            for f in range(F):
                q_next_val = q_values[f][-1] + lambda_forecasts[f][t] - a_warm_val[f, t] - a_cold_val[f, t]
                q_values[f].append(q_next_val)

        for f in range(F):
            for t in range(H):
                results.append({
                    "Function": f"func_{f}",
                    "Time Step": f"t+{t}",
                    "Predicted Arrivals": round(lambda_forecasts[f][t], 2),
                    "Serve by Warm": int(round(a_warm_val[f, t], 2)),
                    "Wait in Queue": int(round(a_wait_val[f, t], 2)),
                    "Cold Start": int(round(a_cold_val[f, t], 2)),
                    "Provision Cold": int(round(p_val[f, t], 2)),
                    "Reclaim Warm": int(round(r_val[f, t], 2)),
                    "Queue Length": int(round(float(q_values[f][t + 1]), 2))
                })
    else:
        print(f"Problem status: {problem.status}")
        return []

    return results

def get_request_rate_mock_data( seed_offset =0):
    np.random.seed(55 + seed_offset)
    return [10 + np.sin(i / 2) + np.random.normal(0, 0.5) for i in range(20)]
def simulate_dummy_data():
    # Simulation parameters
    function_names = ["func-a"]#, "func-b", "func-c"]
    H = 5  # MPC horizon
    num_rounds = 20  # increase to 20 rounds to observe reclaim
    interval = 1  # 1 second per round

    # Initial state for each function
    q_state = {f: 0 for f in function_names}
    w_state = {f: 3 for f in function_names}
    c_state = {f: 2 for f in function_names}

    for round_num in range(num_rounds):
        print(f"\n===== Decision Round {round_num} @ {datetime.now().strftime('%H:%M:%S')} =====")

        # Step 1: generate or override forecast
        lambda_forecasts = []
        for i, fname in enumerate(function_names):
            if round_num < 10:
                history = get_request_rate_mock_data(seed_offset=i)
                forecast = fourier_forecast(history, n_predict=H)
            if round_num >= 10:
            # Linearly decay to zero over rounds 10–19
                decay_factor = max(0, 1 - (round_num - 10) / 10)  # from 1 to 0 over 10 rounds
                forecast = [x * decay_factor for x in forecast]
            lambda_forecasts.append(forecast)

        # Print current state
        for i, fname in enumerate(function_names):
            print(
                f"{fname} [BEFORE]: Warm={w_state[fname]}, Queue={q_state[fname]}, Forecast[0]={round(lambda_forecasts[i][0], 2)}")

        # Step 2: MPC decision
        results = run_mpc_multi(
            lambda_forecasts,
            q0=[q_state[f] for f in function_names],
            w0=[w_state[f] for f in function_names],
            c0=[c_state[f] for f in function_names],
            penalty_q=0.1
        )

        # Step 3: apply decisions and print
        for i, fname in enumerate(function_names):
            row = next(r for r in results if r["Function"] == f"func_{i}" and r["Time Step"] == "t+0")
            print(
                f"{fname}: Serve by warm={row['Serve by Warm']} | Wait={row['Wait in Queue']} | "
                f"Cold={row['Cold Start']} | Provision={row['Provision Cold']} | Reclaim={row['Reclaim Warm']} | "
                f"Queue={row['Queue Length']}"
            )
            q_state[fname] = row["Queue Length"]
            w_state[fname] += row["Provision Cold"] - row["Reclaim Warm"]
            c_state[fname] = row["Provision Cold"]

        time.sleep(interval)

# === Run everything ===
if __name__ == "__main__":
    run_simulate = True
    if run_simulate:
        simulate_dummy_data()
    else:
        function_names = ["func-a", "func-b", "func-c"]
        lambda_forecasts = []
        for fname in function_names:
            # history = get_request_rate(fname)
            history = get_request_rate_mock_data()
            forecast = fourier_forecast(history, n_predict=H)
            lambda_forecasts.append(forecast)

        print(lambda_forecasts)
        results = run_mpc_multi(lambda_forecasts)
        # tools.display_dataframe_to_user(name="Multi-Function MPC Output", dataframe=pd.DataFrame(results))
        df = pd.DataFrame(results)
        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.width', None)  # Don't break lines
        pd.set_option('display.max_colwidth', None)  # Show full cell content

        print(df)


