import math
import time
import json
import requests
import base64
import cvxpy as cp
import numpy as np
from threading import Thread
from numpy.fft import fft, fftfreq
from prometheus.monitoring_function import PrometheusClient
from Reclaim import final_clean_pod_reclaim
from utility import forecast_log
from AutoScaler import prewarm
action_name = "object-detection"
pod_name = "wskopenwhisk-invoker-00-.*-guest-objectdetection"
prom = PrometheusClient(ip="localhost", port=9090) # work with p06
H = 5 # 5 steps not 5 seconds
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
def icebreaker_forecast(trace, n_predict=5, harmonics=10, clip_std_factor=3.0):
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

L_warm = 0.3
max_rps_fallback = 1
request_history = []
last_reclaim_time = 0
RECLAIM_COOLDOWN = 60  # seconds

while True:
    rate = prom.get_request_rate(function=action_name, window="30s")
    print(f'rate={rate}')
    request_history.append(rate)
    if len(request_history) > 12:  # keep 2 minutes of history (if sampled every 10s)
        request_history.pop(0)
    if len(request_history) < 5:
        predicted_rps = 1
        continue
    else:
        forecast = icebreaker_forecast(request_history, n_predict=1)
        predicted_rps = forecast[0]

    response_time_target = 0.38
    #required = math.ceil(predicted_rps * 1.3)
    required = max(1, math.ceil(predicted_rps * response_time_target*1.2*5))

    print(f'required={required}')
    current_estimated_idle = int(prom.get_warm_container_count(pod_name))
    print(f'current_estimated_idle={current_estimated_idle}')
    prewarm_needed = max(0, required - current_estimated_idle)
    reclaim = max(0,   current_estimated_idle - required)

    if prewarm_needed > 0:
        prewarm.prewarm_with_shell(prewarm_needed, action_name)
    now = time.time()
    if  now - last_reclaim_time > RECLAIM_COOLDOWN:
        print(f'continue reclaim as cool down time is over...{now - last_reclaim_time}')
        final_clean_pod_reclaim.reclaim_pods(reclaim)
        last_reclaim_time = now
    time.sleep(5)  # ⏱ forecast interval
