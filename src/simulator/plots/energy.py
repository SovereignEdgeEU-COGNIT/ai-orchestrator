import numpy as np
import matplotlib.pyplot as plt
 
def generate_weather_sequence(initial_state, transition_matrix, num_days):
    states = ['rainy', 'sunny', 'cloudy']
    current_state = initial_state
    sequence = [current_state]

    for _ in range(num_days - 1):
        next_state = np.random.choice(states, p=transition_matrix[states.index(current_state)])
        sequence.append(next_state)
        current_state = next_state

    return np.array(sequence)

transition_matrix = [
    [0.7, 0.1, 0.2],  # raining to [raining, sunny, cloudy]
    [0.1, 0.6, 0.3],  # sunny to [raining, sunny, cloudy]
    [0.3, 0.2, 0.5]   # cloudy to [raining, sunny, cloudy]
]

weather_data = generate_weather_sequence('sunny', transition_matrix, 100)

# generate synthetic genergy mix data based on weather conditions
def generate_energy_mix(weather_state):
    if weather_state == "sunny":
        green_energy = np.random.beta(5, 1.5)
    elif weather_state == "cloudy":
        green_energy = np.random.beta(2, 5)
    elif weather_state == "rainy":
        green_energy = np.random.beta(1.1, 10)

    non_green_energy = 1 - green_energy

    return green_energy, non_green_energy

energy_mix_data = [generate_energy_mix(weather) for weather in weather_data]
energy_mix_data = np.array(energy_mix_data)
 

# Plot the generated energy mix time series
plt.figure(figsize=(10, 6))
plt.plot(energy_mix_data[:, 0], label="Green Energy", color="green")
plt.plot(energy_mix_data[:, 1], label="Non-Green Energy", color="orange")
plt.xlabel("Time")
plt.ylabel("Energy Mix")
plt.title("Generated Green vs Non-Green Energy Mix Time Series")

plt.legend()

plt.show()
