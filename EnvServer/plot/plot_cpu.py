import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('cpu.csv')
df['ts'] = pd.to_datetime(df['ts'])

plt.figure(figsize=(10, 6))
plt.plot(df['ts'], df['cpu'], label='CPU Usage')

plt.title('CPU Usage Over Time')
plt.xlabel('Timestamp')
plt.ylabel('CPU Usage')
plt.legend()
plt.grid(True)

plt.show()

