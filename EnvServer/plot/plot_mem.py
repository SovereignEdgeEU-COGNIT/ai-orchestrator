import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('cpu.csv')
df['ts'] = pd.to_datetime(df['ts'])

plt.figure(figsize=(10, 6))
plt.plot(df['ts'], df['mem'], label='Memory Usage')

plt.title('Memory Usage Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Memory Usage')
plt.legend()
plt.grid(True)

plt.show()

