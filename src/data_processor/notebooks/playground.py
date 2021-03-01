import wntr

import pandas as pd
import matplotlib.pyplot as plt

from datetime import date

demands_file_path = '../../../data/raw/2018 SCADA/Demands.csv'
demands = pd.read_csv(demands_file_path,
                      sep = ';',
                      decimal = ',')

demands['Timestamp'] = pd.to_datetime(demands['Timestamp'])
demands['date'] = demands['Timestamp'].dt.date

plt.plot(demands[demands['date'] == date(2018, 1, 1)].iloc[:, 1:10])
plt.show()

pressures_file_path = '../../../data/raw/2018 SCADA/Pressures.csv'
pressures = pd.read_csv(pressures_file_path,
                        sep = ';',
                        decimal = ',')

pressures['Timestamp'] = pd.to_datetime(pressures['Timestamp'])
pressures['date'] = pressures['Timestamp'].dt.date

plt.plot(pressures[pressures['date'] == date(2018, 1, 1)].iloc[:, 1:-1])
plt.show()

epanet_file_path = '../../../data/raw/EPANET/L-TOWN_no_pattern.inp'
wn = wntr.network.WaterNetworkModel(epanet_file_path)

sim = wntr.sim.EpanetSimulator(wn)
results = sim.run_sim()

for i in range(5):
    pressure = results.node['pressure'].loc[i * 3600, :]
    wntr.graphics.plot_network(wn,
                               node_attribute = pressure,
                               node_size = 12,
                               title = 'Pressure at {} hours'.format(i))
    plt.show()
graph = wn.get_graph()
