from src.base.base_model import BaseModel

import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
sns.set_style('darkgrid')

from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class DemandApproximator(BaseModel):

    def __init__(self):
        super().__init__()

    def decompose_demands(self, demands, seasonal_model = STL, show = False, **kwargs):
        results = seasonal_model(demands, **kwargs)
        if model == STL:
            results = results.fit()
        if show:
            results.plot()
            plt.show()

        return results

    def normalise_demands(self, real_demands, nominal_demands):
        # Divide the demand by the nominal value of the demand
        normalised_demands = real_demands.append(nominal_demands)

        normalised_demands = normalised_demands.dropna(axis = 1, how = 'any')
        normalised_demands = normalised_demands.loc[:, :].div(normalised_demands.iloc[-1]) / 1000

        normalised_demands.replace([np.inf, -np.inf], np.nan, inplace = True)
        normalised_demands = normalised_demands.dropna(axis = 1, how = 'any')
        normalised_demands = normalised_demands[:-1]

        # Change index into timestamp for calculations
        normalised_demands.reset_index(level = 0, inplace = True)
        normalised_demands.set_index(real_demands['Timestamp'], inplace = True)
        normalised_demands.drop(['index'], axis = 1, inplace = True)

        # Add small amount to allow multiplicative
        for column_name, column in normalised_demands.iteritems():
            normalised_demands[column_name] += 0.02
        return normalised_demands

    def plot_sample_demand_patter(self, periods = 3600 * 2 * 24, node = 'n2'):
        plt.plot(normalised_demands['n2'].loc[:periods], linewidth = 0.5)
        plt.show()

    def forward(self, nominal_demands):
        pass


if __name__ == '__main__':
    from utils.definitions import RAW_DATA_DIR
    from src.data_loader.water_network_model import EpanetModel
    from src.data_loader.dataset import get_demands
    from scipy.spatial.distance import euclidean

    import pandas as pd
    from dtaidistance import dtw
    from dtaidistance import dtw_visualisation as dtwvis, clustering

    epanet_file_path = RAW_DATA_DIR / 'EPANET' / 'L-TOWN.inp'
    wn = EpanetModel(epanet_file_path)

    nominal_demands = wn.get_nominal_demands()
    real_demands = get_demands()

    model = DemandApproximator()
    normalised_demands = model.normalise_demands(real_demands, nominal_demands)

    # Multiplicative decompose
    # normalised_demands = normalised_demands.loc[:, normalised_demands.mean(axis = 0) <= 1.5]
    normalised_demands = normalised_demands/normalised_demands.mean(axis = 0)
    results = model.decompose_demands(normalised_demands.to_numpy(),
                                      seasonal_model = seasonal_decompose,
                                      show = True,
                                      period = 288 * 7,
                                      model = 'multiplicative')

    trends = results.trend
    seasonals = results.seasonal
    # truncated_trends = trends[~np.all(np.isnan(trends), axis = 1)]
    trends = pd.DataFrame(trends, index = normalised_demands.index)

    # Plot trends
    plt.plot(trends, alpha = .4, linewidth = .1)
    plt.xlabel('Date')
    plt.ylabel('Q')
    plt.title('Normalised demands trend')
    plt.savefig('../../out/demands.png')


    # Plot seasonal
    # plt.plot(seasonals, alpha = 0.2)
    # plt.show()
