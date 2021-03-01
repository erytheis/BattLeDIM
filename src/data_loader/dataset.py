import pandas as pd
from utils.definitions import RAW_DATA_DIR


def get_pressures():
    pressures_file_path = RAW_DATA_DIR / '2018 SCADA' / 'Pressures.csv'
    return get_contest_csv(pressures_file_path)


def get_contest_csv(file_path):
    """
    Extracts historical data from the contest dataset
    :param file_path:
    :return:
    """
    data = pd.read_csv(file_path,
                       sep = ';',
                       decimal = ',')
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])

    year_start = data['Timestamp'].iloc[0]
    data.set_index((data['Timestamp'] - year_start).dt.total_seconds(), inplace = True)
    data.index = data.index.astype(int)
    return data


def get_demands(demands_file_path = None):
    if demands_file_path is None:
        demands_file_path = RAW_DATA_DIR / '2018 SCADA' / 'Demands.csv'
    return get_contest_csv(demands_file_path)


if __name__ == '__main__':
    # For testing purposes
    demands = get_demands()
    print(demands.index.to_list()[:100])
    print(demands.shape)
