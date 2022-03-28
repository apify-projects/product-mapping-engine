import matplotlib.pyplot as plt
import numpy as np

# NOT USED METHODS
def analyse_dataset(data):
    """
    Compute percent of non zero values, create correlation matrix and plot data distribution for every feature
    @param data: Dataset to analyse
    @return:
    """
    print('\n\nDataset analysis')
    print('----------------------------')
    data_size = data.shape[0]
    for column in data:
        values = data[column].to_numpy()
        len_nonzero = np.count_nonzero(values)
        ratio_of_nonzero_ids = len_nonzero / data_size
        print(f'Pairs with nonzero {column} match: {round(100 * ratio_of_nonzero_ids, 2)}%')

    corr = data.iloc[:, :-1].corr()
    print('\n\nCorrelation matrix of features')
    print(corr.values)
    print('----------------------------')

    plot_features(data)


def plot_features(data):
    """
    Plot graph of data distribution for every feature
    @param data: dataset to visualize
    @return:
    """
    for column in data.iloc[:, :-1]:
        subset = data[[column, 'match']]
        groups = subset.groupby('match')
        for name, group in groups:
            plt.plot(np.arange(0, len(group)), group[column], marker="o", linestyle="", label=name)
        plt.title(f'Data distribution according to the {column} match value')
        plt.xlabel('Data pair')
        plt.ylabel(f'{column} match value')
        plt.legend(['0', '1'])
        plt.show()
        plt.clf()

    subset = data[['words', 'cos', 'match']]
    groups = subset.groupby('match')
    for name, group in groups:
        plt.plot(group['words'], group['cos'], marker="o", linestyle="", label=name)
    plt.title(f'Data distribution according to the words and cos match value')
    plt.xlabel('Words match value')
    plt.ylabel(f'Cos match value')
    plt.legend(['0', '1'])
    plt.show()
