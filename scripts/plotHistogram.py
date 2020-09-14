import sys
from os.path import basename, splitext
import pandas as pd
import matplotlib.pyplot as plt
from itertools import zip_longest

if __name__ == '__main__':
    num_plots = len(sys.argv)-1
    if num_plots == 0:
        print(
            'Usage: python3 plotHistogram.py <csv_path> [<csv2_path>] [<csv3_path>] ... ')
    else:
        cols = 3
        rows = int(num_plots / cols if num_plots % cols == 0
                   else num_plots/cols+1)
        fig, axs = plt.subplots(rows, cols)
        for csv_path, ax in zip_longest(sys.argv[1:], axs.flat):
            if csv_path is None:
                fig.delaxes(ax)
            elif csv_path.endswith('.csv'):
                data = pd.read_csv(csv_path, header=None, skiprows=2)
                bins = data[0][0]
                del data[0]
                data = data.transpose()
                data.plot(kind='bar', legend=False,
                          xticks=range(0, bins, 10), rot=0, ax=ax)
                ax.set_xlabel('Bins')
                ax.set_ylabel('Frequencies')
                ax.set_title(splitext(basename(csv_path))[0])
        plt.tight_layout()
        plt.show()
