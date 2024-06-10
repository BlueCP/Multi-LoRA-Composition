import argparse

import numpy as np
import matplotlib.pyplot as plt

def main(args):

    arrays = []
    filenames = ('merge', 'switch', 'composite')
    labels = ('Merge', 'Switch', 'Composite')
    colours = ('red', 'green', 'blue')
    for filename in filenames:
        arrays.append(np.load(f'results/level1/{filename}.npy').mean(axis=1))
    for array, label, colour in zip(arrays, labels, colours):
        plt.plot(np.arange(1, 200), array, label=label, color=colour)
    # plt.vlines(np.arange(5, 200, 5), 0, 0.4, colors='gray')
    plt.title('Dissimilarity of high-level features between adjacent timesteps')
    plt.xlabel('Timestep')
    plt.ylabel('Normalised RMSE')
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int)
    args = parser.parse_args()
    main(args)