import argparse

import numpy as np
import matplotlib.pyplot as plt

def main(args):
    # arrays = []
    # filenames = ('sdxlt_parti', 'sdxlt_coco2017', 'sdv1-5_parti', 'sdv1-5_coco2017')
    # labels = ('SDXL Turbo, parti', 'SDXL Turbo, coco2017', 'SD v1.5, parti', 'SD v1.5, coco2017')
    # colours = ('red', 'orange', 'green', 'blue')
    # for filename in filenames:
    #     arrays.append(np.load(f'experiments/similarity/{filename}.npy').swapaxes(0, 1).mean(axis=2)[1])
    # for array, label, colour in zip(arrays, labels, colours):
    #     plt.plot(np.linspace(0.0, 100.0, len(array)), array, label=label, color=colour)
    # plt.title('Dissimilarity of high-level features between adjacent timesteps')
    # plt.xlabel('Denoising progress (%)')
    # plt.ylabel('Normalised RMSE')
    # plt.grid()
    # plt.legend()
    # # plt.savefig('figure.png', dpi=300)
    # plt.show()

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