import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

def main():
    with open("after_collate.pkl", 'rb') as f:
        input_data = pickle.load(f)
    
    high_res_raster = input_data["high_res_raster"].squeeze()
    low_res_raster = input_data["low_res_raster"].squeeze()

    for i in range(high_res_raster.shape[2]):
        high = high_res_raster[:, :, i].numpy()
        low = low_res_raster[:, :, i].numpy()
        plt.imshow(high, cmap='gray')
        plt.savefig(f'imgs/index_{i}_high.png')
        plt.imshow(low, cmap='gray')
        plt.savefig(f'imgs/index_{i}_low.png')



if __name__ == "__main__":
    main()