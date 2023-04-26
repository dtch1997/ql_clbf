import numpy as np

if __name__ == "__main__":

    grid_A = np.load("desc_A.npy")
    grid_B = np.load("desc_B.npy")
    grid_AB = np.load("desc_AB.npy")

    print(grid_A)
    print(grid_B)
    print(grid_AB)

    diff = np.minimum(grid_A, grid_B) - grid_AB
    print(np.around(diff, 3))