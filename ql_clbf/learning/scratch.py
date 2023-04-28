import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_array(arr):
    # Create a colormap with red for 0 and green for other values
    cmap = mcolors.ListedColormap(['red', 'green'])
    norm = mcolors.BoundaryNorm([0, 0.5, np.max(arr)], cmap.N)

    # Plot the array using imshow
    plt.imshow(arr, cmap=cmap, norm=norm)

    # Add gridlines
    plt.grid(which='both', color='black', linewidth=2)
    plt.xticks(np.arange(-0.5, 4, 1))
    plt.yticks(np.arange(-0.5, 4, 1))

    # Remove ticks
    plt.gca().xaxis.set_tick_params(width=0)
    plt.gca().yaxis.set_tick_params(width=0)
    plt.gca().xaxis.set_ticklabels([])
    plt.gca().yaxis.set_ticklabels([])

    # Print the grid values in each cell
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            plt.text(j, i, round(arr[i, j], 3), ha='center', va='center', color='black', fontsize=12)

    # Display the plot
    plt.show()

if __name__ == "__main__":

    grid = np.load("desc_default.npy")
    plot_array(grid)