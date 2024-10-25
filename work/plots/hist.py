import numpy as np
import matplotlib.pyplot as plt


def simple_hist(X, name = "The dist that you didn't name", bars = None, bounds=None, mean_and_std = True, fig=None, ax=None):
    X_mean = np.nanmean(X)
    X_std = np.nanstd(X)
    if ax==None and fig ==None : 
        fig, ax = plt.subplots(1, 1)
    # Create histogram with manually adjusted bins
    bins = int(np.sqrt(len(X)))
    if bounds == None :
        bounds = (np.nanmin(X), np.nanmax(X))
    ax.hist(X, bins=bins, edgecolor='black', range = bounds, alpha=0.7)

    # Add mean and standard deviation to the plot
    if mean_and_std and bars is None:
        ax.axvline(X_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {X_mean:.2f}')
        ax.axvline(X_mean + X_std, color='green', linestyle='dashed', linewidth=2, label=f'Mean + Std Dev: {X_mean+X_std:.2f}')
        ax.axvline(X_mean - X_std, color='green', linestyle='dashed', linewidth=2, label=f'Mean - Std Dev: {X_mean-X_std:.2f}')
    elif bars is not None: 
        bar_values = bars[1:-1]
        red = (1,0,0)
        blue = (0,0,1)
        max_bar = np.max(bar_values)
        min_bar = np.min(bar_values)
        for bar in bar_values:
            cf = (bar - min_bar)/(max_bar - min_bar)
            color = tuple(b*(1-cf) + r * cf for b, r in zip(blue, red))
            ax.axvline(bar, color= color, linestyle='dashed', linewidth=2, label = f'{bar}')
    # Add labels and title
    ax.set_xlabel(name)
    ax.set_ylabel('Frequency')
    ax.set_title("Histogram of "+name)
    ax.legend()