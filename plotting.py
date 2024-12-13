import os

import numpy as np
from matplotlib import pyplot as plt


def create_performance_plots_LSH(PC, PQ, F1_star, frac_comp, plots_folder):
    # Average matrices across bootstrap samples (column-wise)
    avg_PC = np.mean(PC, axis=1)
    avg_PQ = np.mean(PQ, axis=1)
    avg_F1_star = np.mean(F1_star, axis=1)
    avg_frac_comp = np.mean(frac_comp, axis=1)

    # Create three separate plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Define font sizes
    title_fontsize = 16
    label_fontsize = 14

    # Pair Completeness Plot
    ax1.plot(avg_frac_comp, avg_PC, marker='o')
    ax1.set_xlabel('Fraction of Comparisons', fontsize=label_fontsize)
    ax1.set_ylabel('Pair Completeness', fontsize=label_fontsize)
    ax1.set_title('Pair Completeness', fontsize=title_fontsize)
    ax1.grid(True)

    # Pair Quality Plot
    ax2.plot(avg_frac_comp, avg_PQ, marker='s', color='green')
    ax2.set_xlabel('Fraction of Comparisons', fontsize=label_fontsize)
    ax2.set_ylabel('Pair Quality', fontsize=label_fontsize)
    ax2.set_title('Pair Quality', fontsize=title_fontsize)
    ax2.grid(True)

    # F1-star Plot
    ax3.plot(avg_frac_comp, avg_F1_star, marker='^', color='red')
    ax3.set_xlabel('Fraction of Comparisons', fontsize=label_fontsize)
    ax3.set_ylabel('F1-star', fontsize=label_fontsize)
    ax3.set_title('F1-star', fontsize=title_fontsize)
    ax3.grid(True)

    plt.tight_layout()
    lsh_plot_path = os.path.join(plots_folder, "LSH_plots.png")
    plt.savefig(lsh_plot_path)
    plt.close()


def create_performance_plots_MSMP_FAST(F1_MSM_FAST, epsilon_values, plots_folder):
    # Calculate the average F1 scores
    avg_F1_MSM_FAST = np.mean(F1_MSM_FAST, axis=1)

    # Create the plot
    plt.figure(figsize=(8, 6))  # Adjust figure size as needed
    plt.plot(epsilon_values, avg_F1_MSM_FAST, marker='s', label='_nolegend_')  # `_nolegend_` hides the legend

    # Add labels, title, and gridlines
    plt.xlabel(r'$\epsilon$', fontsize=12)  # Use Greek letter epsilon
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('MSMP-FAST results on all bootstraps', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)  # Add gridlines with light styling

    # Set x-axis ticks to increment by 0.1
    plt.xticks(np.arange(min(epsilon_values), max(epsilon_values) + 0.1, 0.1))

    # Save the plot
    msmp_plot_path = os.path.join(plots_folder, "MSMP-FAST_results.png")
    plt.savefig(msmp_plot_path, dpi=300)  # Save with higher resolution for better quality
    plt.close()


def create_performance_plots_MSM_both(F1_MSM, F1_MSM_FAST, epsilon_values, plots_folder):
    # Calculate the average F1 scores
    avg_F1_MSM = np.mean(F1_MSM, axis=1)
    avg_F1_MSM_FAST = np.mean(F1_MSM_FAST, axis=1)

    # Create the plot
    plt.figure(figsize=(8, 6))  # Adjust figure size as needed
    plt.plot(epsilon_values, avg_F1_MSM, marker='o', label='F1 MSMP')
    plt.plot(epsilon_values, avg_F1_MSM_FAST, marker='s', label='F1 MSMP-FAST')

    # Add labels, title, and legend
    plt.xlabel(r'$\epsilon$')  # Use Greek letter epsilon
    plt.ylabel('F1 Score')
    plt.title('MSMP vs MSMP-FAST for Different Values of $\epsilon$')
    plt.legend()
    plt.grid(True)  # Optional: Add grid lines for better readability

    # Set x-axis ticks to increment by 0.1
    plt.xticks(np.arange(min(epsilon_values), max(epsilon_values) + 0.1, 0.1))

    # Save the plot
    msmp_plot_path = os.path.join(plots_folder, "MSMP_vs_MSMP_FAST.png")
    plt.savefig(msmp_plot_path, dpi=300)  # Save with higher resolution for better quality
    plt.close()


def create_running_time_plot(run_times, run_folder):
    # Calculate mean and standard deviation for each method
    msmp_fast_times = run_times[0, :]
    msmp_times = run_times[1, :]

    # Compute statistics
    msmp_fast_mean = np.mean(msmp_fast_times)
    msmp_mean = np.mean(msmp_times)

    msmp_fast_std = np.std(msmp_fast_times)
    msmp_std = np.std(msmp_times)

    # Create the plot
    plt.figure(figsize=(5, 7))

    # Create bar plot with error bars
    bar_width = 0.35
    index = np.arange(2)

    plt.bar(index[0], msmp_fast_mean, bar_width,
            yerr=msmp_fast_std,
            capsize=10,
            label='MSMP-FAST',
            color='blue',
            alpha=0.7)

    plt.bar(index[1], msmp_mean, bar_width,
            yerr=msmp_std,
            capsize=10,
            label='MSMP',
            color='green',
            alpha=0.7)

    # Customize the plot
    plt.ylabel('Running Time (seconds)')
    plt.title('Comparison of MSMP-FAST and MSMP Running Times')
    plt.xticks(index, ['MSMP-FAST', 'MSMP'])
    plt.legend()

    # Add value labels on top of each bar
    plt.text(index[0], msmp_fast_mean, f'{msmp_fast_mean:.4f}',
             ha='center', va='bottom')
    plt.text(index[1], msmp_mean, f'{msmp_mean:.4f}',
             ha='center', va='bottom')

    # Save the plot
    plot_path = os.path.join(run_folder, "running_times_comparison.png")
    plt.savefig(plot_path)
    plt.close()

    print("\nMSMP-FAST Running Times:")
    print(f"Mean: {msmp_fast_mean:.4f} seconds")
    print(f"Standard Deviation: {msmp_fast_std:.4f} seconds")
    print("\nMSMP Running Times:")
    print(f"Mean: {msmp_mean:.4f} seconds")
    print(f"Standard Deviation: {msmp_std:.4f} seconds")


def plot_LSH_test(results, lsh_plot_path):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Define font sizes
    title_fontsize = 16
    label_fontsize = 14

    # Pair Completeness Plot
    ax1.plot(results[:, 3], results[:, 0], marker='o')
    ax1.set_xlabel('Fraction of Comparisons', fontsize=label_fontsize)
    ax1.set_ylabel('Pair Completeness', fontsize=label_fontsize)
    ax1.set_title('Pair Completeness', fontsize=title_fontsize)
    ax1.grid(True)

    # Pair Quality Plot
    ax2.plot(results[:, 3], results[:, 1], marker='s', color='green')
    ax2.set_xlabel('Fraction of Comparisons', fontsize=label_fontsize)
    ax2.set_ylabel('Pair Quality', fontsize=label_fontsize)
    ax2.set_title('Pair Quality', fontsize=title_fontsize)
    ax2.grid(True)

    # F1-star Plot
    ax3.plot(results[:, 3], results[:, 2], marker='^', color='red')
    ax3.set_xlabel('Fraction of Comparisons', fontsize=label_fontsize)
    ax3.set_ylabel('F1-star', fontsize=label_fontsize)
    ax3.set_title('F1-star', fontsize=title_fontsize)
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(lsh_plot_path)
    plt.close()

