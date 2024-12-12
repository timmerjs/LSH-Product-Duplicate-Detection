import random
from datetime import datetime
import json
import time
import os
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

import utils
from bootstrapping import create_bootstraps
from dataStatistics import dataStats
from extract_model_words import add_all_model_words, extract_all_model_words, filter_freq_mw, save_freq_mw
from lsh import run_LSH, TestDataLSH
from msmp import msmp, hClustering, Alternative_hClustering
from msmp_fast import msmp_fast, hClustering_fast, Alternative_hClustering_fast
from performance_metrics import compute_f1_score

# TODO: make sure that the base directory exists \
#  and that the original data file is in there!
base_directory = "datatest"

# TODO: make sure that the base directory exists \
#  and that a small subset of the data is in there (~200 products)
test_base_directory = "smalldatatest"

# TODO: set the number of bootstraps (default is 50)!
NUMBER_OF_BOOTSTRAPS = 50


def main():

    t_values = [number / 100 for number in range(5, 95, 5)]
    epsilon_values = [number / 100 for number in range(5, 95, 5)]

    m = NUMBER_OF_BOOTSTRAPS
    initialisation(base_directory,m)  # TODO: Set up environment (can run this only once)

    # Initialise matrices to store results
    PC = np.zeros((len(t_values), m))
    PQ = np.zeros((len(t_values), m))
    F1_star = np.zeros((len(t_values), m))
    frac_comp = np.zeros((len(t_values), m))
    F1_MSM_FAST = np.zeros((len(epsilon_values), m))
    run_times = np.zeros((1, m))

    dataStats(base_directory)
    print(f"Initialisation successful, now iterating over all bootstraps...")
    print("\n")

    big_start = time.time()

    timestamp = datetime.now().strftime("%m-%d-%Y_%Hu%Mm%Ss")
    base_dir_plots = f"{base_directory}/plots"
    os.makedirs(base_dir_plots, exist_ok=True)

    # Create run-specific directory for saving plots and results during running
    run_folder = os.path.join(base_dir_plots, timestamp)
    os.makedirs(run_folder, exist_ok=True)

    print("############################# START LSH #############################")

    # Execute LSH for all bootstraps
    for k in range(1, (m + 1)):
        data_path = f"{base_directory}/bootstraps/bootstrap_{k}/TrainSample_{k}.json"
        mw_path = f"{base_directory}/bootstraps/bootstrap_{k}/regular_modelwords_{k}.json"
        ext_mw_path = f"{base_directory}/bootstraps/bootstrap_{k}/extended_modelwords_{k}.json"
        start = time.time()
        candidate_pairs, LSH_results = run_LSH(data_path, mw_path, ext_mw_path, t_values, f"{base_directory}/real_duplicates.json")
        end = time.time()
        print(f"Bootstrap {k} ended in {end-start:.4f} seconds.")
        PC[:, (k - 1)] = LSH_results[:, 0]
        PQ[:, (k - 1)] = LSH_results[:, 1]
        F1_star[:, (k - 1)] = LSH_results[:, 2]
        frac_comp[:, (k - 1)] = LSH_results[:, 3]

    # Already save these plots to a folder
    create_performance_plots_LSH(PC, PQ, F1_star, frac_comp, run_folder)
    print("\n")
    print(f"LSH analysis completed, plots can be found at \"{run_folder}\".")
    print("\n")
    print("########################## START MSMP-FAST ##########################")

    MSMP_results = Parallel(n_jobs=-1)(
        delayed(process_bootstrap)(k, base_directory, epsilon_values, run_folder, F1_MSM_FAST, run_times)
        for k in range(1, m + 1)
    )

    F1_MSM_FAST = np.array([result['F1_MSM_FAST'] for result in MSMP_results]).mean(axis=0)
    run_times = np.array([result['run_times'] for result in MSMP_results]).mean(axis=0)

    big_end = time.time()

    print(f"Average running time MSMP-FAST algorithm: {run_times.mean(axis=1)}.")

    print(f"TOTAL RUNNING TIME ON ALL BOOTSTRAPS: {big_end-big_start:.4f} seconds.")

    create_performance_plots_MSMP_FAST(F1_MSM_FAST, epsilon_values, run_folder)



def initialisation(base_directory, m):
    # Initialise all boostrap folders with their own dataset and corresponding set of model words
    # Note: this includes the data cleaning process
    print("Creating bootstraps...")
    create_bootstraps(base_directory, n_bootstraps=m)
    print(f"Created {m} bootstrap samples.")
    add_all_model_words(base_directory)  # Add the total set of model words (and extended ones from product attributes)
    utils.extract_all_brands(f"{base_directory}/TVs-all-merged.json", f"{base_directory}/all_brands.json")
    # Create a dictionary of all duplicate pairs
    utils.find_and_save_duplicates(f"{base_directory}/data_identify.json", f"{base_directory}/real_duplicates.json")

    # Also create a small dataset containing only the first 200 products. This is used for evaluating LSH
    with open(f"{base_directory}/data_identify.json", "r") as file:
        data = json.load(file)
    test_data = data[:200]
    with open(f"{base_directory}/TestData.json", "w") as testfile:
        json.dump(test_data, testfile, indent=2)
    test_mw, extended_test_mw = extract_all_model_words(f"{base_directory}/TestData.json")
    test_mw, extended_test_mw = filter_freq_mw(test_mw, extended_test_mw)
    save_freq_mw(test_mw, extended_test_mw, f"{base_directory}/test_mw.json", f"{base_directory}/test_ext_mw.json")


def process_bootstrap(k, base_directory, epsilon_values, run_folder, F1_MSM_FAST, run_times):
    data_path = f"{base_directory}/bootstraps/bootstrap_{k}/TrainSample_{k}.json"

    # Load data and prepare
    with open(data_path, "r") as file:
        data = json.load(file)

    with open(f"{base_directory}/real_duplicates.json", "r") as real_dupl:
        real_duplicates = json.load(real_dupl)

    real_pairs = set()
    for key, vs in real_duplicates.items():
        for value in vs:
            real_pairs.add((min(int(key), int(value)), max(int(key), int(value))))

    fake_candidate_matrix = np.ones((len(data), len(data)))
    start_run = time.time()
    distances_fast, _ = msmp_fast(data_path, fake_candidate_matrix, base_directory)
    end_run = time.time()
    run_times[0, (k - 1)] = end_run - start_run

    np.save(f"{base_directory}/bootstraps/bootstrap_{k}/MSMP_DIST.npy", distances_fast)
    # distances_fast = np.load(f"{base_directory}/bootstraps/bootstrap_{k}/MSMP_DIST.npy")
    for e, epsilon in enumerate(epsilon_values):
        predicted_duplicates = Alternative_hClustering_fast(distances_fast, epsilon, data)
        f1_score = compute_f1_score(real_pairs, predicted_duplicates)
        F1_MSM_FAST[e, (k - 1)] = f1_score

    print(f"MSMP-FAST completed on bootstrap {k} in {end_run - start_run:.4f} seconds.")

    # Save MSMP-FAST results for this bootstrap
    msmp_fast_path = os.path.join(run_folder, f"F1_MSM_FAST.npy")
    np.save(msmp_fast_path, F1_MSM_FAST)

    # Save run_times for both MSMP-FAST and MSMP:
    run_time_path = os.path.join(run_folder, f"run_times.npy")
    np.save(run_time_path, run_times)

    return {
        'F1_MSM_FAST': F1_MSM_FAST,
        'run_times': run_times
    }


def create_performance_plots_LSH(PC, PQ, F1_star, frac_comp, plots_folder):
    # Average matrices across bootstrap samples (column-wise)
    avg_PC = np.mean(PC, axis=1)
    avg_PQ = np.mean(PQ, axis=1)
    avg_F1_star = np.mean(F1_star, axis=1)
    avg_frac_comp = np.mean(frac_comp, axis=1)

    # Create three separate plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Pair Completeness Plot
    ax1.plot(avg_frac_comp, avg_PC, marker='o')
    ax1.set_xlabel('Fraction of Comparisons')
    ax1.set_ylabel('Pair Completeness')
    ax1.set_title('Pair Completeness')
    ax1.grid(True)

    # Pair Quality Plot
    ax2.plot(avg_frac_comp, avg_PQ, marker='s', color='green')
    ax2.set_xlabel('Fraction of Comparisons')
    ax2.set_ylabel('Pair Quality')
    ax2.set_title('Pair Quality')
    ax2.grid(True)

    # F1-star Plot
    ax3.plot(avg_frac_comp, avg_F1_star, marker='^', color='red')
    ax3.set_xlabel('Fraction of Comparisons')
    ax3.set_ylabel('F1-star')
    ax3.set_title('F1-star')
    ax3.grid(True)

    plt.tight_layout()
    lsh_plot_path = os.path.join(plots_folder, "LSH_plots.png")
    plt.savefig(lsh_plot_path)
    plt.close()


def create_performance_plots_MSMP_FAST(F1_MSM_FAST, epsilon_values, plots_folder):

    avg_F1_MSM_FAST = np.mean(F1_MSM_FAST, axis=1)
    plt.plot(epsilon_values, avg_F1_MSM_FAST, marker='s', label='F1 MSMP-FAST')
    plt.text(0.5, 0.5, 'Epsilon',
             horizontalalignment='center', verticalalignment='center')
    plt.title('Epsilon-based Metrics')

    # plt.tight_layout()
    msmp_plot_path = os.path.join(plots_folder, "MSMP-FAST_plot.png")
    plt.savefig(msmp_plot_path)
    plt.close()


def create_performance_plots_MSM_both(F1_MSM, F1_MSM_FAST, epsilon_values, plots_folder):
    # If you have epsilon-based metrics, you can add a second subplot
    # For this example, I'll use the same data, but you should replace with actual epsilon metrics
    plt.subplot(1, 2, 2)
    # Placeholder for epsilon-based metrics
    # Uncomment and modify when you have actual epsilon data
    avg_F1_MSM = np.mean(F1_MSM, axis=1)
    avg_F1_MSM_FAST = np.mean(F1_MSM_FAST, axis=1)
    plt.plot(epsilon_values, avg_F1_MSM, marker='o', label='F1 MSMP')
    plt.plot(epsilon_values, avg_F1_MSM_FAST, marker='s', label='F1 MSMP-FAST')
    plt.text(0.5, 0.5, 'Epsilon Metrics Placeholder',
             horizontalalignment='center', verticalalignment='center')
    plt.title('Epsilon-based Metrics')

    # plt.tight_layout()
    msmp_plot_path = os.path.join(plots_folder, "MSMP(-FAST)_plots.png")
    plt.savefig(msmp_plot_path)
    plt.close()


def testDataLSH(base_directory):

    print("\n")
    print("##################### START LSH ON SMALL DATASET #####################")
    utils.find_and_save_duplicates(f"{base_directory}/TestData.json", f"{base_directory}/testData_duplicates.json")
    results = TestDataLSH(base_directory)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Pair Completeness Plot
    ax1.plot(results[:, 3], results[:, 0], marker='o')
    ax1.set_xlabel('Fraction of Comparisons')
    ax1.set_ylabel('Pair Completeness')
    ax1.set_title('Pair Completeness')
    ax1.grid(True)

    # Pair Quality Plot
    ax2.plot(results[:, 3], results[:, 1], marker='s', color='green')
    ax2.set_xlabel('Fraction of Comparisons')
    ax2.set_ylabel('Pair Quality')
    ax2.set_title('Pair Quality')
    ax2.grid(True)

    # F1-star Plot
    ax3.plot(results[:, 3], results[:, 2], marker='^', color='red')
    ax3.set_xlabel('Fraction of Comparisons')
    ax3.set_ylabel('F1-star')
    ax3.set_title('F1-star')
    ax3.grid(True)

    plt.tight_layout()
    lsh_plot_path = f"{base_directory}/plots/TestDataLSH.png"
    plt.savefig(lsh_plot_path)
    plt.close()

    print(f"Finished testing LSH on a smaller dataset.")
    print(f"Results can be found at: \"{lsh_plot_path}\".")


def compare_MSMPs(base_directory, m):
    print("\n")
    print("################## START COMPARISON MSMP ALGORITHMS ##################")
    initialisation(base_directory, m)
    epsilon_values = [number / 100 for number in range(5, 95, 5)]
    F1_MSM = np.zeros((len(epsilon_values), m))
    F1_MSM_FAST = np.zeros((len(epsilon_values), m))
    run_times = np.zeros((2, m))

    timestamp = datetime.now().strftime("%m-%d-%Y_%Hu%Mm%Ss")
    base_dir_plots = f"{base_directory}/plots"
    os.makedirs(base_dir_plots, exist_ok=True)

    # Create run-specific directory for saving plots and results during running
    run_folder = os.path.join(base_dir_plots, timestamp)
    os.makedirs(run_folder, exist_ok=True)

    # Execute MSMP and MSMP-FAST algorithm separately
    for k in range(1, (m + 1)):
        data_path = f"{base_directory}/bootstraps/bootstrap_{k}/TrainSample_{k}.json"
        with open(data_path, "r") as file:
            data = json.load(file)
        fake_candidate_matrix = np.ones((len(data), len(data)))

        start_run = time.time()
        distances, not_used = msmp_fast(data_path, fake_candidate_matrix, base_directory)
        end_run = time.time()
        run_times[0, (k - 1)] = end_run - start_run
        print(f"MSMP-FAST completed on bootstrap {k} in {end_run - start_run:.4f} seconds.")

        with open(f"{base_directory}/real_duplicates.json", "r") as real_dupl:
            real_duplicates = json.load(real_dupl)
        real_pairs = set()
        for key, vs in real_duplicates.items():
            for value in vs:
                real_pairs.add((min(int(key), int(value)), max(int(key), int(value))))

        # Run MSMP_FAST
        for e, epsilon in enumerate(epsilon_values):
            predicted_duplicates = Alternative_hClustering_fast(distances, epsilon, data)
            f1_score = compute_f1_score(real_pairs, predicted_duplicates)
            F1_MSM_FAST[e, (k - 1)] = f1_score

        np.save(os.path.join(run_folder, "F1_MSM_FAST.npy"), F1_MSM_FAST)


        # Now for the regular MSMP method
        start_run = time.time()
        distances, not_used = msmp(data_path, fake_candidate_matrix, base_directory)
        end_run = time.time()
        run_times[1, (k - 1)] = end_run - start_run
        print(f"MSMP completed on bootstrap {k} in {end_run - start_run:.4f} seconds.")

        for e, epsilon in enumerate(epsilon_values):
            predicted_duplicates = hClustering_fast(distances, epsilon, data)
            f1_score = compute_f1_score(real_pairs, predicted_duplicates)
            F1_MSM[e, (k - 1)] = f1_score
        np.save(os.path.join(run_folder, "F1_MSM_matrix.npy"), F1_MSM)

    create_performance_plots_MSM_both(F1_MSM, F1_MSM_FAST, epsilon_values, run_folder)
    create_running_time_plot(run_times, run_folder)
    print(f"Finished comparing MSMP and MSMP-FAST on a smaller dataset.")
    print(f"Results can be found at: \"{run_folder}\".")


# After your existing code, add:
def create_running_time_plot(run_times, run_folder):
    """
    Create a bar plot comparing running times of MSMP-FAST and MSMP with error bars.

    Parameters:
    - run_times: 2xm numpy array where first row is MSMP-FAST times, second row is MSMP times
    - run_folder: directory to save the plot
    """
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

    # Optional: print out the detailed statistics
    print("MSMP-FAST Running Times:")
    print(f"Mean: {msmp_fast_mean:.4f} seconds")
    print(f"Standard Deviation: {msmp_fast_std:.4f} seconds")
    print("\nMSMP Running Times:")
    print(f"Mean: {msmp_mean:.4f} seconds")
    print(f"Standard Deviation: {msmp_std:.4f} seconds")




if __name__ == "__main__":
    main()
    testDataLSH("datatest")
    compare_MSMPs("smalldatatest", m=NUMBER_OF_BOOTSTRAPS)
