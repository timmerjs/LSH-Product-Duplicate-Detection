from datetime import datetime
import json
import time
import os
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import utils
from bootstrapping import create_bootstraps
from extract_model_words import add_all_model_words, extract_all_model_words, filter_freq_mw, save_freq_mw
from lsh import run_LSH
from msmp import msmp, hClustering
from msmp_fast import msmp_fast, hClustering_fast, Alternative_hClustering_fast
from performance_metrics import compute_f1_score


def main():
    # TODO: make sure that the base directory exists \
    #  and that the original data file is in there!
    base_directory = "data4"

    # TODO: check these parameter values!
    NUMBER_OF_BOOTSTRAPS = 1
    t_values = [number / 100 for number in range(5, 95, 5)]
    epsilon_values = [number / 100 for number in range(5, 95, 5)]

    m = NUMBER_OF_BOOTSTRAPS
    initialisation(base_directory,m)  # Set up environment (run this only once)

    # Initialise matrices to store results
    PC = np.zeros((len(t_values), m))
    PQ = np.zeros((len(t_values), m))
    F1_star = np.zeros((len(t_values), m))
    frac_comp = np.zeros((len(t_values), m))
    F1_MSM = np.zeros((len(epsilon_values), m))
    F1_MSM_FAST = np.zeros((len(epsilon_values), m))
    run_times = np.zeros((2, m))

    # TODO: code snippet for printing the number of duplicates etc
    # TODO: code snippet for printing the number of duplicates etc
    # TODO: code snippet for printing the number of duplicates etc
    # TODO: and data statistics methods

    print(f"Initialisation successful, now iterating over all bootstraps...")
    big_start = time.time()

    timestamp = datetime.now().strftime("%m-%d-%Y_%Hu%Mm%Ss")
    base_dir_plots = f"{base_directory}/plots"
    os.makedirs(base_dir_plots, exist_ok=True)

    # Create run-specific directory for saving plots and results during running
    run_folder = os.path.join(base_dir_plots, timestamp)
    os.makedirs(run_folder, exist_ok=True)

    # Execute LSH for all bootstraps
    for k in range(1, (m + 1)):
        data_path = f"{base_directory}/bootstrap_{k}/TrainSample_{k}.json"
        mw_path = f"{base_directory}/bootstrap_{k}/regular_modelwords_{k}.json"
        ext_mw_path = f"{base_directory}/bootstrap_{k}/extended_modelwords_{k}.json"
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

    # Execute MSMP and MSMP-FAST algorithm separately
    for k in range(1, (m + 1)):
        data_path = f"{base_directory}/bootstrap_{k}/TrainSample_{k}.json"
        with open(data_path, "r") as file:
            data = json.load(file)
        fake_candidate_matrix = np.ones((len(data), len(data)))

        start_run = time.time()
        distances, not_used = msmp_fast(data_path, fake_candidate_matrix)
        end_run = time.time()
        run_times[0, (k - 1)] = end_run - start_run
        print(f"MSMP-FAST completed on bootstrap {k} in {end_run - start_run:.4f} seconds.")


        np.save(f"{base_directory}/bootstrap_{k}/MSMP_FAST_DIST.npy", distances)
        distances = np.load(f"{base_directory}/bootstrap_{k}/MSMP_FAST_DIST.npy")
        with open(f"{base_directory}/real_duplicates.json", "r") as real_dupl:
            real_duplicates = json.load(real_dupl)
        real_pairs = set()
        for key, vs in real_duplicates.items():
            for value in vs:
                real_pairs.add((min(int(key), int(value)), max(int(key), int(value))))
        for e, epsilon in enumerate(epsilon_values):
            predicted_duplicates = Alternative_hClustering_fast(distances, epsilon, data)
            f1_score = compute_f1_score(real_pairs, predicted_duplicates)
            #print(f"Epsilon: {epsilon}, F1_score = {f1_score}")
            F1_MSM_FAST[e, (k - 1)] = f1_score

        np.save(os.path.join(run_folder, "F1_MSM_FAST_matrix.npy"), F1_MSM_FAST)

    #
    #     # Now for the regular MSMP method
    #     start_run = time.time()
    #     distances, not_used = msmp(data_path, fake_candidate_matrix)
    #     end_run = time.time()
    #     run_times[1, (k - 1)] = end_run - start_run
    #     print(f"MSMP completed on bootstrap {k} in {end_run - start_run:.4f} seconds.")
    #
    #     np.save(f"{base_directory}/bootstrap_{k}/MSMP_DIST.npy", distances)
    #     distances = np.load(f"{base_directory}/bootstrap_{k}/MSMP_DIST.npy")
    #     for e, epsilon in enumerate(epsilon_values):
    #         predicted_duplicates = hClustering_fast(distances, epsilon, data)
    #         f1_score = compute_f1_score(real_pairs, predicted_duplicates)
    #         #print(f"Epsilon: {epsilon}, F1_score = {f1_score}")
    #         F1_MSM[e, (k - 1)] = f1_score
    #     np.save(os.path.join(run_folder, "F1_MSM_matrix.npy"), F1_MSM)
    # big_end = time.time()
    # print("\n")
    # print(f"TOTAL RUNNING TIME: {big_end-big_start:.4f} seconds.")

    create_performance_plots_MSM(F1_MSM, F1_MSM_FAST, epsilon_values, run_folder)






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


def create_performance_plots_MSM(F1_MSM, F1_MSM_FAST, epsilon_values, plots_folder):
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



if __name__ == "__main__":
    main()
