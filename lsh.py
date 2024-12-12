import collections
import json
from itertools import combinations
import numpy as np
from performance_metrics import compute_f1_star, compute_pair_quality, compute_pair_completeness
from utils import extract_model_words_title, compute_duplicates
from utils import extract_extended_model_words


def create_bin_vec(products, all_model_words, all_extended_mw):
    big_binary_matrix = np.full(((len(all_model_words)+len(all_extended_mw)), len(products)), 0)
    for j, product in enumerate(products):
        title_mw = extract_model_words_title(product["title"])
        kvp_mw, extended_kvp_mw = extract_extended_model_words(product["featuresMap"])
        title_mw_set = set(title_mw)
        kvp_mw_set = set(kvp_mw)
        extended_kvp_set = set(extended_kvp_mw)
        all_product_mw = title_mw_set.union(kvp_mw_set)
        for i, model_word in enumerate(all_model_words):
            if model_word in all_product_mw:
                big_binary_matrix[i, j] = 1
        for k, extended_model_word in enumerate(all_extended_mw):
            if extended_model_word in extended_kvp_set:
                big_binary_matrix[len(all_model_words) + k, j] = 1
    return big_binary_matrix


def minHashing(binary_matrix):
    num_rows, num_columns = binary_matrix.shape

    # We take half the number of rows as number of rows for our signature matrix
    signature_proportion = 0.5
    n_hashes = int((signature_proportion * num_rows) // 1)
    if n_hashes % 2 != 0:  # Make sure the number of rows is an even number, needed for b and r computation later on.
        n_hashes += 1

    # Initialize signature matrix with infinity
    signature_matrix = np.full((n_hashes, num_columns), np.inf)

    # Generate n_hashes simple hash functions: h_i(x) = (a * x + b) % num_rows
    # a and b are random coefficients
    # See "Mining of Massive datasets" - Leskovec, Rajaraman & Ullman, chapter 3.3
    np.random.seed(42)  # For reproducibility
    a = np.random.randint(low=0, high=10 * num_rows, size=n_hashes)
    b = np.random.randint(low=1, high=10 * num_rows, size=n_hashes)

    # Process each row in the binary matrix
    for row_index in range(num_rows):
        # Compute hash values for the current row index
        hash_values = (a * row_index + b) % n_hashes

        # Update the signature matrix for each column where the row has a 1
        for col_index in range(num_columns):
            if binary_matrix[row_index, col_index] == 1:
                signature_matrix[:, col_index] = np.minimum(signature_matrix[:, col_index], hash_values)

    return signature_matrix


def find_b_r(num_rows, t):
    min_distance = float('inf')  # Initialize with infinity
    best_r = 2  # Start with default value for r
    best_b = num_rows // 2  # Default b is num_rows / 2
    n = num_rows

    # We allow the algorithm to take less rows of the signature matrix, to find most optimal b and r
    for i in range(30):
        n = num_rows - i
        for r in range(2, (1 + n//2)):  # Iterate over r from 1 to 100
            if n % r == 0:  # Only consider cases where n is divisible by r
                b = n // r  # Compute b
                distance = abs((1 / b) ** (1 / r) - t)  # Calculate the distance to threshold

                if distance < min_distance:  # Update best values if closer to target
                    min_distance = distance
                    best_r = r
                    best_b = b
                    #print(f"For t = {t}, n = {n}, b = {b}, r = {r}. Distance = {distance}.")

    return best_b, best_r, n


def locality_sensitive_hashing(signature_matrix, t):
    num_rows, num_columns = signature_matrix.shape
    b, r, new_num_rows = find_b_r(num_rows, t)

    band_buckets = []  # List to store buckets for each band

    for band_idx in range(b):
        start_row = band_idx * r
        end_row = start_row + r

        # Extract the band for each column
        band_matrix = signature_matrix[start_row:end_row, :]

        # Create buckets for this band
        buckets = {}
        for col_idx in range(num_columns):
            band_signature = tuple(band_matrix[:, col_idx])  # Convert to a hashable type (tuple)
            if band_signature not in buckets:
                buckets[band_signature] = []
            buckets[band_signature].append(col_idx)

        # Add non-trivial buckets (with more than one column) to the list
        band_buckets.append({key: value for key, value in buckets.items() if len(value) > 1})

    return band_buckets


# Example: Determine candidate pairs
def get_candidate_pairs(band_buckets, num_products):
    # Initialize an empty candidate matrix
    candidate_matrix = np.zeros((num_products, num_products), dtype=int)

    for buckets in band_buckets:
        for bucket in buckets.values():
            for i, j in combinations(bucket, 2):
                candidate_matrix[i, j] = 1
                candidate_matrix[j, i] = 1  # Symmetry

    return candidate_matrix


def run_LSH(products_path, model_words_path, extended_mw_path, t_values, real_duplicate_path):
    with open(products_path, "r") as file:
        data = json.load(file)
    with open(model_words_path, "r") as mw_file:
        all_model_words = json.load(mw_file)
    with open(extended_mw_path, "r") as mw_file:
        all_extended_mw = json.load(mw_file)
    with open(real_duplicate_path, "r") as file:
        real_duplicates = json.load(file)
    results_matrix = np.zeros((len(t_values), 4))

    binary_vectors = create_bin_vec(data, all_model_words, all_extended_mw)
    signature_matrix = minHashing(binary_vectors)
    for idx, t in enumerate(t_values):
        buckets = locality_sensitive_hashing(signature_matrix, t)
        candidate_pair_matrix = get_candidate_pairs(buckets, len(data))
        predicted_duplicates = compute_duplicates(candidate_pair_matrix)
        real_pairs = set()
        for k, vs in real_duplicates.items():
            for v in vs:
                real_pairs.add((min(int(k), int(v)), max(int(k), int(v))))

        predicted_pairs = set()
        for k, vs in predicted_duplicates.items():
            for v in vs:
                predicted_pairs.add((min(int(k), int(v)), max(int(k), int(v))))

        pq = compute_pair_quality(real_pairs, predicted_pairs)
        pc = compute_pair_completeness(real_pairs, predicted_pairs)
        f1_star = compute_f1_star(real_pairs, predicted_pairs)
        total_possible_pairs = len(data) * (len(data) - 1) / 2
        frac_comparison = len(predicted_pairs) / total_possible_pairs

        results_matrix[idx, :] = np.array([pc, pq, f1_star, frac_comparison])

    return candidate_pair_matrix, results_matrix


def LSH(signature_matrix, threshold, n_products):
    buckets = locality_sensitive_hashing(signature_matrix,threshold)
    candidate_pairs = get_candidate_pairs(buckets, n_products)
    return candidate_pairs


def TestDataLSH(base_directory):
    with open(f"{base_directory}/testData_duplicates.json", "r") as file:
        real_duplicates = json.load(file)

    products_path = f"{base_directory}/TestData.json"
    model_words_path = f"{base_directory}/test_mw.json"
    extended_mw_path = f"{base_directory}/test_ext_mw.json"
    with open(products_path, "r") as file:
        data = json.load(file)
    with open(model_words_path, "r") as mw_file:
        all_model_words = json.load(mw_file)
    with open(extended_mw_path, "r") as mw_file:
        all_extended_mw = json.load(mw_file)

    binary_vectors = create_bin_vec(data, all_model_words, all_extended_mw)
    signature_matrix = minHashing(binary_vectors)

    results = np.zeros((19, 4))

    for t in range(1, 20):
        candidates = LSH(signature_matrix, t / 20, len(data))
        predicted_duplicates = compute_duplicates(candidates)
        real_pairs = set()
        for k, vs in real_duplicates.items():
            for v in vs:
                real_pairs.add((min(int(k), int(v)), max(int(k), int(v))))

        predicted_pairs = set()
        for k, vs in predicted_duplicates.items():
            for v in vs:
                predicted_pairs.add((min(int(k), int(v)), max(int(k), int(v))))

        total_possible_pairs = len(data) * (len(data)-1) / 2
        frac_comparison = len(predicted_pairs) / total_possible_pairs

        pq = compute_pair_quality(real_pairs, predicted_pairs)
        pc = compute_pair_completeness(real_pairs, predicted_pairs)
        f1_star = compute_f1_star(real_pairs, predicted_pairs)

        results[(t-1), :] = [pc, pq, f1_star, frac_comparison]

    return results



# This code is used to test the LSH code
if __name__ == "__main__":
    a = 5
    # small_mw, extra_small_mw = extract_all_mw("data/TestData2.json")
    # save_mw(small_mw, extra_small_mw, "data/Test2_mw.json", "data/extra_Test2_mw.json")

    # small_mw, extra_small_mw = extract_all_mw_fast("data/TestData2.json")
    # small_mw, extra_small_mw = filter_freq_mw(small_mw, extra_small_mw)
    # save_freq_mw(small_mw, extra_small_mw, "data/Test2_mw.json", "data/extra_Test2_mw.json")
    #candidates = LSH("data/smallTestData.json", "data/smallmw.json", 0.45)
    #np.save("data/testLSH.npy", candidates)
    #print(candidates)
    #compute_duplicates(candidates)

    # with open("data4/testData_duplicates.json", "r") as file:
    #     real_duplicates = json.load(file)
    #
    # products_path = "data4/TestData.json"
    # model_words_path = "data4/test_mw.json"
    # extended_mw_path = "data4/test_ext_mw.json"
    # #
    # with open("data/small_real_dupl.json", "r") as file:
    #     real_duplicates = json.load(file)
    #
    # products_path = "data/smallTestData.json"
    # model_words_path = "data/smallmw.json"
    # extended_mw_path = "data/extra_smallmw.json"


    # with open("data/real_duplicates.json", "r") as file:
    #     real_duplicates = json.load(file)
    #
    # products_path = "data/bootstraps/bootstrap_2/bootSampleTrain_2.json"
    # model_words_path = "data/bootstraps/bootstrap_2/regular_mw-fast_2.json"
    # extended_mw_path = "data/bootstraps/bootstrap_2/extended_mw-fast_2.json"

    #
    # with open("data/real_Test2_dupl.json", "r") as file:
    #     real_duplicates = json.load(file)
    #
    # products_path = "data/TestData2.json"
    # model_words_path = "data/Test2_mw.json"
    # extended_mw_path = "data/extra_Test2_mw.json"
    #
    # with open(products_path, "r") as file:
    #     data = json.load(file)
    # with open(model_words_path, "r") as mw_file:
    #     all_model_words = json.load(mw_file)
    # with open(extended_mw_path, "r") as mw_file:
    #     all_extended_mw = json.load(mw_file)
    #
    # binary_vectors = create_bin_vec(data, all_model_words, all_extended_mw)
    # signature_matrix = minHashing(binary_vectors)
    #
    # for t in range(1, 20):
    #     candidates = LSH(signature_matrix, t / 20, len(data))
    #     predicted_duplicates = compute_duplicates(candidates)
    #     real_pairs = set()
    #     for k, vs in real_duplicates.items():
    #         for v in vs:
    #             real_pairs.add((min(int(k), int(v)), max(int(k), int(v))))
    #
    #     predicted_pairs = set()
    #     for k, vs in predicted_duplicates.items():
    #         for v in vs:
    #             predicted_pairs.add((min(int(k), int(v)), max(int(k), int(v))))
    #
    #     total_possible_pairs = len(data) * (len(data)-1) / 2
    #     frac_comparison = len(predicted_pairs) / total_possible_pairs
    #
    #     pq = compute_pair_quality(real_pairs, predicted_pairs)
    #     pc = compute_pair_completeness(real_pairs, predicted_pairs)
    #     f1_star = compute_f1_star(real_pairs, predicted_pairs)
    #     print(f"For t = {t/20}, fraction of comparisons = {frac_comparison:.4f}")
    #     print(f"And PC = {pc:.4f}, PQ  = {pq:.4f}, F1* = {f1_star:.4f}")
    #
