import json
import heapq
import time

import numpy as np

from utils import sameShop, diffBrand, diffScreenSize, calcSim, extract_extended_model_words, mw
import tmwm

gamma = 0.70  # DEFINE!
mu = 0.65  # DEFINE!
#epsilon = 0.5  # DEFINE!
q = 3  # size of q-grams for calcSim function
k = 20  # If both products have more than k features, don't calculate TMWM


def msmp_fast(file_path, candidate_pairs_matrix, base_directory):
    """
    MSMP-FAST Algorithm Implementation.

    Parameters:
        file_path (str): Path to the dataset file containing product information in JSON format.
        candidate_pairs_matrix (np.ndarray): n x n matrix indicating candidate pairs (1 for candidate, 0 otherwise).

    Returns:
        list: Clusters of similar products.
    """
    # Load dataset
    with open(file_path, 'r') as file:
        products = json.load(file)
    n = len(products)  # Total number of products
    dist = np.full((n, n), np.inf)  # Initialize distance matrix with infinity

    num_pairs = 0

    for i, pi in enumerate(products):
        for j, pj in enumerate(products):
            if i >= j:
                continue
            num_pairs += 1
            if candidate_pairs_matrix[i, j] == 0:  # Skip non-candidates
                dist[i, j] = np.inf
                continue

            if sameShop(pi, pj) or diffBrand(pi, pj, base_directory) or diffScreenSize(pi, pj):
                dist[i, j] = np.inf
            else:
                sim, avgSim, m, w = 0, 0, 0, 0
                nmki = pi["featuresMap"].copy()  # Non-matching key-value pairs for pi
                nmkj = pj["featuresMap"].copy()  # Non-matching key-value pairs for pj

                # Key-Value Pair Matching
                for q_key, q_value in pi["featuresMap"].items():
                    for r_key, r_value in pj["featuresMap"].items():
                        keySim = calcSim(q_key, r_key, q)
                        if keySim > gamma:
                            valueSim = calcSim(q_value, r_value, q)
                            weight = keySim
                            sim += weight * valueSim
                            m += 1
                            w += weight
                            del nmki[q_key]
                            del nmkj[r_key]
                if w > 0:
                    avgSim = sim / w

                # Compute Model Word Percentage
                pi_mw_regular, not_used = extract_extended_model_words(nmki)
                pj_mw_regular, not_used = extract_extended_model_words(nmkj)
                mwPerc = mw(pi_mw_regular, pj_mw_regular)

                n_features1 = len(pi["featuresMap"])
                n_features2 = len(pj["featuresMap"])

                # This is the added part for MSMP_FAST:
                # Only consider TMWM if we have a little amount of features (less than 20)
                if n_features1 > 20 and n_features2 > 20:
                    theta1 = m / min(n_features1, n_features2)
                    theta2 = 1 - theta1
                    hSim = theta1 * avgSim + theta2 * mwPerc
                else:
                    titleSim = tmwm.calculate_similarity(pi, pj)
                    if titleSim == -1:
                        theta1 = m / min(n_features1, n_features2)
                        theta2 = 1 - theta1
                        hSim = theta1 * avgSim + theta2 * mwPerc
                    else:
                        theta1 = (1 - mu) * m / min(n_features1, n_features2)
                        theta2 = 1 - mu - theta1
                        hSim = theta1 * avgSim + theta2 * mwPerc + mu * titleSim

                # Transform to dissimilarity
                dist[i, j] = 1 - hSim
                dist[j, i] = 1 - hSim

    # Perform hierarchical clustering
    return dist, products  # ε should be set as a constant in the imported module


def Alternative_hClustering_fast(dist, epsilon, products):
    """
    Perform hierarchical clustering with an emphasis on avoiding clusters with infinite pairwise distances.

    Returns a set of duplicate pairs based on distance matrix and epsilon threshold.
    """
    n = dist.shape[0]
    clusters = {i: [i] for i in range(n)}

    # Initialize duplicate pairs as a set to avoid duplicates and ensure uniqueness
    duplicate_pairs = set()

    # Create initial heap of valid pairwise distances
    heap = [(dist[i, j], i, j) for i in range(n) for j in range(i + 1, n) if dist[i, j] <= epsilon]
    heapq.heapify(heap)

    while heap:
        min_dist, i, j = heapq.heappop(heap)

        # Skip if clusters no longer exist or distance exceeds threshold
        if i not in clusters or j not in clusters:
            continue

        if min_dist > epsilon:
            break

        # Check for valid merge (no infinite pairwise distances)
        if any(dist[x, y] == np.inf for x in clusters[i] for y in clusters[j]):
            continue

        # Merge clusters
        clusters[i].extend(clusters[j])

        # Add all pairwise duplicate relationships within the merged cluster
        for x in clusters[i]:
            for y in clusters[i]:
                if x != y:
                    # Use frozenset to ensure (a,b) and (b,a) are treated as the same pair
                    duplicate_pairs.add(frozenset([
                        products[x]["uniqueProductID"],
                        products[y]["uniqueProductID"]
                    ]))

        del clusters[j]

        # Recalculate inter-cluster distances
        for k in list(clusters.keys()):
            if k != i:
                # Use a generator expression to find minimum distance efficiently
                new_dist = min((dist[x, y] for x in clusters[i] for y in clusters[k]), default=np.inf)

                if new_dist <= epsilon:
                    heapq.heappush(heap, (new_dist, i, k))

    return duplicate_pairs


def hClustering_fast(dist, epsilon, products):
    """
    Perform hierarchical clustering with an emphasis on avoiding clusters with infinite pairwise distances.

    Parameters:
        dist (numpy.ndarray): A 2D array where dist[i, j] is the distance between product i and product j.
        epsilon (float): The threshold for merging clusters.
        products (list): List of product dictionaries, each containing a "uniqueProductID" field.

    Returns:
        dict: A dictionary where each key is a uniqueProductID and the value is a list of duplicate uniqueProductIDs.
    """
    n = dist.shape[0]

    # Initialize clusters: each product starts in its own cluster
    clusters = {i: [i] for i in range(n)}

    # Initialize dictionary for duplicates
    duplicates = {product["uniqueProductID"]: [] for product in products}

    # Initialize a priority queue (heap) with all valid distances <= epsilon
    heap = []
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] <= epsilon:
                heapq.heappush(heap, (dist[i, j], i, j))

    while heap:
        # Pop the smallest distance from the heap
        min_dist, i, j = heapq.heappop(heap)

        # Check if clusters i and j still exist
        if i not in clusters or j not in clusters:
            continue

        # If the minimum distance is greater than epsilon, stop processing
        if min_dist > epsilon:
            break

        # Verify that merging clusters i and j is valid (no pairwise distance is np.inf)
        cluster_i = clusters[i]
        cluster_j = clusters[j]
        valid_merge = True
        for x in cluster_i:
            for y in cluster_j:
                if dist[x, y] == np.inf:
                    valid_merge = False
                    break
            if not valid_merge:
                break

        # Skip merging if any pairwise distance is np.inf
        if not valid_merge:
            continue

        # Merge cluster j into cluster i
        clusters[i].extend(cluster_j)

        # Update the duplicates dictionary
        for x in clusters[i]:
            for y in clusters[i]:
                if x != y:
                    unique_id_x = products[x]["uniqueProductID"]
                    unique_id_y = products[y]["uniqueProductID"]
                    if unique_id_y not in duplicates[unique_id_x]:
                        duplicates[unique_id_x].append(unique_id_y)

        # Delete cluster j
        del clusters[j]

        # Recalculate distances for the updated cluster i with all other clusters
        for k in list(clusters.keys()):
            if k != i:
                new_dist = np.inf
                for x in clusters[i]:
                    for y in clusters[k]:
                        new_dist = min(new_dist, dist[x, y])
                        if new_dist == np.inf:
                            break
                    if new_dist == np.inf:
                        break
                if new_dist <= epsilon:
                    heapq.heappush(heap, (new_dist, i, k))

    # Convert duplicates dictionary to a set of unique duplicate pairs
    duplicate_pairs = set()
    for unique_id, dups in duplicates.items():
        for dup in dups:
            # Use frozenset to ensure (a,b) and (b,a) are considered the same pair
            duplicate_pairs.add(frozenset([unique_id, dup]))

    return duplicate_pairs
