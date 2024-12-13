import json
import heapq
import time

import numpy as np

from utils import sameShop, diffBrand, diffScreenSize, calcSim, extract_extended_model_words, mw
import tmwm

gamma = 0.70  # DEFINE!
mu = 0.65  # DEFINE!
# epsilon = 0.5  # DEFINE!
q = 3  # size of q-grams for calcSim function
k = 20  # If both products have more than k features, don't calculate TMWM


def msmp_fast(file_path, candidate_pairs_matrix, base_directory):
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


def hClustering_fast(dist, epsilon, products):
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
                    # Use sorted tuples to ensure (a,b) and (b,a) are treated as the same pair
                    duplicate_pairs.add(tuple(sorted([
                        products[x]["uniqueProductID"],
                        products[y]["uniqueProductID"]
                    ])))

        del clusters[j]

        # Recalculate inter-cluster distances
        for k in list(clusters.keys()):
            if k != i:
                # Use a generator expression to find minimum distance efficiently
                new_dist = min((dist[x, y] for x in clusters[i] for y in clusters[k]), default=np.inf)

                if new_dist <= epsilon:
                    heapq.heappush(heap, (new_dist, i, k))


    return duplicate_pairs
