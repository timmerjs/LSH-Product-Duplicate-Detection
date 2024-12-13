import json
from collections import Counter
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt


def dataStats(base_directory):
    # Load the JSON data
    with open(f"{base_directory}/TVs-all-merged.json", 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Initialize containers
    model_ids = set()  # For distinct model IDs
    website_counts = Counter()  # For counting products per website
    missing_modelID_count = 0  # Count for modelIDs not in title
    feature_counts = []  # List to store the number of features per product
    total_products = 0

    # Process the JSON data for descriptive statistics
    for product_key, product_list in data.items():
        model_ids.add(product_key)
        for product in product_list:
            total_products += 1
            if product["modelID"] not in product.get("title", ""):
                missing_modelID_count += 1
            website_counts[product["shop"]] += 1
            feature_counts.append(len(product["featuresMap"]))

    # Count the number of distinct model IDs
    distinct_model_ids = len(model_ids)

    # Compute statistics for the feature attributes
    feature_counts_array = np.array(feature_counts)
    stats = {
        "mean": np.mean(feature_counts_array),
        "std_dev": np.std(feature_counts_array),
        "median": np.median(feature_counts_array),
        "min": np.min(feature_counts_array),
        "max": np.max(feature_counts_array),
    }

    all_modelIDs = set()
    n_duplicate_pairs = 0
    for key, values in data.items():
        if len(values) > 1:
            for duplicate_pair in combinations(values, 2):
                n_duplicate_pairs += 1
        for product in values:
            all_modelIDs.add(product["modelID"])

    # Print results
    print("\n")
    print("#################### DATA STATISTICS SUMMARY ####################\n")
    print(f"Total number of products in the dataset: {total_products}")
    print(f"Number of distinct modelIDs: {len(all_modelIDs)}")
    print(f"Number of duplicate pairs: {n_duplicate_pairs}")
    print(f"Number of products where modelID is not in the title: {missing_modelID_count}")

    print("\nNumber of products per website:")
    for website, count in website_counts.items():
        print(f"{website}: {count}")

    # Plot a histogram of feature counts
    plt.figure(figsize=(8, 5))
    plt.hist(feature_counts, bins=stats["max"], color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel("Number of Features", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    statfig_path = f"{base_directory}/n_features_distribution.png"
    plt.savefig(statfig_path)
    plt.close()

    print("\nSummary statistics for the number of listed attributes per product:")
    for stat_name, stat_value in stats.items():
        print(f"{stat_name.capitalize()}: {stat_value}")
    min_features = 20
    percentage_more_than_10 = (np.sum(feature_counts_array >= min_features) / len(feature_counts_array)) * 100
    print(f"\nPercentage of products with more than {min_features} attributes: {percentage_more_than_10:.2f}%")
    print(f"Note: the plot of these statistics can be found at \"{statfig_path}\"\n")



