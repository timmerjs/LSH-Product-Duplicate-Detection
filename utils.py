import re
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def extract_all_brands(datapath, output_path):
    # Load dataset
    with open(datapath, 'r') as file:
        data = json.load(file)

    brand_counter = Counter()
    all_brands = set()
    # Iterate through products and count brand occurrences
    for key, products in data.items():
        for product in products:
            features = product.get("featuresMap", {})
            brand = features.get("Brand") or features.get("Brand Name")
            if brand:
                all_brands.add(brand.lower())
                brand_counter[brand.lower()] += 1

    # Save brand occurrences to a JSON file
    with open(output_path, "w") as file:
        json.dump(list(all_brands), file, indent=4)

    return output_path


def diffBrand(product1, product2, base_directory):
    # Load the brands dataset
    with open(f'{base_directory}/all_brands.json', 'r') as brands_file:
        all_brands = set(json.load(brands_file))

    brand1 = product1.get("featuresMap", {}).get("brand") or product1.get("featuresMap", {}).get("brand name")
    brand2 = product2.get("featuresMap", {}).get("brand") or product2.get("featuresMap", {}).get("brand name")

    if brand1 and brand2:
        return brand1 in all_brands and brand2 in all_brands and brand1 != brand2
    return False  # Return False if one or both brands are not found


def sameShop(product1, product2):
    shop1 = product1.get("shop")
    shop2 = product2.get("shop")
    if shop1 is not None and shop2 is not None and (shop1 in shop2 or shop2 in shop1):
        return True
    else:
        return False


def diffScreenSize(product1, product2):
    def extract_all_screen_sizes(features):
        screen_sizes = []
        for key, value in features.items():
            if "screen size" in key or "display size" in key:
                # First check the numbers in front of "inch"
                matches = re.findall(r"(\d+\.?\d*)inch", str(value), re.IGNORECASE)
                if len(matches) == 0:
                    # If no matches are found, consider all numbers without "inch"
                    matches = re.findall(r"(\d+\.?\d*)", str(value), re.IGNORECASE)
                screen_sizes.extend([float(match) for match in matches])
        return screen_sizes

    sizes1 = extract_all_screen_sizes(product1.get("featuresMap", {}))
    sizes2 = extract_all_screen_sizes(product2.get("featuresMap", {}))

    # If any of the found sizes of the two products is approximately the same, consider these products the same size:
    if sizes1 is not None and sizes2 is not None:
        for size1 in sizes1:
            for size2 in sizes2:
                if abs(size1 - size2) < 1:
                    return False  # Allow for a small error of 1 inch
        return True  # If for all found sizes, no two sizes are approximately the same, then return True
    else:
        return False # Return False if one or both screen sizes are not found


def extract_model_words_title(title):
    # Predefined frequent title words (hard-coded)
    freq_title_words = {"hdtv", "led", "lcd", "smart", "ledlcd", "hd"}

    model_words = set()
    title_regex = re.compile(r"([a-zA-Z0-9]*([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*")
    #title_regex = re.compile(r"[a-zA-Z0-9](?:[0-9]+[^0-9, ]+|[^0-9, ]+[0-9]+)[a-zA-Z0-9]")
    # Split the title into individual words
    words = title.split()

    # Process each word
    for word in words:
        # Check if the word matches the regex
        if title_regex.search(word):
            model_words.add(word)
        # Include predefined frequent words
        if word.lower() in freq_title_words:
            model_words.add(word)
    # Old code, but I hard-coded the extra words in this function instead
    #with open('data/freqTitleWords.json', 'r') as data_file:
    #    extra_mw = json.load(data_file)
    #title_words = title.split(" ")
    #model_words.update(set(extra_mw) & set(title_words))
    return model_words


def extract_extended_model_words(KVP):
    regular_mw = set()
    extended_mw = set()
    #key_value_regex = re.compile(r"^\d{2,}$|^\d+\.\d+[a-zA-Z]+$|^\d+[a-zA-Z]+$")
    key_value_regex = re.compile(r"(\d+(\.\d+)?[a-zA-Z]+$)|\d+(\.\d+)?$")
    for key, value in KVP.items():
        regular_mw.update(extract_model_words_title(value))
        extended_matches = key_value_regex.findall(value)

        for extended_match in extended_matches:
            if extended_match[0]:
                num_part = extended_match[0]  # Extract the matched number
                if any(c.isalpha() for c in num_part):  # Remove non-numeric suffixes
                    num_part = re.sub(r"[a-zA-Z]+$", "", num_part)
                extended_mw.add(num_part)

    return regular_mw, extended_mw


def find_and_save_duplicates(data_file_path, output_file_path):

    # Load the JSON data
    with open(data_file_path, 'r') as file:
        products = json.load(file)

    # Create a dictionary to track modelID to uniqueProductID mapping
    model_id_map = defaultdict(list)
    for product in products:
        model_id = product["modelID"]
        unique_id = product["uniqueProductID"]
        model_id_map[model_id].append(unique_id)

    # Create a dictionary for duplicates
    duplicates = {}
    for model_id, product_ids in model_id_map.items():
        if len(product_ids) > 1:  # Only consider model IDs with duplicates
            for product_id in product_ids:
                duplicates[product_id] = [dup_id for dup_id in product_ids if dup_id != product_id]

    # Save the duplicates dictionary to a JSON file
    Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file_path, 'w') as file:
        json.dump(duplicates, file, indent=4)


def calcSim(a, b, q):
    def generate_qgrams(s, q):
        padded = f"{'#' * (q - 1)}{s}{'#' * (q - 1)}"
        return {padded[i:i + q] for i in range(len(padded) - q + 1)}

    # Generate sets of q-grams for both strings
    qgrams_a = generate_qgrams(a.lower(), q)
    qgrams_b = generate_qgrams(b.lower(), q)

    # Compute the overlap and union of q-grams
    n1 = len(qgrams_a)
    n2 = len(qgrams_b)
    qGramDistance = n1 + n2 - len(qgrams_a.intersection(qgrams_b))

    # Calculate Q-Gram Similarity
    if (n1 + n2) == 0:
        return 0.0  # Avoid division by zero; no similarity
    return (n1 + n2 - qGramDistance) / (n1 + n2)  # Proper similarity between 0 and 1


def mw(mw1, mw2):
    set1 = set(mw1)
    set2 = set(mw2)
    intersection_size = len(set1.intersection(set2))
    union_size = len(set1.union(set2))

    if union_size == 0:
        return 0.0  # Avoid division by zero when both sets are empty

    return intersection_size / union_size


def compute_duplicates(candidate_matrix):
    duplicates = {}
    num_rows, num_cols = candidate_matrix.shape

    # Iterate through the matrix
    for col_idx in range(num_cols):
        # Find row indices where the value is 1
        row_indices = np.where(candidate_matrix[:, col_idx] == 1)[0]

        # Add 1 to all indices for 1-based indexing
        duplicates[col_idx + 1] = (row_indices + 1).tolist()

    return duplicates


