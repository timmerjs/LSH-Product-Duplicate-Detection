import json
import re
import random
import os
from sklearn.utils import resample

from extract_model_words import add_all_model_words


# Function to convert fractional inches to decimal
def convert_fractional_inches(text):
    def fraction_to_decimal(match):
        whole, fraction = match.groups()
        num, denom = map(int, fraction.split('/'))
        decimal = round(int(whole) + num / denom, 1)
        return f"{decimal}inch"

    # Match patterns like "11-1/8inch", "28-7/8inch", etc.
    return re.sub(r'(\d+)-(\d+/\d+)inch', fraction_to_decimal, text)

# Function to normalize units in the dataset
def normalize_units(text):
    # Normalize "inch"
    text = re.sub(r'(?<=\d)\s*(-)?\s*(?:inches|inch|Inches|Inch|\"|-inch|-inches| inch| Inches| Inch)', 'inch', text)
    text = convert_fractional_inches(text)  # Handle fractional inches
    # Normalize "hz"
    text = re.sub(r'(?<=\d)\s*(-)?\s*(?:Hz|hz|HZ|Hertz|hertz| hz| Hz|-hz|-Hz)', 'hz', text)
    # Remove space before "lbs"
    text = re.sub(r'(?<=\d)\s+lbs', 'lbs', text)
    text = re.sub(r'(?<=\d)\s*(?:pounds| lbs| pounds)', 'lbs', text)
    text = text.lower()
    return text

# Function to create bootstrap samples
def create_bootstraps(base_directory, n_bootstraps):
    """
    Create bootstrap samples by drawing with replacement, removing duplicates, and splitting into train/test sets.

    Parameters:
        file_path (str): Path to the input JSON file.
        n_bootstraps (int): Number of bootstrap samples to generate.
    """
    # Set seed for reproducibility
    SEED = 123

    # Load the data
    with open(f"{base_directory}/TVs-all-merged.json", 'r') as file:
        original_data = json.load(file)

    # Normalize features and titles and add unique product IDs for identification
    unique_id = 1
    for key, products in original_data.items():
        for product in products:
            # Normalize the featuresMap
            product['featuresMap'] = {normalize_units(k): normalize_units(v) if isinstance(v, str) else v
                                      for k, v in product['featuresMap'].items()}
            # Normalize the title
            product['title'] = normalize_units(product['title'])
            product['uniqueProductID'] = unique_id
            unique_id += 1

    # Flatten the dataset to a list of products
    product_list = [product for products in original_data.values() for product in products]

    # Save the original data as gold annotation
    with open(f"{base_directory}/data_identify.json", 'w') as file:
        json.dump(product_list, file, indent=2, sort_keys=False)

    # Extract all uniqueProductIDs from the dataset
    all_unique_ids = {product["uniqueProductID"] for product in product_list}

    n_samples = len(product_list)

    for i in range(n_bootstraps):
        # Create a folder for this bootstrap sample
        bootstrap_folder = f"{base_directory}/bootstraps/bootstrap_{i + 1}"
        os.makedirs(bootstrap_folder, exist_ok=True)

        # Generate the bootstrap sample with replacement
        sampled_products = resample(product_list, n_samples=n_samples, replace=True, random_state=SEED + i)

        # Remove duplicates based on uniqueProductID to get the training set
        unique_train_ids = set()
        train_sample = []
        for product in sampled_products:
            if product["uniqueProductID"] not in unique_train_ids:
                unique_train_ids.add(product["uniqueProductID"])
                train_sample.append(product)

        # Remove modelID's since they cannot be used any time
        for train_product in train_sample:
            if 'modelID' in train_product:
                del train_product['modelID']

        # Compute the test_set as all products not in the train_set
        train_ids = {product["uniqueProductID"] for product in train_sample}
        test_sample = [product for product in product_list if product["uniqueProductID"] not in train_ids]

        # Save the training set
        with open(os.path.join(bootstrap_folder, f"TrainSample_{i + 1}.json"), 'w') as train_file:
            json.dump(train_sample, train_file, indent=2, sort_keys=False)

        # Save the test set
        with open(os.path.join(bootstrap_folder, f"TestSample_{i + 1}.json"), 'w') as test_file:
            json.dump(test_sample, test_file, indent=2, sort_keys=False)

