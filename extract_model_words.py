import json
import os
import re

def extract_all_model_words(json_file):
    # Regular expressions for model words
    title_regex = re.compile(r"([a-zA-Z0-9]*([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*")
    key_value_regex = re.compile(r"(\d+(\.\d+)?[a-zA-Z]+$)|\d+(\.\d+)?$")

    # Dictionary to store counts of model words
    regular_mw_counts = {}
    extended_mw_counts = {}

    # Load JSON data
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Count model words from the data
    for product in data:  # datasets are lists of products
        # Extract from title
        title = product.get("title", "")
        title_mw = title_regex.findall(title)
        for modelword in title_mw:
            word = modelword[0]
            if word:
                regular_mw_counts[word] = regular_mw_counts.get(word, 0) + 1

        # Extract from key-value pairs
        features_map = product.get("featuresMap", {})
        for key, value in features_map.items():
            value = str(value)  # Ensure the value is a string
            #title_mw = title_regex.findall(value)
            key_value_matches = key_value_regex.findall(value)

            # Add all regular model words to the first set of model words (like was done for the title)
            # for regular_mw in title_mw:
            #     word = regular_mw[0]
            #     if word:
            #         regular_mw_counts[word] = regular_mw_counts.get(word, 0) + 1
            # Add all model words following the extended definition to the second set of model words
            # This set is only used for the KVP, not for titles
            for extended_mw in key_value_matches:
                num_part = extended_mw[0]  # Extract the matched number
                if any(c.isalpha() for c in num_part):  # Remove non-numeric suffixes
                    num_part = re.sub(r"[a-zA-Z]+$", "", num_part)
                extended_mw_counts[num_part] = extended_mw_counts.get(num_part, 0) + 1

    return regular_mw_counts, extended_mw_counts


# Filter on model words that occur more than once
def filter_freq_mw(regular_mw, extended_mw):
    """
    Filters model words that occur more than once.
    """
    return {word for word, count in regular_mw.items() if count > 1}, {word for word, count in extended_mw.items() if count > 2}


# Save the regular and extended model words to a file
def save_freq_mw(regular_mw, extended_mw, output_file1, output_file2):
    """
    Saves the filtered frequent model words to a JSON file.
    """
    with open(output_file1, 'w') as file:
        json.dump(list(regular_mw), file, indent=2)
    with open(output_file2, 'w') as file:
        json.dump(list(extended_mw), file, indent=2)


def add_all_model_words(base_directory):
    """
    Processes all bootstrap directories to generate mw-fast files.
    """
    # Iterate over all bootstrap directories
    for folder_name in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder_name)
        if os.path.isdir(folder_path) and folder_name.startswith("bootstraps"):
            for i, subfolder in enumerate(os.listdir(folder_path)):
                subfolder_path = os.path.join(folder_path, subfolder)
                if os.path.isdir(subfolder_path) and subfolder.startswith(f"bootstrap_"):
                    for file_name in os.listdir(subfolder_path):
                        if file_name.startswith("TrainSample_") and file_name.endswith(".json"):
                            input_file = os.path.join(subfolder_path, file_name)
                            output_file1 = os.path.join(subfolder_path, f"regular_modelwords_{file_name.split('_')[-1]}")
                            output_file2 = os.path.join(subfolder_path, f"extended_modelwords_{file_name.split('_')[-1]}")

                            # Extract all model words and filter frequent ones
                            regular_mw, extended_mw = extract_all_model_words(input_file)
                            regular_mw, extended_mw = filter_freq_mw(regular_mw, extended_mw)

                            # Save frequent model words
                            save_freq_mw(regular_mw, extended_mw, output_file1, output_file2)


# Example usage
#base_directory = "data"
#bootstraps_add_mw_fast(base_directory)
