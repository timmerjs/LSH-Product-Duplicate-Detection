import json
import re
import math
from collections import Counter
import Levenshtein
import random
from sklearn.utils import resample

from utils import extract_model_words_title

# Adjustable parameters
alpha = 0.6  # Cosine similarity threshold
beta = 0.1   # Weight for average Levenshtein similarity
eta = 0.0  # 0.5 Threshold for first check if products can be classified as different
delta = 0.4  # 0.8 Threshold for updated model word similarity
epsilon = 0.3  # 0.0  Final similarity threshold

def preprocess_title(title):
    title = re.sub(r'[\&\/\-\_\,]', ' ', title)
    title = re.sub(r'\band\b|\bor\b|\bthe\b', ' ', title, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', title.strip())


def split_word(s: str):
    digits = []
    letters = []
    has_both = False

    for char in s:
        if char.isdigit():
            digits.append(char)
        elif char.isalpha():
            letters.append(char)
        # Ignore all other characters
    if len(letters) > 0 and len(digits) > 0:
        has_both = True

    # Convert the digits list to an integer if not empty
    int_value = int("".join(digits)) if digits else None
    str_value = "".join(letters)

    return int_value, str_value, has_both


def check_model_word_pair(model_words1, model_words2, eta):
    num_dissimilar_pairs = 0
    for word1 in model_words1:
        for word2 in model_words2:
            num1, non_num1, has_both1 = split_word(word1)
            num2, non_num2, has_both2 = split_word(word2)

            if has_both1 and has_both2:

                try:
                    num1 = int(num1)
                    num2 = int(num2)
                except ValueError:
                    continue  # Skip if conversion to integer fails

                # Ensure non-numeric parts contain at least 2 characters
                if len(non_num1) < 2 or len(non_num2) < 2:
                    continue

                # Calculate normalized Levenshtein similarity
                non_numeric_dist = Levenshtein.distance(non_num1, non_num2) / max(len(non_num1), len(non_num2))
                non_numeric_sim = 1 - non_numeric_dist

                # Check similarity and numeric mismatch
                if num1 != num2:
                    if non_numeric_sim >= eta:
                        num_dissimilar_pairs += 1  # Allow for one "typo", so only return False for two or more rejections
                        if num_dissimilar_pairs == 2:
                            return False  # Products are not duplicates  # Products are not duplicates

    return True  # No violations found


def calc_cosine_sim(title1, title2):
    vec1, vec2 = Counter(title1.split()), Counter(title2.split())
    intersection = len(vec1 & vec2)
    norm1 = math.sqrt(len(vec1))
    norm2 = math.sqrt(len(vec2))
    return intersection / (norm1 * norm2) if norm1 and norm2 else 0.0

def avg_lv_sim(words1, words2):
    num = 0.0
    den = 0.0
    for word1 in words1:
        for word2 in words2:
            lv = Levenshtein.distance(word1, word2) / max(len(word1), len(word2))
            num += (1-lv)*len(word1)*len(word2)
            den += len(word1)*len(word2)
    return num/den if den > 0.0 else 0.0


def avg_lv_sim_mw(words1, words2):
    num = 0.0
    den = 0.0
    for word1 in words1:
        for word2 in words2:
            num1, non_num1, has_both1 = split_word(word1)
            num2, non_num2, has_both2 = split_word(word2)

            if has_both1 and has_both2:
                non_numeric_sim = 1 - (Levenshtein.distance(non_num1, non_num2) / max(len(non_num1), len(non_num2)))

                if non_numeric_sim <= eta and num1 == num2:
                    lv = Levenshtein.distance(word1, word2) / max(len(word1), len(word2))
                    num += (1-lv)*len(word1)*len(word2)
                    den += len(word1)*len(word2)
    return num/den if den > 0.0 else 0.0

def calculate_similarity(product1, product2):
    title1, title2 = product1["title"], product2["title"]
    title1, title2 = preprocess_title(title1), preprocess_title(title2)
    model_words1, model_words2 = extract_model_words_title(title1), extract_model_words_title(title2)

    cosine_sim = calc_cosine_sim(title1, title2)
    if cosine_sim > alpha:
        return 1

    # Check for model word mismatch criteria
    if not check_model_word_pair(model_words1, model_words2, eta):
        return -1

    model_word_similarity = avg_lv_sim(model_words1, model_words2)
    final_similarity = (beta * cosine_sim + (1 - beta) * model_word_similarity)


    # Check for at least one valid pair of model words
    update_similarity = False

    for word1 in model_words1:
        for word2 in model_words2:
            num1, non_num1, has_both1 = split_word(word1)
            num2, non_num2, has_both2 = split_word(word2)

            if has_both1 and has_both2:
                non_numeric_dist = (Levenshtein.distance(non_num1, non_num2) / max(len(non_num1), len(non_num2)))
                non_numeric_sim = 1 - non_numeric_dist
                if non_numeric_sim >= eta and num1 == num2:
                    # Update similarity if a valid model word pair is found
                    update_similarity = True
    if update_similarity:
        model_word_similarity = avg_lv_sim_mw(model_words1, model_words2)
        final_similarity = delta * model_word_similarity + (1 - delta) * final_similarity

    if final_similarity > epsilon:
        return final_similarity
    return -1
