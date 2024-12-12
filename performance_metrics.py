def compute_f1_score(real_pairs, predicted_pairs):
    # True Positives
    true_positives = len(real_pairs & predicted_pairs)

    # False Positives
    false_positives = len(predicted_pairs - real_pairs)

    # False Negatives
    false_negatives = len(real_pairs - predicted_pairs)

    # Precision and Recall
    precision = true_positives / (true_positives + false_positives) if predicted_pairs else 0
    recall = true_positives / (true_positives + false_negatives) if real_pairs else 0

    # F1 Score
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def compute_pair_quality(real_pairs, candidate_pairs):
    # Correctly predicted candidate pairs
    correct_duplicates = len(real_pairs & candidate_pairs)

    # Total amount of candidate pairs predicted
    total_candidate_pairs = len(candidate_pairs)

    return correct_duplicates / total_candidate_pairs if total_candidate_pairs > 0 else 0


def compute_pair_completeness(real_pairs, candidate_pairs):
    # Correctly predicted duplicates
    correct_duplicates = len(real_pairs & candidate_pairs)

    # Total number of real duplicate pairs
    total_real_duplicates = len(real_pairs)

    return correct_duplicates / total_real_duplicates if total_real_duplicates > 0 else 0


def compute_f1_star(real_pairs, candidate_pairs):
    pair_quality = compute_pair_quality(real_pairs, candidate_pairs)
    pair_completeness = compute_pair_completeness(real_pairs, candidate_pairs)

    if pair_quality + pair_completeness == 0:
        return 0
    return 2 * (pair_quality * pair_completeness) / (pair_quality + pair_completeness)


