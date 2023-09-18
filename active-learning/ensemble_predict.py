import os
import torch.nn.functional as F
import torch
import csv
import re
import sys
import numpy as np


def read_tsv_files(directory, i):
    files_data = []
    pattern = re.compile(rf".*test_{i}_\d+\.tsv$")
    for filename in os.listdir(directory):
        if pattern.search(filename):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                file_data = []
                for row in reader:
                    # Skip the header row
                    if row[0] == "target":
                        print(f"Reading file: {filename}. Header: {row}")
                        continue
                    file_data.append(row)
                files_data.append(file_data)
            # Explicitly close the file
            f.close()
    return files_data


def log_probs_to_probs(log_probs):
    return F.softmax(torch.tensor(log_probs), dim=0).tolist()


def calculate_entropy(probs):
    return -sum(p * torch.log(p) for p in probs if p >= 0.05)


def calculate_average_probability(files_data, use_log_probs=False):
    results = []
    correct_predictions = 0
    total_predictions = len(files_data[0])

    for index, row in enumerate(files_data[0]):
        sequence_denorm_probs = {}
        sequence_probs = {}
        sequence_dists = {}
        for file_data in files_data:
            if index >= len(file_data):
                print(f"Error: Index {index} is out of range for current file_data with length {len(file_data)}. Skipping this file.")
                continue
            _, all_predictions, all_log_probs, all_probs, all_dists = file_data[index]
            predictions = all_predictions.split('|')
            dists = all_dists.split('|')
            denorm_probs = [float(p) for p in all_probs.split('|')]
            if use_log_probs:
                log_probs = all_log_probs.split('|')
                probs = log_probs_to_probs([float(lp) for lp in log_probs])
            else:
                probs = denorm_probs

            for pred, prob, dist, denorm_probs in zip(predictions, probs, dists, denorm_probs):
                if pred not in sequence_probs:
                    sequence_probs[pred] = []
                sequence_probs[pred].append(prob)
                if pred not in sequence_denorm_probs:
                    sequence_denorm_probs[pred] = []
                sequence_denorm_probs[pred].append(denorm_probs)
                sequence_dists[pred] = dist

        # No need to divide, order of probs doesn't change
        # for pred, probs in sequence_probs.items():
        #     denominator = len(files_data)
        #     if denominator == 0:
        #         print(f"Warning: Denominator is zero for pred: {pred}")
        #         sequence_probs[pred] = 0
        #     else:
        #         sequence_probs[pred] = sum(probs) / denominator

        best_prediction = max(sequence_probs, key=sequence_probs.get)
        best_prob = sequence_probs[best_prediction]
        target = row[0]

        # Retrieve the edit distance for the best prediction
        edit_dist = sequence_dists[best_prediction]

        if best_prediction == target:
            correct_predictions += 1

        # Calculate entropy using denormalized probability
        entropy = calculate_entropy(torch.tensor(list(sequence_denorm_probs.values())))
        results.append(
            (best_prediction, target, best_prob, entropy.item(), edit_dist))

    accuracy = correct_predictions / total_predictions
    return results, accuracy


def main(directory, i, use_log_probs=False):
    files_data = read_tsv_files(directory, i)
    print(f"Total number of files being processed: {len(files_data)}")
    results, accuracy = calculate_average_probability(files_data, use_log_probs)

    with open(os.path.join(directory, f'ensemble_results_{i}.tsv'), 'w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['prediction', 'target', 'average probability', 'entropy', 'dist'])
        for result in results:
            writer.writerow(result)

    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <learning_step> <use_log_probs>")
        sys.exit(1)

    i = sys.argv[1]
    use_log_probs = sys.argv[2].lower() == 'true'
    directory = "checkpoints/sig22/transformer"
    main(directory, i, use_log_probs)
