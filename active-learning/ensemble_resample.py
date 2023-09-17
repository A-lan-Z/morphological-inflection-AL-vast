import os
import torch.nn.functional as F
import torch
import csv
import re
import sys


def read_tsv_files(directory, i):
    files_data = []
    pattern = re.compile(rf".*test_pool_{i}_\d+\.tsv$")
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

def calculate_entropy(probs):
    return -sum(p * torch.log(p) for p in probs if p > 0)


def resample_and_calculate_entropy(files_data):
    results = []
    entropies = []

    for i, row in enumerate(files_data[0]):
        sequence_probs = {}
        for file_data in files_data:
            # print(f"Processing row {i} of {len(files_data[0])}. Current file_data length: {len(file_data)}")
            if i >= len(file_data):
                print(f"Error: Index {i} is out of range for current file_data with length {len(file_data)}. Skipping this file.")
                continue
            _, all_predictions, all_probs = file_data[i]
            predictions = all_predictions.split('|')
            probs = [float(p) for p in all_probs.split('|')]

            for pred, prob in zip(predictions, probs):
                if pred not in sequence_probs:
                    sequence_probs[pred] = []
                sequence_probs[pred].append(prob)

        for pred, probs in sequence_probs.items():
            # Normalize the accumulated probability by the length of the prediction
            denominator = len(files_data) * len(pred.split())
            if denominator == 0:
                print(f"Warning: Denominator is zero for pred: {pred}")
                sequence_probs[pred] = 0
            else:
                sequence_probs[pred] = sum(probs) / denominator

        best_prediction = max(sequence_probs, key=sequence_probs.get)
        best_prob = sequence_probs[best_prediction]
        target = row[0]

        entropy = calculate_entropy(torch.tensor(list(sequence_probs.values())))
        results.append((best_prediction, target, best_prob, entropy.item()))
        entropies.append((i, entropy.item()))

    return results, sorted(entropies, key=lambda x: x[1], reverse=True)[:250]


def main(directory, i, train_file, test_file):
    files_data = read_tsv_files(directory, i)
    results, uncertain_samples_indices = resample_and_calculate_entropy(files_data)

    # Write the results to a new TSV file
    with open(os.path.join(directory, f'ensemble_results_pool_{i}.tsv'), 'w', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['prediction', 'target', 'average probability', 'entropy'])
        for result in results:
            writer.writerow(result)

    # Read the actual test data
    with open(test_file, 'r', encoding="utf-8") as f:
        test_data = [line.strip() for line in f.readlines()]
        print(f"Total number of test samples: {len(test_data)}")

    # Append the uncertain samples to the training file
    with open(train_file, 'a', encoding="utf-8") as f:
        for idx, _ in uncertain_samples_indices:
            f.write(test_data[idx] + '\n')
        print(f"Added {len(uncertain_samples_indices)} uncertain samples to the training file.")

    # Rewrite the test file without the selected uncertain samples
    with open(test_file, 'w', encoding="utf-8") as f:
        for idx, line in enumerate(test_data):
            if idx not in [sample[0] for sample in uncertain_samples_indices]:
                f.write(line + '\n')


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script_name.py <learning_step> <train_file> <test_file>")
        sys.exit(1)

    i = sys.argv[1]
    train_file = sys.argv[2]
    test_file = sys.argv[3]
    directory = "checkpoints/sig22/transformer"
    main(directory, i, train_file, test_file)
