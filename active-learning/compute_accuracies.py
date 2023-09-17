import os
import csv
import sys

def get_correct_indices_and_dists(test_file):
    correct_indices = []
    dist_values = {}
    with open(test_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # skip header
        for idx, row in enumerate(reader, 1):
            dist = float(row[3])  # dist column
            dist_values[idx] = dist
            if dist == 0:
                correct_indices.append(idx)
    return correct_indices, dist_values

def compute_accuracies_and_dists(difficulty_file, correct_indices, dist_values):
    difficulties_correct = {i: 0 for i in range(1, 5)}
    difficulties_total = {i: 0 for i in range(1, 5)}
    difficulties_dist_sum = {i: 0.0 for i in range(1, 5)}  # Sum of dist values for each difficulty

    with open(difficulty_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # skip header
        for idx, row in enumerate(reader, 1):
            difficulty = int(row[1])
            difficulties_total[difficulty] += 1
            difficulties_dist_sum[difficulty] += dist_values.get(idx, 0)  # Add dist value to the sum
            if idx in correct_indices:
                difficulties_correct[difficulty] += 1

    overall_accuracy = len(correct_indices) / sum(difficulties_total.values())
    difficulty_accuracies = {i: difficulties_correct[i] / difficulties_total[i] for i in range(1, 5)}

    overall_dist = sum(dist_values.values()) / len(dist_values)
    difficulty_dists = {i: difficulties_dist_sum[i] / difficulties_total[i] for i in range(1, 5)}  # Average dist for each difficulty

    return overall_accuracy, difficulty_accuracies, overall_dist, difficulty_dists



def process_directory(directory, difficulty_file, output_file):
    results = []
    dist_results = []

    # Calculate total instances for each difficulty level once
    with open(difficulty_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        difficulties_total = {i: 0 for i in range(1, 5)}
        for row in reader:
            difficulty = int(row[1])
            difficulties_total[difficulty] += 1

    for filename in os.listdir(directory):
        if filename.endswith(".tsv") and '.decode.test_' in filename:
            test_name = filename.split('.decode.')[1].split('.tsv')[0]
            correct_indices, dist_values = get_correct_indices_and_dists(os.path.join(directory, filename))
            overall_accuracy, difficulty_accuracies, overall_dist, difficulty_dists = compute_accuracies_and_dists(
                difficulty_file, correct_indices, dist_values)
            results.append([test_name, overall_accuracy] + list(difficulty_accuracies.values()))
            dist_results.append([test_name, overall_dist] + list(difficulty_dists.values()))

        # Calculate averages for test results
    test_avg = [sum(row[i] for row in results) / len(results) for i in range(1, 6)]
    test_dist_avg = [sum(row[i] for row in dist_results) / len(dist_results) for i in range(1, 6)]

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['test result name', 'overall accuracy', 'both', 'lemma', 'feats', 'neither'])
        writer.writerows(results)
        writer.writerow(['total_instances'] + [''] + list(difficulties_total.values()))
        writer.writerow(['test_avg'] + test_avg)

    # Write dist results to kor_dist.tsv
    with open(output_file.replace('accuracies', 'dist'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['test result name', 'overall dist', 'both', 'lemma', 'feats', 'neither'])
        writer.writerows(dist_results)
        writer.writerow(['test_avg'] + test_dist_avg)

    print(f"Results written to {output_file}")
    print(f"Distances written to {output_file.replace('accuracies', 'dist')}")


if __name__ == "__main__":
    language = input("Please enter the language code (e.g. khk, kor, eng): ").strip()

    if not language:
        print("Language code cannot be empty.")
        sys.exit(1)

    directory_path = f"../experiments/experiment_0/{language}"
    difficulty_file_path = f"../dataset/{language}_difficulty.tsv"
    output_file_path = f"../experiments/experiment_0/{language}/{language}_accuracies.tsv"

    process_directory(directory_path, difficulty_file_path, output_file_path)

