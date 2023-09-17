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

def compute_accuracies(difficulty_file, correct_indices, dist_values):
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

    return overall_accuracy, difficulty_accuracies, difficulties_total, overall_dist, difficulty_dists

def process_directory(directory, accuracy_output_file, difficulty_output_file, dist_output_file):
    accuracy_results = []
    difficulty_results = []
    dist_results = []

    # Sort filenames based on the integer value after the underscore
    filenames = sorted(os.listdir(directory), key=lambda x: int(x.split('_')[-1].split('.tsv')[0]) if x.endswith(".tsv") and '.decode.test_' in x else 0)

    for filename in filenames:
        if filename.endswith(".tsv") and '.decode.test_' in filename:
            test_name = filename.split('.decode.')[1].split('.tsv')[0]
            test_num = test_name.split('_')[-1]
            difficulty_file_path = f"../experiments/experiment_{experiment}/{language}/{seed}/{seed}_{language}_difficulty_{test_num}.tsv"  # Define the difficulty_file_path here
            correct_indices, dist_values = get_correct_indices_and_dists(os.path.join(directory, filename))
            overall_accuracy, difficulty_accuracies, difficulties_total, overall_dist, difficulty_dists = compute_accuracies(
                difficulty_file_path, correct_indices, dist_values)
            accuracy_results.append([test_name, overall_accuracy] + list(difficulty_accuracies.values()))
            difficulty_results.append(['total_instances_' + test_name.split('_')[-1]] + list(difficulties_total.values()))
            dist_results.append([test_name, overall_dist] + list(difficulty_dists.values()))

    # Write accuracy results to {seed}_{language}_accuracies.tsv
    with open(accuracy_output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['test result name', 'overall accuracy', 'both', 'lemma', 'feats', 'neither'])
        writer.writerows(accuracy_results)

    # Write difficulty results to {seed}_{language}_difficulties.tsv
    with open(difficulty_output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['test result name', 'both', 'lemma', 'feats', 'neither'])
        writer.writerows(difficulty_results)

    # Write dist results to {seed}_{language}_dist.tsv
    with open(dist_output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['test result name', 'overall dist', 'both', 'lemma', 'feats', 'neither'])
        writer.writerows(dist_results)

    print(f"Accuracy results written to {accuracy_output_file}")
    print(f"Difficulty results written to {difficulty_output_file}")
    print(f"Dist results written to {dist_output_file}")


if __name__ == "__main__":
    languages = input("Please enter the language codes separated by commas (e.g. khk,kor_1st,eng): ").strip().split(',')
    seeds = input("Please enter the seeds separated by commas: ").strip().split(',')
    experiments = input("Please enter the experiment numbers separated by commas: ").strip().split(',')

    if not languages:
        print("Language codes cannot be empty.")
        sys.exit(1)

    for language in languages:
        for seed in seeds:
            for experiment in experiments:
                directory_path = f"../experiments/experiment_{experiment.strip()}/{language.strip()}/{seed.strip()}"
                accuracy_output_file_path = f"../experiments/experiment_{experiment.strip()}/{language.strip()}/{seed.strip()}_{language.strip()}_accuracies.tsv"
                difficulty_output_file_path = f"../experiments/experiment_{experiment.strip()}/{language.strip()}/{seed.strip()}_{language.strip()}_difficulties.tsv"
                dist_output_file_path = f"../experiments/experiment_{experiment.strip()}/{language.strip()}/{seed.strip()}_{language.strip()}_dist.tsv"

                process_directory(directory_path, accuracy_output_file_path, difficulty_output_file_path, dist_output_file_path)

