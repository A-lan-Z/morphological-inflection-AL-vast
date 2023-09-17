import os
import re
import csv

def extract_info_from_log(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    results = {'gold': [], 'pool': []}
    decode_mode = None
    beam_count = 0

    for line in lines:
        if "command line argument: decode" in line:
            if "Decode.beam" in line:
                decode_mode = 'beam'
        elif decode_mode and "TEST" in line:
            match = re.search(r"acc (\d+\.\d+) dist (\d+\.\d+)", line)
            if match:
                acc, dist = match.groups()
                if beam_count % 2 == 0:
                    results['gold'].append((float(acc), float(dist)))
                elif beam_count % 2 == 1:
                    results['pool'].append((float(acc), float(dist)))
                beam_count += 1
                decode_mode = ''

    return results

def al_process_logs_in_directory(directory, output_file):
    log_files = [f for f in os.listdir(directory) if f.endswith('.log')]

    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Training Size', 'Dataset', 'Gold Accuracy', 'Gold Edit Distance', 'Pool Accuracy', 'Pool Edit Distance'])

        datasets = {}

        for log_file in log_files:
            dataset_name = log_file.split('.')[0]
            if dataset_name not in datasets:
                datasets[dataset_name] = {'gold': [], 'pool': []}

            results = extract_info_from_log(os.path.join(directory, log_file))

            datasets[dataset_name]['gold'].extend(results.get('gold', []))
            datasets[dataset_name]['pool'].extend(results.get('pool', []))

        for dataset_name, results in datasets.items():
            print(f"Dataset: {dataset_name}")
            print("Training Size            Gold               |           Pool              ")
            print("               Accuracy | Avg Edit Distance | Accuracy | Avg Edit Distance")
            print("           ------------------------------------------------------------")

            gold_values = results.get('gold', [])
            pool_values = results.get('pool', [])
            training_size = 1000

            # Identify the row with the highest gold accuracy
            max_gold_acc_index = gold_values.index(max(gold_values, key=lambda x: x[0])) if gold_values else None

            for i in range(max(len(gold_values), len(pool_values))):
                gold_acc = f"{gold_values[i][0]:<8}" if i < len(gold_values) else '      '
                gold_dist = f"{gold_values[i][1]:<17}" if i < len(gold_values) else '                 '
                pool_acc = f"{pool_values[i][0]:<8}" if i < len(pool_values) else '      '
                pool_dist = f"{pool_values[i][1]:<17}" if i < len(pool_values) else '                 '
                print(f"{training_size:<14} {gold_acc} | {gold_dist} | {pool_acc} | {pool_dist:}")
                csvwriter.writerow([training_size, dataset_name, gold_acc.strip(), gold_dist.strip(), pool_acc.strip(), pool_dist.strip()])
                training_size += 250

            # Print the row with the highest gold accuracy at the bottom
            gold_acc = f"{gold_values[max_gold_acc_index][0]:<8}"
            gold_dist = f"{gold_values[max_gold_acc_index][1]:<17}"
            pool_acc = f"{pool_values[max_gold_acc_index][0]:<8}" if max_gold_acc_index < len(pool_values) else '      '
            pool_dist = f"{pool_values[max_gold_acc_index][1]:<17}" if max_gold_acc_index < len(pool_values) else '                 '
            print("           ------------------------------------------------------------")
            print(f"Best         | {gold_acc} | {gold_dist} | {pool_acc} | {pool_dist:}")
            csvwriter.writerow(["best_" + dataset_name, gold_acc.strip(), gold_dist.strip(), pool_acc.strip(), pool_dist.strip()])
            print("\n")

if __name__ == "__main__":
    # directory = input("Enter the directory path containing the log files: ")
    # output_file = input("Enter the name for the output CSV file: ")
    directory = "../experiments/experiment_1/kor_1st/2594"       # "../experiments/experiment_0"
    output_file = "../experiments/experiment_1/kor/2594/experiment_1.csv"
    al_process_logs_in_directory(directory, output_file)
