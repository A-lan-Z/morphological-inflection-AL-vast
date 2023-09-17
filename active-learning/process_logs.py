import os
import re
import csv

def extract_info_from_log(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:  # specify encoding here
        lines = f.readlines()

    decode_mode = None
    results = {'greedy': [], 'beam': []}

    for line in lines:
        if "command line argument: decode" in line:
            if "Decode.greedy" in line:
                decode_mode = 'greedy'
            elif "Decode.beam" in line:
                decode_mode = 'beam'
        elif decode_mode and "TEST" in line:
            match = re.search(r"acc (\d+\.\d+) dist (\d+\.\d+)", line)
            if match:
                acc, dist = match.groups()
                results[decode_mode].append((float(acc), float(dist)))

    return results

def process_logs_in_directory(directory, output_file):
    log_files = [f for f in os.listdir(directory) if f.endswith('.log')]

    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Dataset', 'Greedy Accuracy', 'Greedy Edit Distance', 'Beam Accuracy', 'Beam Edit Distance'])

        datasets = {}  # To store results by dataset name

        for log_file in log_files:
            dataset_name = log_file.split('.')[0]
            if dataset_name not in datasets:
                datasets[dataset_name] = {'greedy': [], 'beam': []}

            results = extract_info_from_log(os.path.join(directory, log_file))

            datasets[dataset_name]['greedy'].extend(results.get('greedy', []))
            datasets[dataset_name]['beam'].extend(results.get('beam', []))

        for dataset_name, results in datasets.items():
            print(f"Dataset: {dataset_name}")
            print("                     Greedy             |           Beam             ")
            print("           Accuracy | Avg Edit Distance | Accuracy | Avg Edit Distance")
            print("           ------------------------------------------------------------")

            greedy_values = results.get('greedy', [])
            beam_values = results.get('beam', [])

            for i in range(max(len(greedy_values), len(beam_values))):
                greedy_acc = f"{greedy_values[i][0]:<8}" if i < len(greedy_values) else '      '
                greedy_dist = f"{greedy_values[i][1]:<17}" if i < len(greedy_values) else '                 '
                beam_acc = f"{beam_values[i][0]:<8}" if i < len(beam_values) else '      '
                beam_dist = f"{beam_values[i][1]:<17}" if i < len(beam_values) else '                 '

                print(f"           {greedy_acc} | {greedy_dist} | {beam_acc} | {beam_dist:}")
                csvwriter.writerow([dataset_name, greedy_acc.strip(), greedy_dist.strip(), beam_acc.strip(), beam_dist.strip()])

            # Calculate and display averages
            avg_greedy_acc = sum([v[0] for v in greedy_values]) / len(greedy_values) if greedy_values else 0
            avg_greedy_dist = sum([v[1] for v in greedy_values]) / len(greedy_values) if greedy_values else 0
            avg_beam_acc = sum([v[0] for v in beam_values]) / len(beam_values) if beam_values else 0
            avg_beam_dist = sum([v[1] for v in beam_values]) / len(beam_values) if beam_values else 0

            print("           ------------------------------------------------------------")
            print(f"Average  | {avg_greedy_acc:<8.4f} | {avg_greedy_dist:<17.4f} | {avg_beam_acc:<8.4f} | {avg_beam_dist:<17.4f}")
            print("\n")

            # Write the average values to the CSV
            csvwriter.writerow(["avg_" + dataset_name, avg_greedy_acc, avg_greedy_dist, avg_beam_acc, avg_beam_dist])


if __name__ == "__main__":
    # directory = input("Enter the directory path containing the log files: ")
    # output_file = input("Enter the name for the output CSV file: ")
    directory = "../checkpoints/sig22/transformer"       # "../experiments/experiment_0"
    output_file = "../experiments/experiment_1/experiment_1.csv"
    process_logs_in_directory(directory, output_file)
