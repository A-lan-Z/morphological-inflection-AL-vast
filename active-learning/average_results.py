import os
import csv
import sys


def average_files(files):
    # Initialize a dictionary to store the sum of values
    summed_data = {}
    num_files = len(files)

    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)
            for row in reader:
                test_name = row[0]
                values = [float(val) for val in row[1:]]
                if test_name not in summed_data:
                    summed_data[test_name] = values
                else:
                    summed_data[test_name] = [sum(x) for x in zip(summed_data[test_name], values)]

    # Calculate the average
    for key in summed_data:
        summed_data[key] = [x / num_files for x in summed_data[key]]

    return header, summed_data


def write_averaged_file(output_file, header, data):
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        for key, values in data.items():
            writer.writerow([key] + values)


def get_filename_suffix(experiment):
    if experiment == "1":
        return "_random.tsv"
    elif experiment == "3":
        return "_entropy.tsv"
    else:
        return ".tsv"

def main():
    languages = input("Please enter the language codes separated by commas (e.g. khk,kor_1st,eng): ").strip().split(',')
    experiments = input("Please enter the experiment numbers separated by commas: ").strip().split(',')

    for language in languages:
        for experiment in experiments:
            directory_path = f"../experiments/experiment_{experiment.strip()}/{language.strip()}"
            output_path = f"../experiments/experiment_{experiment.strip()}"

            # Get the filename suffix based on experiment
            suffix = get_filename_suffix(experiment.strip())

            # Collect all files for each type with the appropriate suffix
            accuracy_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if
                              f.endswith(f'_accuracies{suffix}')]
            difficulty_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if
                                f.endswith(f'_difficulties{suffix}')]
            dist_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if
                          f.endswith(f'_dist{suffix}')]

            # Average the files
            accuracy_header, averaged_accuracy_data = average_files(accuracy_files)
            difficulty_header, averaged_difficulty_data = average_files(difficulty_files)
            dist_header, averaged_dist_data = average_files(dist_files)

            # Write the averaged data to new files with the appropriate suffix
            write_averaged_file(os.path.join(output_path, f'{language}_average_accuracies{suffix}'), accuracy_header,
                                averaged_accuracy_data)
            write_averaged_file(os.path.join(output_path, f'{language}_average_difficulties{suffix}'), difficulty_header,
                                averaged_difficulty_data)
            write_averaged_file(os.path.join(output_path, f'{language}_average_dist{suffix}'), dist_header,
                                averaged_dist_data)

            print(f"Averaged files for {language} in experiment {experiment} have been written.")


if __name__ == "__main__":
    main()
