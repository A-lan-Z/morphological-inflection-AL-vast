import os
import pandas as pd


def get_iterations(directory):
    """Extract unique iteration numbers from file names in the directory."""
    iterations = set()
    for filename in os.listdir(directory):
        if filename.startswith("ensemble_results_"):
            # Extract iteration number from the filename
            try:
                iteration = int(filename.split('_')[2].split('.')[0])
                iterations.add(iteration)
            except ValueError:
                # Skip files that don't have a number after "ensemble_results_"
                pass
    return sorted(iterations)


def read_files(directory, iteration, language):
    """Read the ensemble_results and difficulty files for a given iteration."""
    ensemble_file = os.path.join(directory, f"ensemble_results_{iteration}.tsv")
    difficulty_file = os.path.join(directory, f"{language}_difficulty_25.tsv")
    ensemble_df = pd.read_csv(ensemble_file, sep='\t')
    difficulty_df = pd.read_csv(difficulty_file, sep='\t')
    return ensemble_df, difficulty_df



def calculate_overall_accuracy(ensemble_df):
    """Calculate overall accuracy from ensemble_results dataframe."""
    correct_predictions = (ensemble_df['prediction'] == ensemble_df['target']).sum()
    total_predictions = len(ensemble_df)
    return correct_predictions / total_predictions


def calculate_difficulty_accuracy(ensemble_df, difficulty_df):
    """Calculate accuracy for each difficulty level."""
    difficulty_names = {1: "Both", 2: "lemma", 3: "feature", 4: "neither"}
    accuracies = {}

    for difficulty, name in difficulty_names.items():
        # Filter rows for the current difficulty level
        indices = difficulty_df[difficulty_df['level of difficulty'] == difficulty]['test instance index'].values
        valid_indices = [idx for idx in indices - 1 if 0 <= idx < len(ensemble_df)]
        filtered_ensemble = ensemble_df.iloc[valid_indices]

        # Calculate accuracy for the current difficulty level
        correct_predictions = (filtered_ensemble['prediction'] == filtered_ensemble['target']).sum()
        total_predictions = len(filtered_ensemble)
        accuracies[name] = correct_predictions / total_predictions

    return accuracies


def save_results_to_tsv(results, save_path):
    """Save the results to a TSV file."""
    with open(save_path, 'w') as file:
        # Write the header
        file.write("test result name\toverall accuracy\tboth\tlemma\tfeats\tneither\n")
        for iteration, values in results.items():
            line = f"test_{iteration}\t{values['Overall']:.4f}\t{values['Both']:.4f}\t{values['lemma']:.4f}\t{values['feature']:.4f}\t{values['neither']:.4f}\n"
            file.write(line)


def calculate_overall_dist(ensemble_df):
    if 'dist' in ensemble_df.columns:
        return ensemble_df['dist'].mean()
    return None

def calculate_difficulty_dist(ensemble_df, difficulty_df):
    if 'dist' not in ensemble_df.columns:
        return {}
    difficulty_names = {1: "Both", 2: "lemma", 3: "feature", 4: "neither"}
    distances = {}
    for difficulty, name in difficulty_names.items():
        indices = difficulty_df[difficulty_df['level of difficulty'] == difficulty]['test instance index'].values
        valid_indices = [idx for idx in indices - 1 if 0 <= idx < len(ensemble_df)]
        filtered_ensemble = ensemble_df.iloc[valid_indices]
        distances[name] = filtered_ensemble['dist'].mean()
    return distances

def save_dists_to_tsv(dists, save_path):
    """Save the distance results to a TSV file."""
    with open(save_path, 'w') as file:
        file.write("test result name\toverall dist\tboth\tlemma\tfeats\tneither\n")
        for iteration, values in dists.items():
            line = f"test_{iteration}\t{values['Overall']:.4f}\t{values['Both']:.4f}\t{values['lemma']:.4f}\t{values['feature']:.4f}\t{values['neither']:.4f}\n"
            file.write(line)


def calculate_difficulty_counts(difficulty_df):
    difficulty_names = {1: "Both", 2: "lemma", 3: "feature", 4: "neither"}
    counts = {}
    for difficulty, name in difficulty_names.items():
        counts[name] = len(difficulty_df[difficulty_df['level of difficulty'] == difficulty])
    return counts


def save_counts_to_tsv(counts, save_path):
    """Save the difficulty counts to a TSV file."""
    with open(save_path, 'w') as file:
        file.write("test result name\tboth\tlemma\tfeats\tneither\n")
        for iteration, values in counts.items():
            line = f"test_{iteration}\t{values['Both']}\t{values['lemma']}\t{values['feature']}\t{values['neither']}\n"
            file.write(line)


def process_files(directory):
    results = {}
    dist_results = {}
    difficulty_counts = {}
    iterations = get_iterations(directory)

    for iteration in iterations:
        ensemble_df, difficulty_df = read_files(directory, iteration, language)

        # Difficulty counts calculation
        counts = calculate_difficulty_counts(difficulty_df)
        difficulty_counts[iteration] = counts

        overall_accuracy = calculate_overall_accuracy(ensemble_df)
        difficulty_accuracies = calculate_difficulty_accuracy(ensemble_df, difficulty_df)
        results[iteration] = {
            'Overall': overall_accuracy,
            'Both': difficulty_accuracies['Both'],
            'lemma': difficulty_accuracies['lemma'],
            'feature': difficulty_accuracies['feature'],
            'neither': difficulty_accuracies['neither']
        }

        overall_dist = calculate_overall_dist(ensemble_df)
        if overall_dist is not None:
            difficulty_dists = calculate_difficulty_dist(ensemble_df, difficulty_df)
            dist_results[iteration] = {
                'Overall': overall_dist,
                'Both': difficulty_dists['Both'],
                'lemma': difficulty_dists['lemma'],
                'feature': difficulty_dists['feature'],
                'neither': difficulty_dists['neither']
            }
    return results, dist_results, difficulty_counts


def main(language, criteria):
    directory_path = f'../experiments/ensemble_{criteria}/{language}'
    results, dist_results, difficulty_counts = process_files(directory_path)

    # Save accuracies
    save_path = os.path.join(f'../experiments/ensemble_{criteria}', f'{language}_accuracies_ensemble_{criteria}.tsv')
    save_results_to_tsv(results, save_path)
    print(f"Results saved to {save_path}")

    # Save distances
    dist_save_path = os.path.join(f'../experiments/ensemble_{criteria}', f'{language}_dists_ensemble_{criteria}.tsv')
    save_dists_to_tsv(dist_results, dist_save_path)
    print(f"Distances saved to {dist_save_path}")

    # Save difficulty counts
    difficulty_counts_save_path = os.path.join(f'../experiments/ensemble_{criteria}', f'{language}_difficulty_counts_ensemble_{criteria}.tsv')
    save_counts_to_tsv(difficulty_counts, difficulty_counts_save_path)
    print(f"Difficulty counts saved to {difficulty_counts_save_path}")


if __name__ == '__main__':
    # Take languages as user input
    languages = input("Please enter the language codes separated by commas (e.g. khk,kor,eng): ").strip().split(',')

    # Take criteria as user input
    criteria_list = input("Please enter the criteria separated by commas (e.g. entropy,edit_distance): ").strip().split(',')

    if not languages or '' in languages:
        print("Language codes cannot be empty.")
        exit(1)

    if not criteria_list or '' in criteria_list:
        print("Criteria cannot be empty.")
        exit(1)

    for language in languages:
        for criteria in criteria_list:
            print(f"Processing for language: {language} and criteria: {criteria}")
            main(language.strip(), criteria.strip())

