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
    difficulty_file = os.path.join(directory, f"{language}_difficulty_{iteration}.tsv")
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

def process_files(directory):
    results = {}
    dist_results = {}
    iterations = get_iterations(directory)
    for iteration in iterations:
        ensemble_df, difficulty_df = read_files(directory, iteration, language)
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
    return results, dist_results


if __name__ == '__main__':
    # Take language as user input
    language = input("Please enter the language code (e.g. khk, kor, eng): ").strip()

    if not language:
        print("Language code cannot be empty.")
        exit(1)

    directory_path = f'../experiments/ensemble_entropy/{language}'
    results, dist_results = process_files(directory_path)

    save_path = os.path.join(f'../experiments/ensemble_entropy', f'{language}_accuracies.tsv')
    save_results_to_tsv(results, save_path)
    print(f"Results saved to {save_path}")

    dist_save_path = os.path.join(f'../experiments/ensemble_entropy', f'{language}_dists.tsv')
    save_dists_to_tsv(dist_results, dist_save_path)
    print(f"Distances saved to {dist_save_path}")
