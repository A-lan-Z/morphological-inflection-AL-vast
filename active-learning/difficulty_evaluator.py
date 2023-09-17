import sys

def get_difficulty_level(train_lemmas, train_features, lemma, feature):
    lemma_present = lemma in train_lemmas
    feature_present = feature in train_features

    if lemma_present and feature_present:
        return 1
    elif lemma_present:
        return 2
    elif feature_present:
        return 3
    else:
        return 4

def main(train_file, test_file, output_file):
    # Extract lemmas and features from the training set
    train_lemmas = set()
    train_features = set()

    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            lemma, _, feature = line.strip().split('\t')
            train_lemmas.add(lemma)
            train_features.add(feature)

    # Evaluate the difficulty level for each instance in the test set
    difficulties = []

    with open(test_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            lemma, _, feature = line.strip().split('\t')
            level = get_difficulty_level(train_lemmas, train_features, lemma, feature)
            difficulties.append((idx, level))

    # Write the results to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("test instance index\tlevel of difficulty\n")
        for idx, level in difficulties:
            f.write(f"{idx}\t{level}\n")

    print(f"Results written to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python difficulty_evaluator.py <train_file_name> <test_file_name>")
        sys.exit(1)

    train_file_name = sys.argv[1]
    test_file_name = sys.argv[2]

    train_file_path = f"../2022InflectionST/part1/development_languages/{train_file_name}"
    test_file_path = f"../2022InflectionST/part1/development_languages/{test_file_name}"
    output_file_path = f"checkpoints/sig22/transformer/{test_file_name.split('.')[0]}_difficulty.tsv"

    main(train_file_path, test_file_path, output_file_path)

