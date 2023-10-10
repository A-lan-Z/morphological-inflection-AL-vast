import numpy as np

# 1. Data Representation
def to_vector(lemma, charset, features, feature_set):
    vector = [0] * (len(charset) + len(feature_set))
    for char in lemma:
        if char in charset:
            vector[charset.index(char)] = 1
    for feature in features:
        if feature in feature_set:
            vector[len(charset) + feature_set.index(feature)] = 1
    return vector

# 2. Compute Cosine Similarity
def cosine_similarity(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

# 3. Calculate Density Term
def compute_density(lemmas, charset, feature_set):
    print("Computing vectors for lemmas...")
    vectors = [to_vector(lemma[0], charset, lemma[1], feature_set) for lemma in lemmas]
    densities = []
    print("Calculating density for each lemma...")
    for i, vector in enumerate(vectors):
        similarities = [cosine_similarity(vector, other_vector) for other_vector in vectors]
        average_similarity = sum(similarities) / len(similarities)
        densities.append(average_similarity)
        if (i + 1) % 1 == 0:
            print(f"Processed {i + 1} lemmas out of {len(lemmas)}")
    return densities

# 4. Store in a File
def save_to_file(lemmas, densities, filename):
    print(f"Saving results to {filename}...")
    with open(filename, 'w', encoding='utf-8') as file:
        # Write headers
        file.write("index\tlemma\tinformation density\n")
        for index, (lemma, density) in enumerate(zip(lemmas, densities)):
            file.write(f"{index}\t{lemma}\t{density}\n")
    print("Results saved successfully!")

def main():
    input_file = 'dataset/ara_pool.train'
    output_file = 'ara_density.tsv'

    print(f"Reading data from {input_file}...")
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        dataset = [(line.strip().split('\t')[0], line.strip().split('\t')[2].split(';')) for line in lines]

    # Extract lemmas and create a character set
    lemmas = [data[0] for data in dataset]
    features = [data[1] for data in dataset]
    charset = sorted(set(''.join(lemmas)))
    feature_set = sorted(set(feature for feature_list in features for feature in feature_list))
    print(f"Found {len(lemmas)} unique lemmas and {len(feature_set)} unique features.")

    # Compute densities
    densities = compute_density(dataset, charset, feature_set)

    # Save to file
    save_to_file(lemmas, densities, output_file)


if __name__ == "__main__":
    main()

