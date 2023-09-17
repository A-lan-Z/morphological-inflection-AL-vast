import numpy as np

# 1. Data Representation
def lemma_to_vector(lemma, charset):
    vector = [0] * len(charset)
    for char in lemma:
        if char in charset:
            vector[charset.index(char)] = 1
    return vector

# 2. Compute Cosine Similarity
def cosine_similarity(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

# 3. Calculate Density Term
def compute_density(lemmas, charset):
    print("Computing vectors for lemmas...")
    vectors = [lemma_to_vector(lemma, charset) for lemma in lemmas]
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
        for lemma, density in zip(lemmas, densities):
            file.write(f"{lemma}\t{density}\n")
    print("Results saved successfully!")

def main():
    input_file = '../../2022InflectionST/part1/development_languages/kor_pool.train'
    output_file = 'kor_density.tsv'

    print(f"Reading data from {input_file}...")
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        dataset = [line.strip().split('\t') for line in lines]

    # Extract lemmas and create a character set
    lemmas = [data[0] for data in dataset]
    charset = sorted(set(''.join(lemmas)))
    print(f"Found {len(lemmas)} unique lemmas.")

    # Compute densities
    densities = compute_density(lemmas, charset)

    # Save to file
    save_to_file(lemmas, densities, output_file)

if __name__ == "__main__":
    main()
