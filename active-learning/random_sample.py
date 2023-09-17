import random


def select_random_instances(input_file, output_file, num_instances):
    """
    Randomly select a given number of instances from the input file, save to the output file, and remove them from the input file.

    Parameters:
    - input_file (str): Path to the input file.
    - output_file (str): Path to the output file.
    - num_instances (int): Number of instances to select.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Randomly select instances
    selected_lines = random.sample(lines, num_instances)

    # Remove selected instances from the original list
    for line in selected_lines:
        lines.remove(line)

    # Write selected instances to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(selected_lines)

    # Write the remaining instances back to the input file
    with open(input_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)

if __name__ == "__main__":
    input_file = input("Enter the path to the input file: ")
    output_file = input("Enter the path to the output file: ")
    num_instances = int(input("Enter the number of instances to select: "))

    select_random_instances(input_file, output_file, num_instances)
    print(f"Selected {num_instances} random instances, saved to {output_file}, and removed from {input_file}.")