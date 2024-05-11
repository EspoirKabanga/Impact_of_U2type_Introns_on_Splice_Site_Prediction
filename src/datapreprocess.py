from collections import OrderedDict

def one_hot_encode(sequences):
    """
    One-hot encodes DNA sequences.

    Args:
    - sequences (list): List of DNA sequences.

    Returns:
    - list: One-hot encoded sequences.
    """
    # Define a mapping from nucleotide to one-hot encoding
    mapping = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'N': [0, 0, 0, 0]
    }

    return [[mapping[nucleotide] for nucleotide in seq] for seq in sequences]


def read_file(f1, f2):
    """
    Reads sequences from two files and returns one-hot encoded versions.

    Args:
    - f1 (str): File path for positive sequences.
    - f2 (str): File path for negative sequences.

    Returns:
    - tuple: One-hot encoded positive and negative sequences.
    """
    # Read sequences from the files
    lines_pos = open(f1).readlines()
    lines_neg = open(f2).readlines()

    # Remove duplicates sequences
    lines_pos = list(OrderedDict.fromkeys(lines_pos))
    lines_neg = list(OrderedDict.fromkeys(lines_neg))

    # Remove the '\n' from each sequence
    seqs_pos = list(i.strip() for i in lines_pos)
    seqs_neg = list(i.strip() for i in lines_neg)

    return one_hot_encode(seqs_pos), one_hot_encode(seqs_neg)



