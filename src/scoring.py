def compute_sp_score(msa):
    """
    Calculate the Sum-of-Pairs (SP) score for a Multiple Sequence Alignment (MSA).

    Args:
    - msa: A list of lists representing the MSA. Each inner list corresponds to a sequence,
           and the integers represent residues or characters.

    Returns:
    - sp_score: The SP score of the MSA.
    """

    num_sequences = len(msa)
    sequence_length = len(msa[0])

    # Initialize the SP score
    sp_score = 0

    # Loop through all pairs of sequences
    for i in range(num_sequences):
        for j in range(i + 1, num_sequences):
            for k in range(sequence_length):
                residue_i = msa[i][k]
                residue_j = msa[j][k]

                # Check if both residues are non-gap (-1) and not the same (0)
                if residue_i != -1 and residue_j != -1 and residue_i != residue_j:
                    sp_score += 1

    # Normalize the SP score by the total number of residue pairs
    total_pairs = num_sequences * (num_sequences - 1) * sequence_length
    sp_score /= total_pairs

    return sp_score

# Example usage
if __name__ == "__main__":
    sample_msa = [
        [7, 20, 1, 7, 3],
        [7, 20, 1, 0, 0],
        [20, 1, 7, 7, 0],
        [7, 3, 0, 0, 0],
        [7, 20, 1, 7, 3]
    ]

    # Calculate the SP score
    sp_score = compute_sp_score(sample_msa)
    print("SP Score:", sp_score)
