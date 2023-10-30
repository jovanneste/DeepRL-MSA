import numpy as np

def compute_sp_score(msa):
    num_sequences = len(msa)
    sequence_length = len(msa[0])

    sp_score = 0

    for i in range(num_sequences):
        for j in range(i + 1, num_sequences):
            for k in range(sequence_length):
                residue_i = msa[i][k]
                residue_j = msa[j][k]

                # Check if both residues are non-gap (-1) and not the same (0)
                if residue_i != -1 and residue_j != -1 and residue_i != residue_j:
                    sp_score += 1

    total_pairs = num_sequences * (num_sequences - 1) * sequence_length
    sp_score /= total_pairs

    return sp_score




if __name__ == "__main__":
    sample_msa = np.asarray([
        [7, 20, 1, 7, 3],
        [7, 20, 1, 0, 0],
        [20, 1, 7, 7, 0],
        [7, 3, 0, 0, 0],
        [7, 20, 1, 7, 3]
    ])

    # Calculate the SP score
    sp_score = compute_sp_score(sample_msa)
    print("SP Score:", sp_score)