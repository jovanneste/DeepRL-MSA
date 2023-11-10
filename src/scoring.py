import numpy as np
import math

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

    return 1-sp_score




def calculate_entropy(msa_array):
    num_rows, num_columns = msa_array.shape
    
    entropies = []
    
    for col in range(num_columns):
        column_data = msa_array[:, col]
        residue_counts = dict()
        
        for residue in column_data:
            if residue in residue_counts:
                residue_counts[residue] += 1
            else:
                residue_counts[residue] = 1
        
        total_residues = num_rows
        
        entropy = 0
        for count in residue_counts.values():
            probability = count / total_residues
            entropy -= probability * math.log2(probability)
        
        entropies.append(entropy)
    
    return sum(entropies)/len(entropies)
