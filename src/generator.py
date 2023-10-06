import random
import numpy as np

def generate_sequence(n_sequences, length, mutation_prob=0.2, del_prob=0.2):
    sequences = []
    aa = ['M', 'A', 'T', 'G', 'Y']

    initial_sequence = [random.choice(aa) for _ in range(length)]
    sequences.append(initial_sequence)

    for i in range(n_sequences-1):
        seq_mut = [i if random.random() > mutation_prob else random.choice(aa) for i in initial_sequence]
        sequences.append([i for i in seq_mut if random.random() > del_prob])
    
    padded_array = np.array([row + ['_'] * (length - len(row)) for row in np.asarray(sequences)])
    return padded_array