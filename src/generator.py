import random
import numpy as np


def char_to_int(sequence):
    char_to_number = {chr(65 + i): i + 1 for i in range(26) if chr(65 + i) != '_'}
    return np.array([[char_to_number.get(c, '_') for c in row] for row in sequence])

def generate_sequence(n_sequences, length, mutation_prob=0.2, del_prob=0.2):
    sequences = []
    aa = ['M', 'A', 'T', 'G', 'Y']

    initial_sequence = [random.choice(aa) for _ in range(length)]
    sequences.append(initial_sequence)

    for i in range(n_sequences-1):
        seq_mut = [i if random.random() > mutation_prob else random.choice(aa) for i in initial_sequence]
        seq_del = [i for i in seq_mut if random.random() > del_prob]
        if len(seq_del)==0:
            continue
        else:
            sequences.append(seq_del)

    padded_array = np.array([row + ['_'] * (length - len(row)) for row in np.asarray(sequences, dtype=object)])
    return char_to_int(padded_array)

