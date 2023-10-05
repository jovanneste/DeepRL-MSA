import random
import numpy as np

length = 5
n_sequences = 3
mutation_prob = 0.2
gap_prob = 0.2
sequences = []

aa = ['M', 'A', 'T', 'G', 'Y']

initial_sequence = [random.choice(aa) for _ in range(length)]

sequences.append(initial_sequence)

for i in range(n_sequences-1):
    seq_mut = [i if random.random() > mutation_prob else random.choice(aa) for i in initial_sequence]
    sequences.append([i if random.random() > gap_prob else '_' for i in seq_mut])

print(np.asarray(sequences).reshape(n_sequences, length))