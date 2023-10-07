from cnn import *
from generator import *
from alignment-agent import *

s = generate_sequence(5,5)
print(s)
print(score(s))