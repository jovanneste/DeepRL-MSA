import matplotlib.pyplot as plt
import pickle 
import numpy as np

def plot_percentiles(x):
#    plt.hist(model_percentiles, bins=100, alpha=0.5, density=True, edgecolor='white')
#    plt.xlim(0, 100)
#    plt.xlabel("Chosen action ranking percentile")
#    plt.ylabel("% datasets")
#    plt.show()

    plt.figure(figsize=(14,7)) # Make it 14x7 inch
    plt.style.use('seaborn-whitegrid') # nice and clean grid
    plt.hist(x, bins=100, facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5)
    plt.xlim(1,100)
    plt.xlabel('Bins') 
    plt.ylabel('Values') 
    plt.show()

    
def percentage_greater_than(lst, x):
    count_greater_than_x = sum(1 for item in lst if item < x)
    percentage = (count_greater_than_x / len(lst)) * 100
    
    return percentage
    

with open('newmodel/6x6percentiles.pkl', 'rb') as file:
    scores = pickle.load(file)



plot_percentiles(scores)