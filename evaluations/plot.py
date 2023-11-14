import matplotlib.pyplot as plt
import pickle 
import numpy as np

def plot_percentiles(x):
    plt.figure(figsize=(14,7)) 
    plt.style.use('seaborn-whitegrid') 
    plt.hist(x, bins=20, density=True, facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5)
    plt.xlim(0,100)
    plt.xlabel('Predicted action percentile') 
    plt.ylabel('Density of ') 
    plt.show()

    
def percentage_greater_than(lst, x):
    count_greater_than_x = sum(1 for item in lst if item < x)
    percentage = (count_greater_than_x / len(lst)) * 100
    
    return percentage
    

with open('newmodel/6x6percentiles.pkl', 'rb') as file:
    scores = pickle.load(file)

scores = [i-5 for i in scores]
print(percentage_greater_than(scores, 20))
plot_percentiles(scores)