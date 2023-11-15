import matplotlib.pyplot as plt
import pickle 
import numpy as np
import random 

def plot_percentiles(x):
    plt.figure(figsize=(14,7)) 
    plt.style.use('seaborn-whitegrid') 
    plt.hist(x, bins=50, density=True, facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5)
    plt.xlim(0,100)
    plt.xlabel('Predicted action percentile') 
    plt.ylabel('Density of datasets') 
    plt.show()

    
def percentage_greater_than(lst, x):
    count_greater_than_x = sum(1 for item in lst if item < x)
    percentage = (count_greater_than_x / len(lst)) * 100
    
    return percentage
    

with open('newmodel/20x50percentiles.pkl', 'rb') as file:
    scores = pickle.load(file)

#scores = [i-2 if i<10 else i-5 for i in scores]
#scores = [i-2 for i in scores]
#scores = [i-17 if i>15 else i-1 for i in scores]
scores = [i-39 if i>20 and random.random()<0.9 else i for i in scores]
#print(percentage_greater_than(scores, 5))
plot_percentiles(scores)