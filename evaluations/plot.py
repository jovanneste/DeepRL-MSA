import matplotlib.pyplot as plt
import pickle 
import numpy as np
import random 
import pandas as pd

def plot_percentiles(x):
    plt.figure(figsize=(5,5)) 
    plt.style.use('seaborn-whitegrid') 
    plt.hist(x,bins=10,density=False, facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.75)
    plt.xlim(0,10)
    plt.xlabel('Predicted action percentile') 
    plt.ylabel('Frequency') 
    plt.show()

    
def percentage_greater_than(lst, x):
    count_greater_than_x = sum(1 for item in lst if item < x)
    percentage = (count_greater_than_x / len(lst)) * 100
    return percentage
    

with open('oldmodel/10x10percentiles.pkl', 'rb') as file:
    scores = pickle.load(file)
    #plotted in Jupyter notebook
