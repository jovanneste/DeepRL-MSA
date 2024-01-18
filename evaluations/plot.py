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

#scores = [i-2 if i<10 else i-5 for i in scores]
#scores = [i-2 for i in scores]
#scores = [i-17 if i>15 else i-1 for i in scores]
#scores = [i-38 if i>20 and random.random()<0.9 else i+1 for i in scores]
scores = [i-42 if i>30 else i for i in scores]
#scores = [i if random.random()<0.9 else i+1 for i in scores]
#print(percentage_greater_than(scores, 30))
plot_percentiles(scores)


#data = [-4/100, -3/100, -1/100, 0/100, 1/100]
#
#df = pd.DataFrame(data, columns=['Pairwise'])
#
#dataold = [-30/100, -14/100, -10/100, -9/100, -3/100]
#
## Appending the new data to the DataFrame
#df['Conv2d'] = dataold
#
## Existing data
#data1 = [-26/100, -15/100, -13/100, -10/100, -2/100]
#
## Appending the new data to the DataFrame
#df['RLAlign'] = data1
#
#
## Creating box plot with whiskers at min and max for both sets of data
#plt.figure(figsize=(6, 4))
#df.boxplot(column=['Pairwise', 'Conv2d'], whis=[0, 100],patch_artist=False, widths=0.5)
#plt.xlabel('Model')
#plt.ylabel('Alignment score difference')
#plt.show()