import matplotlib.pyplot as plt
import pickle 
import numpy as np

def plot_percentiles(model_percentiles):
    plt.hist(model_percentiles, density=True,alpha=0.5, label='Dataset 3', edgecolor='white')
    plt.xlim(0, 100)
    plt.title("Histograms of Datasets")
    plt.xlabel("Chosen action ranking percentile")
    plt.ylabel("% datasets")
    plt.legend()
    plt.show()

    
def percentage_greater_than(lst, x):
    count_greater_than_x = sum(1 for item in lst if item < x)
    percentage = (count_greater_than_x / len(lst)) * 100
    
    return percentage
    

try:
    with open('10x10percentiles.pkl', 'rb') as file:
        loaded_array = pickle.load(file)
except EOFError:
    print('Error: The file is empty or not properly pickled.')
except Exception as e:
    print(f'An error occurred: {e}')




print(percentage_greater_than(loaded_array, 20))
