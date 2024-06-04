import pandas as pd
import matplotlib.pyplot as plt

def plot_start(figsize=(15, 5), title="Vector Prediction"):
    # Create a new figure for plotting
    plt.figure(figsize=(15, 5))

    # Add title and labels
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Value')


def plot_end():
    # Show legend
    plt.legend()
    
    # Display the plot
    plt.show()

def plot_vectors_layer(data, label="Line", linestyle='solid'):
    # Plot each column as a separate line
    for column in data.columns:
        plt.plot(data[column], label=f'{label} {column + 1}', linestyle=linestyle)
        

def plot_vectors(data):
    plot_start()
    plot_vectors_layer(pd.DataFrame(data))
    plot_end()
    
#pData = plot_vectors(pd.read_csv('fft.csv', header=None)) # Plot .csv file for example, where row is vector
