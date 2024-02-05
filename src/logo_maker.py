import pandas as pd
import matplotlib.pyplot as plt
import logomaker

import numpy as np

#import PIL
import matplotlib
import seaborn as sns
matplotlib.use('WebAgg')

def makeLogo(file, name):

    fig = plt.plot()

    tp = pd.read_pickle(file)
    logomaker.validate_matrix(tp)
    n=logomaker.Logo(tp,shade_below=0,fade_below=.7)
    logomaker.Logo.highlight_position_range(n,pmin=200,pmax=201,color='antiquewhite')

    n.style_spines(visible=False)
    n.style_spines(spines=['left', 'bottom'], visible=True)
    n.ax.set_xlim(200-20, 200+20)
    n.ax.set_ylabel('Contribution Score')
    n.ax.set_xlabel('Position')
    # n.ax.set_xticks(range(197-20,197+20))
    # n.ax.set_xticklabels([i for i in range(-19,21)])
    n.ax.set_xticks([200 + i for i in range(-20, 21, 5)])
    n.ax.set_xticklabels([i for i in range(-20, 21, 5)])
    n.ax.set_yticks(range(-14,6,2))
    n.ax.set_yticklabels([i for i in range(-14,6,2)])

    # plt.show()
    plt.savefig(f'{name}.png')

def makeViolinPlot(file, name):
    fig, ax = plt.subplots(figsize=(10, 6))

    fig = plt.plot()

    tp = pd.read_pickle(file)
    
    # Melt the DataFrame for violin plot
    melted_tp = tp.melt(var_name='Nucleotide', value_name='Contribution Score')
    
    # Create a violin plot
    sns.violinplot(x=melted_tp['Nucleotide'], y=melted_tp['Contribution Score'], ax=ax)
    
    # Highlight a specific position
    ax.axhline(y=0, color='red', linestyle='--', label='Contribution Score 0')
    
    # Set labels and title
    ax.set_xlabel('Nucleotide')
    ax.set_ylabel('Contribution Score')
    ax.set_title('DNA Sequence Logo - Violin Plot')
    
    # Add legend
    ax.legend()
    
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{name}.png')


def makePolarPlot(file, name):
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection='polar')

    tp = pd.read_pickle(file)
    
    # Get the columns (nucleotides) and positions
    nucleotides = tp.columns
    positions = np.arange(180, 221)  # Positions from 180 to 220
    num_positions = len(positions)
    
    # Convert positions to radians for polar plot
    theta = np.linspace(0, 2 * np.pi, num_positions, endpoint=False)
    
    # Normalize contribution scores for plotting
    normalized_scores = tp.loc[180:220, :] / np.max(tp)
    
    # Create a polar plot for each nucleotide
    for idx, nucleotide in enumerate(nucleotides):
        ax.plot(theta, normalized_scores[nucleotide], label=nucleotide)
    
    # Set the y-axis to show 0 to 1
    ax.set_ylim(0, 1)
    
    # Highlight a specific position
    highlighted_theta = ((200 - 180) / num_positions) * 2 * np.pi
    ax.plot([highlighted_theta, highlighted_theta], [0, 1], color='red', linestyle='--', label='Highlighted Position')
    
    # Set labels and title
    ax.set_xticks(theta)
    ax.set_xticklabels(positions)
    ax.set_xlabel('Position')
    ax.set_title('DNA Sequence Logo - Polar Plot')
    
    # Add a legend
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'{name}.png')



