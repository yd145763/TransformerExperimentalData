# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 11:33:52 2024

@author: limyu
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ast

from matplotlib.ticker import StrMethodFormatter
df_vit = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/ViT_full_result1.csv", index_col = 0)
df_trans = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/transformer_full_result1.csv", index_col = 0)
df_vit = df_vit.iloc[:150, :]
df_trans = df_trans.iloc[:150, :]


def find_TV_difference(df):
    difference_list = []
    epoch_list = []
    for i in range(len(df)):
        validation_accuracy_list = df['validation_accuracy_list']
        validation_accuracy = validation_accuracy_list[i]
        # Convert the string to a list
        validation_accuracy = ast.literal_eval(validation_accuracy)
        epoch_list.append(len(validation_accuracy))
        validation_accuracy = np.array(validation_accuracy)
        max_index = validation_accuracy.argmax()
        
        training_loss_list = df['training_loss_list']
        training_loss = training_loss_list[i]
        # Convert the string to a list
        training_loss = ast.literal_eval(training_loss)
        training_loss_best = training_loss[max_index]
        
        validation_loss_list = df['validation_loss_list']
        validation_loss = validation_loss_list[i]
        # Convert the string to a list
        validation_loss = ast.literal_eval(validation_loss)
        validation_loss_best = validation_loss[max_index]
        difference = validation_loss_best - training_loss_best
        
        difference_list.append(difference)
    
    return difference_list, epoch_list

difference_trans, epoch_list_trans = find_TV_difference(df_trans)
difference_vit, epoch_list_vit = find_TV_difference(df_vit)

# Combine data into a list
data =[difference_vit, difference_trans]

fig = plt.figure(figsize=(6, 4))
ax = plt.axes()
# Create box plots
box = ax.boxplot(data, patch_artist=True, widths=0.5)

# Set the fill color of each box to 'none' and adjust line thickness
for patch in box['boxes']:
    patch.set(facecolor='none')
    patch.set_linewidth(1)  # Adjust the thickness of the box lines

# Adjust whisker and cap thickness
for whisker in box['whiskers']:
    whisker.set_linewidth(1)  # Adjust the thickness of the whisker lines
for cap in box['caps']:
    cap.set_linewidth(1)  # Adjust the thickness of the cap lines

# Adjust median line thickness
for median in box['medians']:
    median.set_linewidth(1.5)  # Adjust the thickness of the median line
    median.set(color='red')

# Adjust flier (outlier) markers
for flier in box['fliers']:
    flier.set_markeredgewidth(1)  # Adjust the thickness of the marker edges

#graph formatting     
ax.tick_params(which='major', width=2.00)
ax.tick_params(which='minor', width=2.00)
ax.xaxis.label.set_fontsize(15)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(15)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
labels = ['Input Patches', 'Input Sequence']
ax.set_xticklabels(labels, weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
#ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_linewidth(1)
ax.spines["top"].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)

plt.ylabel("(Validation - Training)")

# Show the plot
plt.show()

# Combine data into a list
data =[epoch_list_vit, epoch_list_vit]

fig = plt.figure(figsize=(6, 4))
ax = plt.axes()
# Create box plots
box = ax.boxplot(data, patch_artist=True, widths=0.5)

# Set the fill color of each box to 'none' and adjust line thickness
for patch in box['boxes']:
    patch.set(facecolor='none')
    patch.set_linewidth(1)  # Adjust the thickness of the box lines

# Adjust whisker and cap thickness
for whisker in box['whiskers']:
    whisker.set_linewidth(1)  # Adjust the thickness of the whisker lines
for cap in box['caps']:
    cap.set_linewidth(1)  # Adjust the thickness of the cap lines

# Adjust median line thickness
for median in box['medians']:
    median.set_linewidth(1.5)  # Adjust the thickness of the median line
    median.set(color='red')

# Adjust flier (outlier) markers
for flier in box['fliers']:
    flier.set_markeredgewidth(1)  # Adjust the thickness of the marker edges

#graph formatting     
ax.tick_params(which='major', width=2.00)
ax.tick_params(which='minor', width=2.00)
ax.xaxis.label.set_fontsize(15)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(15)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
labels = ['Input Patches', 'Input Sequence']
ax.set_xticklabels(labels, weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
#ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_linewidth(1)
ax.spines["top"].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)

plt.ylabel("Epoch\n(Before Stopping)")

# Show the plot
plt.show()

time_list_vit = df_vit['time_list']
time_list_trans = df_trans['time_list']
# Combine data into a list
data =[time_list_vit, time_list_trans]

fig = plt.figure(figsize=(6, 4))
ax = plt.axes()
# Create box plots
box = ax.boxplot(data, patch_artist=True, widths=0.5)

# Set the fill color of each box to 'none' and adjust line thickness
for patch in box['boxes']:
    patch.set(facecolor='none')
    patch.set_linewidth(1)  # Adjust the thickness of the box lines

# Adjust whisker and cap thickness
for whisker in box['whiskers']:
    whisker.set_linewidth(1)  # Adjust the thickness of the whisker lines
for cap in box['caps']:
    cap.set_linewidth(1)  # Adjust the thickness of the cap lines

# Adjust median line thickness
for median in box['medians']:
    median.set_linewidth(1.5)  # Adjust the thickness of the median line
    median.set(color='red')

# Adjust flier (outlier) markers
for flier in box['fliers']:
    flier.set_markeredgewidth(1)  # Adjust the thickness of the marker edges

#graph formatting     
ax.tick_params(which='major', width=2.00)
ax.tick_params(which='minor', width=2.00)
ax.xaxis.label.set_fontsize(15)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(15)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
labels = ['Input Patches', 'Input Sequence']
ax.set_xticklabels(labels, weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
#ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_linewidth(1)
ax.spines["top"].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)

plt.ylabel("Training duration (s)")

# Show the plot
plt.show()


























conf = df_trans['confusion_matrix_list']
x = conf[99]
print(x)

x = x.replace('[', '').replace(']', '')

confusion_matrix = np.array([list(map(int, row.split())) for row in x.strip().split('\n')])

plt.figure(figsize=(6, 4))
ax = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="cool", 
            xticklabels=['Region A', 'Region B', 'Region C', 'Region D'], 
            yticklabels=['Region A', 'Region B', 'Region C', 'Region D'])


cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=15, width=2, length=6, pad=10, direction="out", colors="black", labelcolor="black")
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")

font = {'color': 'black', 'weight': 'bold', 'size': 15}
ax.set_ylabel("Actual", fontdict=font)
ax.set_xlabel("Predicted", fontdict=font)

# Setting tick labels bold
ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold")
ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold")
#ax.tick_params(axis='both', labelsize=12, weight='bold')
for i, text in enumerate(ax.texts):
    text.set_fontsize(15)
for i, text in enumerate(ax.texts):
    text.set_fontweight('bold')
#plt.title("Confusion Matrix"+"_"+"HEADS"+str(N_HEADS)+"_"+"LAYERS"+str(N_LAYERS)+"_"+"MLP"+str(N_DENSE_UNITS)+"_"+"ImageSize"+str(SIZE)
#          +"\n"+"Accuracy"+str(accuracy))
plt.show()
plt.close()



# Example data
vit_accuracy = df_vit['accuracy_list']
trans_accuracy = df_trans['accuracy_list']

# Combine data into a list
data =[vit_accuracy, trans_accuracy]

fig = plt.figure(figsize=(6, 4))
ax = plt.axes()
# Create box plots
box = ax.boxplot(data, patch_artist=True, widths=0.5)

# Set the fill color of each box to 'none' and adjust line thickness
for patch in box['boxes']:
    patch.set(facecolor='none')
    patch.set_linewidth(1)  # Adjust the thickness of the box lines

# Adjust whisker and cap thickness
for whisker in box['whiskers']:
    whisker.set_linewidth(1)  # Adjust the thickness of the whisker lines
for cap in box['caps']:
    cap.set_linewidth(1)  # Adjust the thickness of the cap lines

# Adjust median line thickness
for median in box['medians']:
    median.set_linewidth(1.5)  # Adjust the thickness of the median line
    median.set(color='red')

# Adjust flier (outlier) markers
for flier in box['fliers']:
    flier.set_markeredgewidth(1)  # Adjust the thickness of the marker edges

#graph formatting     
ax.tick_params(which='major', width=2.00)
ax.tick_params(which='minor', width=2.00)
ax.xaxis.label.set_fontsize(15)
ax.xaxis.label.set_weight("bold")
ax.yaxis.label.set_fontsize(15)
ax.yaxis.label.set_weight("bold")
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_yticklabels(ax.get_yticks(), weight='bold')
labels = ['Input Patches', 'Input Sequence']
ax.set_xticklabels(labels, weight='bold')
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
#ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.spines["right"].set_linewidth(1)
ax.spines["top"].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)

plt.ylabel("Accuracy")

# Show the plot
plt.show()

# Extracting minimum and maximum points from the box plot
for i, whisker in enumerate(box['whiskers']):
    if i % 2 == 0:  # even index, represents minimum point
        min_val = whisker.get_ydata()[1]
        print(f'Minimum point for box {i//2 + 1}: {min_val}')
    else:  # odd index, represents maximum point
        max_val = whisker.get_ydata()[1]
        print(f'Maximum point for box {i//2 + 1}: {max_val}')
