# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 20:18:02 2024

@author: limyu
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 07:18:42 2024

@author: limyu
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
training_loss_list = []
training_accuracy_list = []
validation_loss_list = []
validation_accuracy_list = []
confusion_matrix_list = []
accuracy_list = []
time_list = []
confusion_matrix_list06 = []
confusion_matrix_list08 = []
confusion_matrix_list09 = []
confusion_matrix_list12 = []

N = np.arange(0,161,1)

n = 40

df_full = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/last_three_from_source/last_three_from_source_"+str(n)+".ascii_0001.ascii.csv")
df_full = df_full.reset_index(drop = True)
df_full.columns = range(df_full.shape[1])
df = df_full.iloc[25:65, 129:169]

   
df = df.reset_index(drop = True)
df.columns = range(df.shape[1])


data = np.array(df)

# Create a figure and axes
fig, ax = plt.subplots()

# Plot the data using imshow
im = ax.imshow(data, cmap='cool', interpolation='none')
# Adjust spacing between subplots
plt.tight_layout()
plt.axis('off')
# Show the plot
plt.show()

# Example 40x40 data array
#data = np.random.rand(40, 40)

# Define patch size
patch_size = 10
num_patches = data.shape[0] // patch_size

# Calculate the global min and max for normalization
data_min = np.min(data)
data_max = np.max(data)

# Create a figure with subplots
fig, axs = plt.subplots(num_patches, num_patches, figsize=(10, 10))

# Plot each patch in its own subplot with standardized color scales
for i in range(num_patches):
    for j in range(num_patches):
        patch = data[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
        ax = axs[i, j]
        im = ax.imshow(patch, cmap='cool', interpolation='none', vmin=data_min, vmax=data_max)
        ax.set_xticks([])
        ax.set_yticks([])

# Adjust spacing between subplots
plt.tight_layout()


# Show the plot
plt.show()





#====================================Transformer=============================

# pitch = 0.9 um
N = np.arange(0,161,1)
dataset = []
label = []
pitch = []
for n in N[:30]:
    df_full = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/second_last_three_from_source/second_last_three_from_source_"+str(n)+".ascii_0001.ascii.csv")
    df_full = df_full.reset_index(drop = True)
    df_full.columns = range(df_full.shape[1])
    df_full.iloc[44,241] = (df_full.iloc[43,241] + df_full.iloc[45,241])/2
    df = df_full.iloc[25:65, 205:285]
    df = df.reset_index(drop = True)
    df.columns = range(df.shape[1])
    df_smaller = df.iloc[10:30,30:50]
    
    row_index, col_index = np.unravel_index(df_smaller.values.argmax(), df_smaller.shape)
    row_index = row_index +10 
    col_index = col_index+30
    
    e = df.iloc[row_index, col_index-20:col_index+20]
    e = np.array(e)
    """
    plt.plot(e)
    plt.title(str(n))
    plt.show()
    plt.close()
    """
    e = np.reshape(e, (1, 40, 1))
    dataset.append(e)
    label.append(0)
    pitch.append(2)
    print('0.9um', n)

for n in N[30:80]:
    df_full = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/second_last_three_from_source/second_last_three_from_source_"+str(n)+".ascii_0001.ascii.csv")
    df_full = df_full.reset_index(drop = True)
    df_full.columns = range(df_full.shape[1])
    df_full.iloc[44,241] = (df_full.iloc[43,241] + df_full.iloc[45,241])/2
    df = df_full.iloc[25:65, 205:285]
    df = df.reset_index(drop = True)
    df.columns = range(df.shape[1])
    df_smaller = df.iloc[10:30,30:50]
    
    row_index, col_index = np.unravel_index(df_smaller.values.argmax(), df_smaller.shape)
    row_index = row_index +10 
    col_index = col_index+30
    
    e = df.iloc[row_index, col_index-20:col_index+20]
    e = np.array(e)
    """
    plt.plot(e)
    plt.title(str(n))
    plt.show()
    plt.close()
    """
    e = np.reshape(e, (1, 40, 1))
    dataset.append(e)
    label.append(1)
    pitch.append(2)
    print('0.9um', n)

for n in N[80:120]:
    df_full = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/second_last_three_from_source/second_last_three_from_source_"+str(n)+".ascii_0001.ascii.csv")
    df_full = df_full.reset_index(drop = True)
    df_full.columns = range(df_full.shape[1])
    df_full.iloc[44,241] = (df_full.iloc[43,241] + df_full.iloc[45,241])/2
    df = df_full.iloc[25:65, 205:285]
    df = df.reset_index(drop = True)
    
    
    df.columns = range(df.shape[1])
    df_smaller = df.iloc[10:30,30:50]
    
    row_index, col_index = np.unravel_index(df_smaller.values.argmax(), df_smaller.shape)
    row_index = row_index +10 
    col_index = col_index+30
    
    e = df.iloc[row_index, col_index-20:col_index+20]
    e = np.array(e)
    """
    plt.plot(e)
    plt.title(str(n))
    plt.show()
    plt.close()
    """
    e = np.reshape(e, (1, 40, 1))
    dataset.append(e)
    label.append(2)
    pitch.append(2)
    print('0.9um', n)
    
for n in N[120:161]:
    df_full = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/second_last_three_from_source/second_last_three_from_source_"+str(n)+".ascii_0001.ascii.csv")
    df_full = df_full.reset_index(drop = True)
    df_full.columns = range(df_full.shape[1])
    df_full.iloc[44,241] = (df_full.iloc[43,241] + df_full.iloc[45,241])/2
    df = df_full.iloc[25:65, 205:285]
    
 
    df = df.reset_index(drop = True)

    df.columns = range(df.shape[1])
    
    df_smaller = df.iloc[10:30,30:50]
    
    row_index, col_index = np.unravel_index(df_smaller.values.argmax(), df_smaller.shape)
    row_index = row_index +10 
    col_index = col_index+30
    
    e = df.iloc[row_index, col_index-20:col_index+20]
    e = np.array(e)
    """
    plt.plot(e)
    plt.title(str(n))
    plt.show()
    plt.close()
    """
    e = np.reshape(e, (1, 40, 1))
    dataset.append(e)
    label.append(3)
    pitch.append(2)
    print('0.9um', n)


# pitch = 0.9 um second
N = np.arange(0,161,1)
for n in N[:30]:
    df_full = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/second_last_three_from_source/second_last_three_from_source_"+str(n)+".ascii_0001.ascii.csv")
    df_full = df_full.reset_index(drop = True)
    df_full.columns = range(df_full.shape[1])
    df_full.iloc[25,118] = (df_full.iloc[25,117] + df_full.iloc[25,119])/2
    df_full.iloc[20,136] = (df_full.iloc[20,135] + df_full.iloc[20,134])/2
    df = df_full.iloc[20:60, 77:157]
     
    df = df.reset_index(drop = True)
    df.columns = range(df.shape[1])
    df_smaller = df.iloc[10:30,30:50]
    
    row_index, col_index = np.unravel_index(df_smaller.values.argmax(), df_smaller.shape)
    row_index = row_index +10 
    col_index = col_index+30
    
    e = df.iloc[row_index, col_index-20:col_index+20]
    e = np.array(e)
    """
    plt.plot(e)
    plt.title(str(n))
    plt.show()
    plt.close()
    """
    e = np.reshape(e, (1, 40, 1))
    dataset.append(e)
    label.append(0)
    pitch.append(2)
    print('0.9um', n)
    
    
for n in N[30:80]:
    df_full = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/second_last_three_from_source/second_last_three_from_source_"+str(n)+".ascii_0001.ascii.csv")
    df_full = df_full.reset_index(drop = True)
    df_full.columns = range(df_full.shape[1])
    df_full.iloc[25,118] = (df_full.iloc[25,117] + df_full.iloc[25,119])/2
    df_full.iloc[20,136] = (df_full.iloc[20,135] + df_full.iloc[20,134])/2
    df = df_full.iloc[20:60, 77:157]
     
    df = df.reset_index(drop = True)
    df.columns = range(df.shape[1])
    df_smaller = df.iloc[10:30,30:50]
    
    row_index, col_index = np.unravel_index(df_smaller.values.argmax(), df_smaller.shape)
    row_index = row_index +10 
    col_index = col_index+30
    
    e = df.iloc[row_index, col_index-20:col_index+20]
    e = np.array(e)
    """
    plt.plot(e)
    plt.title(str(n))
    plt.show()
    plt.close()
    """
    e = np.reshape(e, (1, 40, 1))
    dataset.append(e)
    label.append(1)
    pitch.append(2)
    print('0.9um', n)

for n in N[80:120]:
    df_full = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/second_last_three_from_source/second_last_three_from_source_"+str(n)+".ascii_0001.ascii.csv")
    df_full = df_full.reset_index(drop = True)
    df_full.columns = range(df_full.shape[1])
    df_full.iloc[25,118] = (df_full.iloc[25,117] + df_full.iloc[25,119])/2
    df_full.iloc[20,136] = (df_full.iloc[20,135] + df_full.iloc[20,134])/2
    df = df_full.iloc[20:60, 77:157]
     
    df = df.reset_index(drop = True)
    df.columns = range(df.shape[1])
    df_smaller = df.iloc[10:30,30:50]
    
    row_index, col_index = np.unravel_index(df_smaller.values.argmax(), df_smaller.shape)
    row_index = row_index +10 
    col_index = col_index+30
    
    e = df.iloc[row_index, col_index-20:col_index+20]
    e = np.array(e)
    """
    plt.plot(e)
    plt.title(str(n))
    plt.show()
    plt.close()
    """
    e = np.reshape(e, (1, 40, 1))
    dataset.append(e)
    label.append(2)
    pitch.append(2)
    print('0.9um', n)
    
for n in N[120:161]:
    df_full = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/second_last_three_from_source/second_last_three_from_source_"+str(n)+".ascii_0001.ascii.csv")
    df_full = df_full.reset_index(drop = True)
    df_full.columns = range(df_full.shape[1])
    df_full.iloc[25,118] = (df_full.iloc[25,117] + df_full.iloc[25,119])/2
    df_full.iloc[20,136] = (df_full.iloc[20,135] + df_full.iloc[20,134])/2
    df = df_full.iloc[20:60, 77:157]
     
    df = df.reset_index(drop = True)
    df.columns = range(df.shape[1])
    df_smaller = df.iloc[10:30,30:50]
    
    row_index, col_index = np.unravel_index(df_smaller.values.argmax(), df_smaller.shape)
    row_index = row_index +10 
    col_index = col_index+30
    
    e = df.iloc[row_index, col_index-20:col_index+20]
    e = np.array(e)
    """
    plt.plot(e)
    plt.title(str(n))
    plt.show()
    plt.close()
    """
    e = np.reshape(e, (1, 40, 1))
    dataset.append(e)
    label.append(3)
    pitch.append(2)
    print('0.9um', n)

# pitch = 0.6 um
N = np.arange(0,161,1)
for n in N[:30]:
    df_full = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/last_three_from_source/last_three_from_source_"+str(n)+".ascii_0001.ascii.csv")
    df_full = df_full.reset_index(drop = True)
    df_full.columns = range(df_full.shape[1])
    df = df_full.iloc[32:72, 235:315]
     
    df = df.reset_index(drop = True)
    df.columns = range(df.shape[1])
    df_smaller = df.iloc[10:30,30:50]
    
    row_index, col_index = np.unravel_index(df_smaller.values.argmax(), df_smaller.shape)
    row_index = row_index +10 
    col_index = col_index+30
    
    e = df.iloc[row_index, col_index-20:col_index+20]
    e = np.array(e)
    """
    plt.plot(e)
    plt.title(str(n))
    plt.show()
    plt.close()
    """
    e = np.reshape(e, (1, 40, 1))
    dataset.append(e)
    label.append(0)
    pitch.append(0)
    print('0.6um', n)

for n in N[30:80]:
    df_full = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/last_three_from_source/last_three_from_source_"+str(n)+".ascii_0001.ascii.csv")
    df_full = df_full.reset_index(drop = True)
    df_full.columns = range(df_full.shape[1])
    df = df_full.iloc[32:72, 235:315]
     
    df = df.reset_index(drop = True)
    df.columns = range(df.shape[1])
    df_smaller = df.iloc[10:30,30:50]
    
    row_index, col_index = np.unravel_index(df_smaller.values.argmax(), df_smaller.shape)
    row_index = row_index +10 
    col_index = col_index+30
    
    e = df.iloc[row_index, col_index-20:col_index+20]
    e = np.array(e)
    """
    plt.plot(e)
    plt.title(str(n))
    plt.show()
    plt.close()
    """
    e = np.reshape(e, (1, 40, 1))
    dataset.append(e)
    label.append(1)
    pitch.append(0)
    print('0.6um', n)

for n in N[80:120]:
    df_full = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/last_three_from_source/last_three_from_source_"+str(n)+".ascii_0001.ascii.csv")
    df_full = df_full.reset_index(drop = True)
    df_full.columns = range(df_full.shape[1])
    df = df_full.iloc[32:72, 235:315]
     
    df = df.reset_index(drop = True)
    df.columns = range(df.shape[1])
    df_smaller = df.iloc[10:30,30:50]
    
    row_index, col_index = np.unravel_index(df_smaller.values.argmax(), df_smaller.shape)
    row_index = row_index +10 
    col_index = col_index+30
    
    e = df.iloc[row_index, col_index-20:col_index+20]
    e = np.array(e)
    """
    plt.plot(e)
    plt.title(str(n))
    plt.show()
    plt.close()
    """
    e = np.reshape(e, (1, 40, 1))
    dataset.append(e)
    label.append(2)
    pitch.append(0)
    print('0.6um', n)
    
for n in N[120:161]:
    df_full = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/last_three_from_source/last_three_from_source_"+str(n)+".ascii_0001.ascii.csv")
    df_full = df_full.reset_index(drop = True)
    df_full.columns = range(df_full.shape[1])
    df = df_full.iloc[32:72, 235:315]
     
    df = df.reset_index(drop = True)
    df.columns = range(df.shape[1])
    df_smaller = df.iloc[10:30,30:50]
    
    row_index, col_index = np.unravel_index(df_smaller.values.argmax(), df_smaller.shape)
    row_index = row_index +10 
    col_index = col_index+30
    
    e = df.iloc[row_index, col_index-20:col_index+20]
    e = np.array(e)
    """
    plt.plot(e)
    plt.title(str(n))
    plt.show()
    plt.close()
    """
    e = np.reshape(e, (1, 40, 1))
    dataset.append(e)
    label.append(3)
    pitch.append(0)
    print('0.6um', n)

# pitch = 0.6 um second
N = np.arange(0,161,1)
for n in N[:30]:
    df_full = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/last_three_from_source/last_three_from_source_"+str(n)+".ascii_0001.ascii.csv")
    df_full = df_full.reset_index(drop = True)
    df_full.columns = range(df_full.shape[1])
    df = df_full.iloc[25:65, 109:189]
     
    df = df.reset_index(drop = True)
    df.columns = range(df.shape[1])
    df_smaller = df.iloc[10:30,30:50]
    
    row_index, col_index = np.unravel_index(df_smaller.values.argmax(), df_smaller.shape)
    row_index = row_index +10 
    col_index = col_index+30
    
    e = df.iloc[row_index, col_index-20:col_index+20]
    e = np.array(e)
    """
    plt.plot(e)
    plt.title(str(n))
    plt.show()
    plt.close()
    """
    e = np.reshape(e, (1, 40, 1))
    dataset.append(e)
    label.append(0)
    pitch.append(0)
    print('0.6um', n)

for n in N[30:80]:
    df_full = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/last_three_from_source/last_three_from_source_"+str(n)+".ascii_0001.ascii.csv")
    df_full = df_full.reset_index(drop = True)
    df_full.columns = range(df_full.shape[1])
    df = df_full.iloc[25:65, 109:189]
     
    df = df.reset_index(drop = True)
    df.columns = range(df.shape[1])
    df_smaller = df.iloc[10:30,30:50]
    
    row_index, col_index = np.unravel_index(df_smaller.values.argmax(), df_smaller.shape)
    row_index = row_index +10 
    col_index = col_index+30
    
    e = df.iloc[row_index, col_index-20:col_index+20]
    e = np.array(e)
    """
    plt.plot(e)
    plt.title(str(n))
    plt.show()
    plt.close()
    """
    e = np.reshape(e, (1, 40, 1))
    dataset.append(e)
    label.append(1)
    pitch.append(0)
    print('0.6um', n)

for n in N[80:120]:
    df_full = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/last_three_from_source/last_three_from_source_"+str(n)+".ascii_0001.ascii.csv")
    df_full = df_full.reset_index(drop = True)
    df_full.columns = range(df_full.shape[1])
    df = df_full.iloc[25:65, 109:189]
     
    df = df.reset_index(drop = True)
    df.columns = range(df.shape[1])
    df_smaller = df.iloc[10:30,30:50]
    
    row_index, col_index = np.unravel_index(df_smaller.values.argmax(), df_smaller.shape)
    row_index = row_index +10 
    col_index = col_index+30
    
    e = df.iloc[row_index, col_index-20:col_index+20]
    e = np.array(e)
    """
    plt.plot(e)
    plt.title(str(n))
    plt.show()
    plt.close()
    """
    e = np.reshape(e, (1, 40, 1))
    dataset.append(e)
    label.append(2)
    pitch.append(0)
    print('0.6um', n)
    
for n in N[120:161]:
    df_full = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/last_three_from_source/last_three_from_source_"+str(n)+".ascii_0001.ascii.csv")
    df_full = df_full.reset_index(drop = True)
    df_full.columns = range(df_full.shape[1])
    df = df_full.iloc[25:65, 109:189]
     
    df = df.reset_index(drop = True)
    df.columns = range(df.shape[1])
    df_smaller = df.iloc[10:30,30:50]
    
    row_index, col_index = np.unravel_index(df_smaller.values.argmax(), df_smaller.shape)
    row_index = row_index +10 
    col_index = col_index+30
    
    e = df.iloc[row_index, col_index-20:col_index+20]
    e = np.array(e)
    """
    plt.plot(e)
    plt.title(str(n))
    plt.show()
    plt.close()
    """
    e = np.reshape(e, (1, 40, 1))
    dataset.append(e)
    label.append(3)
    pitch.append(0)
    print('0.6um', n)
    
# pitch = 1.2 um
N = np.arange(0,161,1)
for n in N[:30]:
    df_full = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/last_three_from_source/last_three_from_source_"+str(n)+".ascii_0001.ascii.csv")
    df_full = df_full.reset_index(drop = True)
    df_full.columns = range(df_full.shape[1])
    df = df_full.iloc[198:238, 228:308]
     
    df = df.reset_index(drop = True)
    df.columns = range(df.shape[1])
    df_smaller = df.iloc[10:30,30:50]
    
    row_index, col_index = np.unravel_index(df_smaller.values.argmax(), df_smaller.shape)
    row_index = row_index +10 
    col_index = col_index+30
    
    e = df.iloc[row_index, col_index-20:col_index+20]
    e = np.array(e)
    """
    plt.plot(e)
    plt.title(str(n))
    plt.show()
    plt.close()
    """
    e = np.reshape(e, (1, 40, 1))
    dataset.append(e)
    label.append(0)
    pitch.append(3)
    print('1.2um', n)

for n in N[30:80]:
    df_full = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/last_three_from_source/last_three_from_source_"+str(n)+".ascii_0001.ascii.csv")
    df_full = df_full.reset_index(drop = True)
    df_full.columns = range(df_full.shape[1])
    df = df_full.iloc[198:238, 228:308]
     
    df = df.reset_index(drop = True)
    df.columns = range(df.shape[1])
    df_smaller = df.iloc[10:30,30:50]
    
    row_index, col_index = np.unravel_index(df_smaller.values.argmax(), df_smaller.shape)
    row_index = row_index +10 
    col_index = col_index+30
    
    e = df.iloc[row_index, col_index-20:col_index+20]
    e = np.array(e)
    """
    plt.plot(e)
    plt.title(str(n))
    plt.show()
    plt.close()
    """
    e = np.reshape(e, (1, 40, 1))
    dataset.append(e)
    label.append(1)
    pitch.append(3)
    print('1.2um', n)

for n in N[80:120]:
    df_full = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/last_three_from_source/last_three_from_source_"+str(n)+".ascii_0001.ascii.csv")
    df_full = df_full.reset_index(drop = True)
    df_full.columns = range(df_full.shape[1])
    df = df_full.iloc[198:238, 228:308]
     
    df = df.reset_index(drop = True)
    df.columns = range(df.shape[1])
    df_smaller = df.iloc[10:30,30:50]
    
    row_index, col_index = np.unravel_index(df_smaller.values.argmax(), df_smaller.shape)
    row_index = row_index +10 
    col_index = col_index+30
    
    e = df.iloc[row_index, col_index-20:col_index+20]
    e = np.array(e)
    """
    plt.plot(e)
    plt.title(str(n))
    plt.show()
    plt.close()
    """
    e = np.reshape(e, (1, 40, 1))
    dataset.append(e)
    label.append(2)
    pitch.append(3)
    print('1.2um', n)
    
for n in N[120:161]:
    df_full = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/last_three_from_source/last_three_from_source_"+str(n)+".ascii_0001.ascii.csv")
    df_full = df_full.reset_index(drop = True)
    df_full.columns = range(df_full.shape[1])
    df = df_full.iloc[198:238, 228:308]
     
    df = df.reset_index(drop = True)
    df.columns = range(df.shape[1])
    df_smaller = df.iloc[10:30,30:50]
    
    row_index, col_index = np.unravel_index(df_smaller.values.argmax(), df_smaller.shape)
    row_index = row_index +10 
    col_index = col_index+30
    
    e = df.iloc[row_index, col_index-20:col_index+20]
    e = np.array(e)
    """
    plt.plot(e)
    plt.title(str(n))
    plt.show()
    plt.close()
    """
    e = np.reshape(e, (1, 40, 1))
    dataset.append(e)
    label.append(3)
    pitch.append(3)
    print('1.2um', n)
    

# pitch = 1.2 um second
N = np.arange(0,161,1)
for n in N[:30]:
    df_full = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/last_three_from_source/last_three_from_source_"+str(n)+".ascii_0001.ascii.csv")
    df_full = df_full.reset_index(drop = True)
    df_full.columns = range(df_full.shape[1])
    df_full.iloc[207, 132:134] = (df_full.iloc[206, 132:134] +df_full.iloc[208, 132:134])/2
    df_full.iloc[207, 134] = (df_full.iloc[206, 134] +df_full.iloc[208, 134])/2
    df_full.iloc[211, 128:131] = (df_full.iloc[210, 128:131] +df_full.iloc[209, 128:131])/2
    df_full.iloc[212, 128:131] = (df_full.iloc[213, 128:131] +df_full.iloc[214, 128:131])/2
    df_full.iloc[223, 125:137] = (df_full.iloc[222, 125:137] +df_full.iloc[224, 125:137])/2
    df = df_full.iloc[194:234, 102:182]
     
    df = df.reset_index(drop = True)
    df.columns = range(df.shape[1])
    df_smaller = df.iloc[10:30,30:50]
    
    row_index, col_index = np.unravel_index(df_smaller.values.argmax(), df_smaller.shape)
    row_index = row_index +10 
    col_index = col_index+30
    
    e = df.iloc[row_index, col_index-20:col_index+20]
    e = np.array(e)
    """
    plt.plot(e)
    plt.title(str(n))
    plt.show()
    plt.close()
    """
    e = np.reshape(e, (1, 40, 1))
    dataset.append(e)
    label.append(0)
    pitch.append(3)
    print('1.2um', n)

for n in N[30:80]:
    df_full = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/last_three_from_source/last_three_from_source_"+str(n)+".ascii_0001.ascii.csv")
    df_full = df_full.reset_index(drop = True)
    df_full.columns = range(df_full.shape[1])
    df_full.iloc[207, 132:134] = (df_full.iloc[206, 132:134] +df_full.iloc[208, 132:134])/2
    df_full.iloc[207, 134] = (df_full.iloc[206, 134] +df_full.iloc[208, 134])/2
    df_full.iloc[211, 128:131] = (df_full.iloc[210, 128:131] +df_full.iloc[209, 128:131])/2
    df_full.iloc[212, 128:131] = (df_full.iloc[213, 128:131] +df_full.iloc[214, 128:131])/2
    df_full.iloc[223, 125:137] = (df_full.iloc[222, 125:137] +df_full.iloc[224, 125:137])/2   
    df = df_full.iloc[194:234, 102:182]
     
    df = df.reset_index(drop = True)
    df.columns = range(df.shape[1])
    df_smaller = df.iloc[10:30,30:50]
    
    row_index, col_index = np.unravel_index(df_smaller.values.argmax(), df_smaller.shape)
    row_index = row_index +10 
    col_index = col_index+30
    
    e = df.iloc[row_index, col_index-20:col_index+20]
    e = np.array(e)
    """
    plt.plot(e)
    plt.title(str(n))
    plt.show()
    plt.close()
    """
    e = np.reshape(e, (1, 40, 1))
    dataset.append(e)
    label.append(1)
    pitch.append(3)
    print('1.2um', n)

for n in N[80:120]:
    df_full = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/last_three_from_source/last_three_from_source_"+str(n)+".ascii_0001.ascii.csv")
    df_full = df_full.reset_index(drop = True)
    df_full.columns = range(df_full.shape[1])
    df_full.iloc[207, 132:134] = (df_full.iloc[206, 132:134] +df_full.iloc[208, 132:134])/2
    df_full.iloc[207, 134] = (df_full.iloc[206, 134] +df_full.iloc[208, 134])/2
    df_full.iloc[211, 128:131] = (df_full.iloc[210, 128:131] +df_full.iloc[209, 128:131])/2
    df_full.iloc[212, 128:131] = (df_full.iloc[213, 128:131] +df_full.iloc[214, 128:131])/2
    df_full.iloc[223, 125:137] = (df_full.iloc[222, 125:137] +df_full.iloc[224, 125:137])/2
    df = df_full.iloc[194:234, 102:182]
     
    df = df.reset_index(drop = True)
    df.columns = range(df.shape[1])
    df_smaller = df.iloc[10:30,30:50]
    
    row_index, col_index = np.unravel_index(df_smaller.values.argmax(), df_smaller.shape)
    row_index = row_index +10 
    col_index = col_index+30
    
    e = df.iloc[row_index, col_index-20:col_index+20]
    e = np.array(e)
    """
    plt.plot(e)
    plt.title(str(n))
    plt.show()
    plt.close()
    """
    e = np.reshape(e, (1, 40, 1))
    dataset.append(e)
    label.append(2)
    pitch.append(3)
    print('1.2um', n)
    
for n in N[120:161]:
    df_full = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/last_three_from_source/last_three_from_source_"+str(n)+".ascii_0001.ascii.csv")
    df_full = df_full.reset_index(drop = True)
    df_full.columns = range(df_full.shape[1])
    df_full.iloc[207, 132:134] = (df_full.iloc[206, 132:134] +df_full.iloc[208, 132:134])/2
    df_full.iloc[207, 134] = (df_full.iloc[206, 134] +df_full.iloc[208, 134])/2
    df_full.iloc[211, 128:131] = (df_full.iloc[210, 128:131] +df_full.iloc[209, 128:131])/2
    df_full.iloc[212, 128:131] = (df_full.iloc[213, 128:131] +df_full.iloc[214, 128:131])/2
    df_full.iloc[223, 125:137] = (df_full.iloc[222, 125:137] +df_full.iloc[224, 125:137])/2
    df = df_full.iloc[194:234, 102:182]
     
    df = df.reset_index(drop = True)
    df.columns = range(df.shape[1])
    df_smaller = df.iloc[10:30,30:50]
    
    row_index, col_index = np.unravel_index(df_smaller.values.argmax(), df_smaller.shape)
    row_index = row_index +10 
    col_index = col_index+30
    
    e = df.iloc[row_index, col_index-20:col_index+20]
    e = np.array(e)
    """
    plt.plot(e)
    plt.title(str(n))
    plt.show()
    plt.close()
    """
    e = np.reshape(e, (1, 40, 1))
    dataset.append(e)
    label.append(3)
    pitch.append(3)
    print('1.2um', n)


# pitch = 0.8 um
N = np.arange(0,161,1)
for n in N[:30]:
    df_full = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/last_three_from_source/last_three_from_source_"+str(n)+".ascii_0001.ascii.csv")
    df_full = df_full.reset_index(drop = True)
    df_full.columns = range(df_full.shape[1])
    df = df_full.iloc[115:155, 230:310]
     
    df = df.reset_index(drop = True)
    df.columns = range(df.shape[1])
    df_smaller = df.iloc[10:30,30:50]
    
    row_index, col_index = np.unravel_index(df_smaller.values.argmax(), df_smaller.shape)
    row_index = row_index +10 
    col_index = col_index+30
    
    e = df.iloc[row_index, col_index-20:col_index+20]
    e = np.array(e)
    """
    plt.plot(e)
    plt.title(str(n))
    plt.show()
    plt.close()
    """
    e = np.reshape(e, (1, 40, 1))
    dataset.append(e)
    label.append(0)
    pitch.append(1)
    print('0.8um', n)

for n in N[30:80]:
    df_full = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/last_three_from_source/last_three_from_source_"+str(n)+".ascii_0001.ascii.csv")
    df_full = df_full.reset_index(drop = True)
    df_full.columns = range(df_full.shape[1])
    df = df_full.iloc[115:155, 230:310]
     
    df = df.reset_index(drop = True)
    df.columns = range(df.shape[1])
    df_smaller = df.iloc[10:30,30:50]
    
    row_index, col_index = np.unravel_index(df_smaller.values.argmax(), df_smaller.shape)
    row_index = row_index +10 
    col_index = col_index+30
    
    e = df.iloc[row_index, col_index-20:col_index+20]
    e = np.array(e)
    """
    plt.plot(e)
    plt.title(str(n))
    plt.show()
    plt.close()
    """
    e = np.reshape(e, (1, 40, 1))
    dataset.append(e)
    label.append(1)
    pitch.append(1)
    print('0.8um', n)

for n in N[80:120]:
    df_full = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/last_three_from_source/last_three_from_source_"+str(n)+".ascii_0001.ascii.csv")
    df_full = df_full.reset_index(drop = True)
    df_full.columns = range(df_full.shape[1])
    df = df_full.iloc[115:155, 230:310]
     
    df = df.reset_index(drop = True)
    df.columns = range(df.shape[1])
    df_smaller = df.iloc[10:30,30:50]
    
    row_index, col_index = np.unravel_index(df_smaller.values.argmax(), df_smaller.shape)
    row_index = row_index +10 
    col_index = col_index+30
    
    e = df.iloc[row_index, col_index-20:col_index+20]
    e = np.array(e)
    """
    plt.plot(e)
    plt.title(str(n))
    plt.show()
    plt.close()
    """
    e = np.reshape(e, (1, 40, 1))
    dataset.append(e)
    label.append(2)
    pitch.append(1)
    print('0.8um', n)
    
for n in N[120:161]:
    df_full = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/last_three_from_source/last_three_from_source_"+str(n)+".ascii_0001.ascii.csv")
    df_full = df_full.reset_index(drop = True)
    df_full.columns = range(df_full.shape[1])
    df = df_full.iloc[115:155, 230:310]
     
    df = df.reset_index(drop = True)
    df.columns = range(df.shape[1])
    df_smaller = df.iloc[10:30,30:50]
    
    row_index, col_index = np.unravel_index(df_smaller.values.argmax(), df_smaller.shape)
    row_index = row_index +10 
    col_index = col_index+30
    
    e = df.iloc[row_index, col_index-20:col_index+20]
    e = np.array(e)
    """
    plt.plot(e)
    plt.title(str(n))
    plt.show()
    plt.close()
    """
    e = np.reshape(e, (1, 40, 1))
    dataset.append(e)
    label.append(3)
    pitch.append(1)
    print('0.8um', n)


# pitch = 0.8 um second
N = np.arange(0,161,1)
for n in N[:30]:
    df_full = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/last_three_from_source/last_three_from_source_"+str(n)+".ascii_0001.ascii.csv")
    df_full = df_full.reset_index(drop = True)
    df_full.columns = range(df_full.shape[1])
    df_full.iloc[125,163] = (df_full.iloc[125,162] + df_full.iloc[125,164])/2
    df = df_full.iloc[108:148, 105:185]
     
    df = df.reset_index(drop = True)
    df.columns = range(df.shape[1])
    df_smaller = df.iloc[10:30,30:50]
    
    row_index, col_index = np.unravel_index(df_smaller.values.argmax(), df_smaller.shape)
    row_index = row_index +10 
    col_index = col_index+30
    
    e = df.iloc[row_index, col_index-20:col_index+20]
    e = np.array(e)
    """
    plt.plot(e)
    plt.title(str(n))
    plt.show()
    plt.close()
    """
    e = np.reshape(e, (1, 40, 1))
    dataset.append(e)
    label.append(0)
    pitch.append(1)
    print('0.8um', n)

for n in N[30:80]:
    df_full = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/last_three_from_source/last_three_from_source_"+str(n)+".ascii_0001.ascii.csv")
    df_full = df_full.reset_index(drop = True)
    df_full.columns = range(df_full.shape[1])
    df_full.iloc[125,163] = (df_full.iloc[125,162] + df_full.iloc[125,164])/2
    df = df_full.iloc[108:148, 105:185]
     
    df = df.reset_index(drop = True)
    df.columns = range(df.shape[1])
    df_smaller = df.iloc[10:30,30:50]
    
    row_index, col_index = np.unravel_index(df_smaller.values.argmax(), df_smaller.shape)
    row_index = row_index +10 
    col_index = col_index+30
    
    e = df.iloc[row_index, col_index-20:col_index+20]
    e = np.array(e)
    """
    plt.plot(e)
    plt.title(str(n))
    plt.show()
    plt.close()
    """
    e = np.reshape(e, (1, 40, 1))
    dataset.append(e)
    label.append(1)
    pitch.append(1)
    print('0.8um', n)

for n in N[80:120]:
    df_full = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/last_three_from_source/last_three_from_source_"+str(n)+".ascii_0001.ascii.csv")
    df_full = df_full.reset_index(drop = True)
    df_full.columns = range(df_full.shape[1])
    df_full.iloc[125,163] = (df_full.iloc[125,162] + df_full.iloc[125,164])/2
    df = df_full.iloc[108:148, 105:185]
     
    df = df.reset_index(drop = True)
    df.columns = range(df.shape[1])
    df_smaller = df.iloc[10:30,30:50]
    
    row_index, col_index = np.unravel_index(df_smaller.values.argmax(), df_smaller.shape)
    row_index = row_index +10 
    col_index = col_index+30
    
    e = df.iloc[row_index, col_index-20:col_index+20]
    e = np.array(e)
    """
    plt.plot(e)
    plt.title(str(n))
    plt.show()
    plt.close()
    """
    e = np.reshape(e, (1, 40, 1))
    dataset.append(e)
    label.append(2)
    pitch.append(1)
    print('0.8um', n)
    
for n in N[120:161]:
    df_full = pd.read_csv("https://raw.githubusercontent.com/yd145763/TransformerExperimentalData/main/last_three_from_source/last_three_from_source_"+str(n)+".ascii_0001.ascii.csv")
    df_full = df_full.reset_index(drop = True)
    df_full.columns = range(df_full.shape[1])
    df_full.iloc[125,163] = (df_full.iloc[125,162] + df_full.iloc[125,164])/2
    df = df_full.iloc[108:148, 105:185]
     
    df = df.reset_index(drop = True)
    df.columns = range(df.shape[1])
    df_smaller = df.iloc[10:30,30:50]
    
    row_index, col_index = np.unravel_index(df_smaller.values.argmax(), df_smaller.shape)
    row_index = row_index +10 
    col_index = col_index+30
    
    e = df.iloc[row_index, col_index-20:col_index+20]
    e = np.array(e)
    """
    plt.plot(e)
    plt.title(str(n))
    plt.show()
    plt.close()
    """
    e = np.reshape(e, (1, 40, 1))
    dataset.append(e)
    label.append(3)
    pitch.append(1)
    print('0.8um', n)

dataset_list = dataset

import tensorflow as tf### models
import numpy as np### math computations
import matplotlib.pyplot as plt### plotting bar chart
import sklearn### machine learning library
import cv2## image processing
from sklearn.metrics import confusion_matrix, roc_curve### metrics
import seaborn as sns### visualizations
import datetime
import pathlib
import io
import os
import time
import random
from PIL import Image

import matplotlib.cm as cm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import (GlobalAveragePooling2D, Activation, MaxPooling2D, Add, Conv2D, MaxPool2D, Dense,
                                    Flatten, InputLayer, BatchNormalization, Input, Embedding, Permute,
                                    Dropout, RandomFlip, RandomRotation, LayerNormalization, MultiHeadAttention,
                                    RandomContrast, Rescaling, Resizing, Reshape)
from tensorflow.keras.losses import BinaryCrossentropy,CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy,TopKCategoricalAccuracy, CategoricalAccuracy, SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (Callback, CSVLogger, EarlyStopping, LearningRateScheduler,
                                        ModelCheckpoint, ReduceLROnPlateau)
from tensorflow.keras.regularizers  import L2, L1
from tensorflow.train import BytesList, FloatList, Int64List
from tensorflow.train import Example, Features, Feature

CONFIGURATION = {
    "BATCH_SIZE": 16,
    "SEQ_LENGTH": 40,
    "LEARNING_RATE": 1e-3,
    "N_EPOCHS": 5,
    "DROPOUT_RATE": 0.0,
    "REGULARIZATION_RATE": 0.0,
    "N_FILTERS": 2,
    "KERNEL_SIZE": 3,
    "N_STRIDES": 1,
    "POOL_SIZE": 2,
    "N_DENSE_1": 1024,
    "N_DENSE_2": 256,
    "NUM_CLASSES": 4,
    "PROJ_DIM": 192,
    "CLASS_NAMES": ["Region A", "Region B", "Region C", "Region D"],
    "HIDDEN_SIZE": 1,
}


class PatchEncoder(Layer):
    def __init__(self, SEQ_LENGTH, HIDDEN_SIZE):
        super(PatchEncoder, self).__init__(name='patch_encoder')
        self.linear_projection = Dense(HIDDEN_SIZE)
        self.positional_embedding = Embedding(SEQ_LENGTH, HIDDEN_SIZE)
        self.SEQ_LENGTH = SEQ_LENGTH

    def call(self, x):
        embedding_input = tf.range(start=0, limit=self.SEQ_LENGTH, delta=1)
        position_embeddings = self.positional_embedding(embedding_input)
        position_embeddings = tf.reshape(position_embeddings, [1, self.SEQ_LENGTH, -1])
        position_embeddings = tf.tile(position_embeddings, [tf.shape(x)[0], 1, 1])
        output = self.linear_projection(x) + position_embeddings
        return output


class TransformerEncoder(Layer):
    def __init__(self, N_HEADS, HIDDEN_SIZE):
        super(TransformerEncoder, self).__init__(name='transformer_encoder')
        self.layer_norm_1 = LayerNormalization()
        self.layer_norm_2 = LayerNormalization()
        self.multi_head_att = MultiHeadAttention(N_HEADS, HIDDEN_SIZE)
        self.dense_1 = Dense(HIDDEN_SIZE, activation=tf.nn.gelu)
        self.dense_2 = Dense(HIDDEN_SIZE, activation=tf.nn.gelu)

    def call(self, input):
        x_1 = self.layer_norm_1(input)
        x_1 = self.multi_head_att(x_1, x_1)
        x_1 = Add()([x_1, input])
        x_2 = self.layer_norm_2(x_1)
        x_2 = self.dense_1(x_2)
        output = self.dense_2(x_2)
        output = Add()([output, x_1])
        return output

class ViT(Model):
    def __init__(self, N_HEADS, HIDDEN_SIZE, SEQ_LENGTH, N_LAYERS, N_DENSE_UNITS):
        super(ViT, self).__init__(name='vision_transformer')
        self.N_LAYERS = N_LAYERS
        self.patch_encoder = PatchEncoder(SEQ_LENGTH, HIDDEN_SIZE)
        self.trans_encoders = [TransformerEncoder(N_HEADS, HIDDEN_SIZE) for _ in range(N_LAYERS)]
        self.flatten = Flatten()
        self.dense_1 = Dense(N_DENSE_UNITS, activation=tf.nn.gelu)
        self.dense_2 = Dense(N_DENSE_UNITS, activation=tf.nn.gelu)
        self.dense_3 = Dense(CONFIGURATION["NUM_CLASSES"], activation='softmax')

    def call(self, input, training=True):
        x = tf.reshape(input, (-1, CONFIGURATION["SEQ_LENGTH"], CONFIGURATION["HIDDEN_SIZE"]))  # Reshape to the correct shape
        x = self.patch_encoder(x)
        for i in range(self.N_LAYERS):
            x = self.trans_encoders[i](x)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        return self.dense_3(x)


# Define the model
vit = ViT(
    N_HEADS=1,
    HIDDEN_SIZE=CONFIGURATION["HIDDEN_SIZE"],
    SEQ_LENGTH=CONFIGURATION["SEQ_LENGTH"],
    N_LAYERS=1,  # Example: 1 transformer layer
    N_DENSE_UNITS=64  # Example dense unit size
)

# Compile the model
vit.compile(optimizer=tf.keras.optimizers.Adam(CONFIGURATION["LEARNING_RATE"]),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

# Example data
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Assuming dataset is your list of numpy arrays with shape (1, 40, 1)
# Convert list of numpy arrays to a single numpy array
dataset = np.array(dataset)

# Assuming labels are provided
label = np.array(label)

pitch_arr = np.array(pitch).reshape(len(pitch), 1, 1, 1)
pitch_arr = np.tile(pitch_arr, (1, 1, 40, 1))
combined_dataset = np.concatenate([dataset, pitch_arr], axis=-1)

# Split the data
X_train_full, X_test_full, y_train, y_test = train_test_split(combined_dataset, to_categorical(label), test_size=0.40, random_state=0)

X_train = X_train_full[..., :-1]
train_pitch = X_train_full[..., -1].squeeze()
X_test = X_test_full[..., :-1]
test_pitch = X_test_full[..., -1].squeeze()

# Define the EarlyStopping callback
from tensorflow.keras.callbacks import EarlyStopping
early_stopping_callback = EarlyStopping(
    monitor='val_accuracy',  # Monitor validation accuracy
    min_delta=0,  # Minimum change to qualify as an improvement
    patience=30,  # Number of epochs with no improvement after which training will be stopped
    verbose=1,  # Verbosity mode
    mode='max',  # Maximize the monitored quantity
    restore_best_weights=True  # Whether to restore model weights to the best observed during training
)

start_time = time.time()
# Train the model
history = vit.fit(X_train, 
                y_train, 
                batch_size=CONFIGURATION["BATCH_SIZE"], 
                verbose=1, 
                epochs=100,      
                validation_split=0.4,
                shuffle=True,
                callbacks=[early_stopping_callback]
)

end_time = time.time()

time_consumed = end_time - start_time
time_list.append(time_consumed)


print("Test_Accuracy: {:.2f}%".format(vit.evaluate(np.array(X_test), np.array(y_test))[1]*100))
accuracy = vit.evaluate(np.array(X_test), np.array(y_test))[1]*100



f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('CNN Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

max_epoch = len(history.history['accuracy'])+1
epoch_list = list(range(1,max_epoch))
ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(1, max_epoch, 5))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(1, max_epoch, 5))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")

training_loss = history.history['loss']
training_loss_list.append(training_loss)
training_accuracy = history.history['accuracy']
training_accuracy_list.append(training_accuracy)
validation_loss = history.history['val_loss']
validation_loss_list.append(validation_loss)
validation_accuracy = history.history['val_accuracy']
validation_accuracy_list.append(validation_accuracy)


from sklearn.metrics import confusion_matrix
import seaborn as sns

# Predict the test set results
y_pred = vit.predict(np.array(X_test))
# Convert predictions to binary classes
y_pred_classes = np.argmax(y_pred, axis=1)
# Convert true labels to binary classes
y_true = np.argmax(np.array(y_test), axis=1)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

#confusion_matrix_list.append(conf_matrix)
total_sum = np.sum(conf_matrix)
correct_answer = conf_matrix[0,0] + conf_matrix[1,1] +conf_matrix[2,2]+conf_matrix[3,3]
accuracy = correct_answer/total_sum
accuracy_list.append(accuracy)
confusion_matrix_list.append(conf_matrix)

plt.figure(figsize=(6, 4))
ax = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="cool", 
            xticklabels=['Region A', 'Region B', 'Region C', 'Region D'], 
            yticklabels=['Region A', 'Region B', 'Region C', 'Region D'])


cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12, width=2, length=6, pad=10, direction="out", colors="black", labelcolor="black")
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")

font = {'color': 'black', 'weight': 'bold', 'size': 12}
ax.set_ylabel("Actual", fontdict=font)
ax.set_xlabel("Predicted", fontdict=font)

# Setting tick labels bold
ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold")
ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold")
#ax.tick_params(axis='both', labelsize=12, weight='bold')
for i, text in enumerate(ax.texts):
    text.set_fontsize(12)
for i, text in enumerate(ax.texts):
    text.set_fontweight('bold')
#plt.title("Confusion Matrix"+"_"+"HEADS"+str(N_HEADS)+"_"+"LAYERS"+str(N_LAYERS)+"_"+"MLP"+str(N_DENSE_UNITS)+"_"+"ImageSize"+str(SIZE)
#          +"\n"+"Accuracy"+str(accuracy))
plt.show()
plt.close()



test_pitch = test_pitch[:, 0]

#+++++++++++++++++++++++++++++++++==0.6 um+++++++++++++++++++++++++++++
um06_index = [index for index, value in enumerate(test_pitch) if value == 0]

X_test_06 = [X_test[i] for i in um06_index]
X_test_06 = np.array(X_test_06)

y_test_06 = [y_test[i] for i in um06_index]
y_test_06 = np.array(y_test_06)

# Predict the test set results
y_pred = vit.predict(np.array(X_test_06))
# Convert predictions to binary classes
y_pred_classes = np.argmax(y_pred, axis=1)
# Convert true labels to binary classes
y_true = np.argmax(np.array(y_test_06), axis=1)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

#confusion_matrix_list.append(conf_matrix)
total_sum = np.sum(conf_matrix)
correct_answer = conf_matrix[0,0] + conf_matrix[1,1] +conf_matrix[2,2]+conf_matrix[3,3]

confusion_matrix_list06.append(conf_matrix)

plt.figure(figsize=(6, 4))
ax = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="cool", 
            xticklabels=['Region A', 'Region B', 'Region C', 'Region D'], 
            yticklabels=['Region A', 'Region B', 'Region C', 'Region D'])


cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12, width=2, length=6, pad=10, direction="out", colors="black", labelcolor="black")
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")

font = {'color': 'black', 'weight': 'bold', 'size': 12}
ax.set_ylabel("Actual", fontdict=font)
ax.set_xlabel("Predicted", fontdict=font)

# Setting tick labels bold
ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold")
ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold")
#ax.tick_params(axis='both', labelsize=12, weight='bold')
for i, text in enumerate(ax.texts):
    text.set_fontsize(12)
for i, text in enumerate(ax.texts):
    text.set_fontweight('bold')
#plt.title("Confusion Matrix"+"_"+"HEADS"+str(N_HEADS)+"_"+"LAYERS"+str(N_LAYERS)+"_"+"MLP"+str(N_DENSE_UNITS)+"_"+"ImageSize"+str(SIZE)
#          +"\n"+"Accuracy"+str(accuracy))
plt.show()
plt.close()


#+++++++++++++++++++++++++++++++++==0.8 um+++++++++++++++++++++++++++++
um08_index = [index for index, value in enumerate(test_pitch) if value == 1]

X_test_08 = [X_test[i] for i in um08_index]
X_test_08 = np.array(X_test_08)

y_test_08 = [y_test[i] for i in um08_index]
y_test_08 = np.array(y_test_08)

# Predict the test set results
y_pred = vit.predict(np.array(X_test_08))
# Convert predictions to binary classes
y_pred_classes = np.argmax(y_pred, axis=1)
# Convert true labels to binary classes
y_true = np.argmax(np.array(y_test_08), axis=1)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

#confusion_matrix_list.append(conf_matrix)
total_sum = np.sum(conf_matrix)
correct_answer = conf_matrix[0,0] + conf_matrix[1,1] +conf_matrix[2,2]+conf_matrix[3,3]

confusion_matrix_list08.append(conf_matrix)

plt.figure(figsize=(6, 4))
ax = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="cool", 
            xticklabels=['Region A', 'Region B', 'Region C', 'Region D'], 
            yticklabels=['Region A', 'Region B', 'Region C', 'Region D'])


cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12, width=2, length=6, pad=10, direction="out", colors="black", labelcolor="black")
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")

font = {'color': 'black', 'weight': 'bold', 'size': 12}
ax.set_ylabel("Actual", fontdict=font)
ax.set_xlabel("Predicted", fontdict=font)

# Setting tick labels bold
ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold")
ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold")
#ax.tick_params(axis='both', labelsize=12, weight='bold')
for i, text in enumerate(ax.texts):
    text.set_fontsize(12)
for i, text in enumerate(ax.texts):
    text.set_fontweight('bold')
#plt.title("Confusion Matrix"+"_"+"HEADS"+str(N_HEADS)+"_"+"LAYERS"+str(N_LAYERS)+"_"+"MLP"+str(N_DENSE_UNITS)+"_"+"ImageSize"+str(SIZE)
#          +"\n"+"Accuracy"+str(accuracy))
plt.show()
plt.close()


#+++++++++++++++++++++++++++++++++==0.9 um+++++++++++++++++++++++++++++
um09_index = [index for index, value in enumerate(test_pitch) if value == 2]

X_test_09 = [X_test[i] for i in um09_index]
X_test_09 = np.array(X_test_09)

y_test_09 = [y_test[i] for i in um09_index]
y_test_09 = np.array(y_test_09)

# Predict the test set results
y_pred = vit.predict(np.array(X_test_09))
# Convert predictions to binary classes
y_pred_classes = np.argmax(y_pred, axis=1)
# Convert true labels to binary classes
y_true = np.argmax(np.array(y_test_09), axis=1)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

#confusion_matrix_list.append(conf_matrix)
total_sum = np.sum(conf_matrix)
correct_answer = conf_matrix[0,0] + conf_matrix[1,1] +conf_matrix[2,2]+conf_matrix[3,3]

confusion_matrix_list09.append(conf_matrix)

plt.figure(figsize=(6, 4))
ax = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="cool", 
            xticklabels=['Region A', 'Region B', 'Region C', 'Region D'], 
            yticklabels=['Region A', 'Region B', 'Region C', 'Region D'])


cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12, width=2, length=6, pad=10, direction="out", colors="black", labelcolor="black")
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")

font = {'color': 'black', 'weight': 'bold', 'size': 12}
ax.set_ylabel("Actual", fontdict=font)
ax.set_xlabel("Predicted", fontdict=font)

# Setting tick labels bold
ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold")
ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold")
#ax.tick_params(axis='both', labelsize=12, weight='bold')
for i, text in enumerate(ax.texts):
    text.set_fontsize(12)
for i, text in enumerate(ax.texts):
    text.set_fontweight('bold')
#plt.title("Confusion Matrix"+"_"+"HEADS"+str(N_HEADS)+"_"+"LAYERS"+str(N_LAYERS)+"_"+"MLP"+str(N_DENSE_UNITS)+"_"+"ImageSize"+str(SIZE)
#          +"\n"+"Accuracy"+str(accuracy))
plt.show()
plt.close()


#+++++++++++++++++++++++++++++++++==1.2 um+++++++++++++++++++++++++++++
um12_index = [index for index, value in enumerate(test_pitch) if value == 3]

X_test_12 = [X_test[i] for i in um12_index]
X_test_12 = np.array(X_test_12)

y_test_12 = [y_test[i] for i in um12_index]
y_test_12 = np.array(y_test_12)

# Predict the test set results
y_pred = vit.predict(np.array(X_test_12))
# Convert predictions to binary classes
y_pred_classes = np.argmax(y_pred, axis=1)
# Convert true labels to binary classes
y_true = np.argmax(np.array(y_test_12), axis=1)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

#confusion_matrix_list.append(conf_matrix)
total_sum = np.sum(conf_matrix)
correct_answer = conf_matrix[0,0] + conf_matrix[1,1] +conf_matrix[2,2]+conf_matrix[3,3]

confusion_matrix_list12.append(conf_matrix)

plt.figure(figsize=(6, 4))
ax = sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="cool", 
            xticklabels=['Region A', 'Region B', 'Region C', 'Region D'], 
            yticklabels=['Region A', 'Region B', 'Region C', 'Region D'])


cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=12, width=2, length=6, pad=10, direction="out", colors="black", labelcolor="black")
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")

font = {'color': 'black', 'weight': 'bold', 'size': 12}
ax.set_ylabel("Actual", fontdict=font)
ax.set_xlabel("Predicted", fontdict=font)

# Setting tick labels bold
ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold")
ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold")
#ax.tick_params(axis='both', labelsize=12, weight='bold')
for i, text in enumerate(ax.texts):
    text.set_fontsize(12)
for i, text in enumerate(ax.texts):
    text.set_fontweight('bold')
#plt.title("Confusion Matrix"+"_"+"HEADS"+str(N_HEADS)+"_"+"LAYERS"+str(N_LAYERS)+"_"+"MLP"+str(N_DENSE_UNITS)+"_"+"ImageSize"+str(SIZE)
#          +"\n"+"Accuracy"+str(accuracy))
plt.show()
plt.close()


df_results = pd.DataFrame()
df_results['training_loss_list'] = training_loss_list
df_results['validation_loss_list'] = validation_loss_list 
df_results['training_accuracy_list'] = training_accuracy_list
df_results['validation_accuracy_list'] = validation_accuracy_list
df_results['confusion_matrix_list'] = confusion_matrix_list 
df_results['confusion_matrix_list06'] = confusion_matrix_list06
df_results['confusion_matrix_list08'] = confusion_matrix_list08
df_results['confusion_matrix_list09'] = confusion_matrix_list09
df_results['confusion_matrix_list12'] = confusion_matrix_list12    
df_results['accuracy_list'] = accuracy_list 
df_results['time_list'] = time_list