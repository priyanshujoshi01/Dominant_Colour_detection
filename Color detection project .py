#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install opencv-python 

#it provides tools and functions for various tasks related to computer vision, image processing, and machine learning
#used for tasks like image and video analysis, object detection, face recognition, and more.
# In[2]:


pip install --upgrade numpy


# In[7]:


pip list      #list all of the Python packages that are installed in our current environment


# In[1]:


import cv2               # it is our Open cv -python library previsously it is showing error 


# In[9]:


import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans 


# In[5]:


# Load the image data 
image_path = 'C:/Users/Lenovo/Pictures/Saved Pictures/logo1.jpg' 
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB color space

Duing above code I faced syntax error (unicode error) 'unicodeescape' codec can't decode bytes in position 2-3: truncated \UXXXXXXXX escape"
reason :-a file path with a backslash \ character in a string, and Python interprets it as an escape sequence.
solution - resolve in a 3 ways - use raw string (r') or  \\ or / in file path 

# In[6]:


# Reshape the image to a 2D array of pixels
pixels = image.reshape((-1, 3))


# In[7]:


# here I Define the number of clusters (colors) we want to detect
num_clusters = 5


# In[10]:


# Apply k-means clustering to find the dominant colors
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(pixels)

 in above code I faced -name 'KMeans' is not defined error because KMeans class from the scikit-learn library and I forgot to import skitlearn . 
# In[11]:


# Get the RGB values of the cluster centers (dominant colors)
dominant_colors = kmeans.cluster_centers_.astype(int)


# In[12]:


# Count the number of pixels assigned to each cluster
pixel_counts = Counter(kmeans.labels_)


# In[13]:


# Find the most dominant color by sorting the clusters by pixel count
most_dominant_color = dominant_colors[np.argmax(list(pixel_counts.values()))]


# In[14]:


# Display the original image
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')


# In[15]:


# Create a swatch of dominant colors
color_swatch = np.zeros((50, 50, 3), dtype=np.uint8)
color_swatch[:, :, :] = most_dominant_color
plt.subplot(1, 2, 2)
plt.imshow(color_swatch)
plt.title('Dominant Color')
plt.tight_layout()
plt.show()


# In[16]:


plt.tight_layout()
plt.show()


# In[17]:


# Print the RGB values of the dominant color
print(f"Dominant Color (RGB): {most_dominant_color}")

so we can use our Dominant colour insight in a diff way -
1- in graphic design and branding, we can analyze the dominant colors in images or logos associated with a brand. This analysis can help ensure consistency in color schemes across marketing materials and products.
2-In the fashion industry, understanding color trends and the dominant colors in clothing can guide decisions on what colors to produce or stock for a particular season.
3- we can use color information to study changes in ecosystems, identify pollution levels, or track the health of plants based on leaf color. 
and many more use cases .