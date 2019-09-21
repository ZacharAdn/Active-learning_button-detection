# import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches

im = '/homes/zahara/PycharmProjects/text_button_detection/data/file.png'
image = plt.imread(im)
plt.imshow(image)
print('image printed')

# fig = plt.figure()
#
# # add axes to the image
# ax = fig.add_axes([0, 0, 1, 1])
#
# # read and plot the image
# image = plt.imread(im)
# plt.imshow(image)
#
# xmin = 11
# xmax = 496
# ymin = 174
# ymax = 517
#
# width = xmax - xmin
# height = ymax - ymin
#
# # assign different color to different classes of objects
# edgecolor = 'r'
# ax.annotate('button', xy=(xmax - 40, ymin + 20))
#
# # add bounding boxes to the image
# rect = patches.Rectangle((xmin, ymin), width, height, edgecolor=edgecolor, facecolor='none')
#
# ax.add_patch(rect)
#
# plt.imshow(image)
'''
# iterating over the image for different objects
for _,row in train[train.image_names == "1.jpg"].iterrows():
    xmin = row.xmin
    xmax = row.xmax
    ymin = row.ymin
    ymax = row.ymax
    
    width = xmax - xmin
    height = ymax - ymin
    
    # assign different color to different classes of objects
    if row.cell_type == 'RBC':
        edgecolor = 'r'
        ax.annotate('RBC', xy=(xmax-40,ymin+20))
    elif row.cell_type == 'WBC':
        edgecolor = 'b'
        ax.annotate('WBC', xy=(xmax-40,ymin+20))
    elif row.cell_type == 'Platelets':
        edgecolor = 'g'
        ax.annotate('Platelets', xy=(xmax-40,ymin+20))
        
    # add bounding boxes to the image
    rect = patches.Rectangle((xmin,ymin), width, height, edgecolor = edgecolor, facecolor = 'none')
    
    ax.add_patch(rect)

'''
