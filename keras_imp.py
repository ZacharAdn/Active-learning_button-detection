# import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import os

data_dir = 'data/train'
objects_list = []
annotate_file = open('annotate.txt', 'w')

for r, d, files in os.walk(data_dir):
    for i,f in enumerate(files):
        if '.xml' in f:
            file_path = os.path.join(data_dir, f)
            xml_file = open(file_path)

            image_file = file_path[:-3] + 'png'
            image = plt.imread(image_file)
            fig = plt.figure()

            for row in xml_file:
                if 'xmin' in row:
                    xmin = int(row.split('>')[1].split('<')[0])
                if 'ymin' in row:
                    ymin = int(row.split('>')[1].split('<')[0])
                if 'xmax' in row:
                    xmax = int(row.split('>')[1].split('<')[0])
                if 'ymax' in row:
                    ymax = int(row.split('>')[1].split('<')[0])

                    # add axes to the image
                    ax = fig.add_axes([0, 0, 1, 1])
                    width = xmax - xmin
                    height = ymax - ymin

                    # assign different color to different classes of objects
                    edgecolor = 'r'
                    ax.annotate('', xy=(xmax - 40, ymin + 20))

                    # add bounding boxes to the image
                    rect = patches.Rectangle((xmin, ymin), width, height, edgecolor=edgecolor, facecolor='none')

                    ax.add_patch(rect)

                    ann_object = f'{os.path.abspath(image_file)},{xmin},{ymin},{xmax},{ymax},button\n'
                    objects_list.append(ann_object)
                    annotate_file.write(ann_object)

            # plt.imshow(image)
            # plt.show()


