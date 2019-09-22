import matplotlib.pyplot as plt
from matplotlib import patches
import os
import sys

sys.path.append('.')

data_dir = 'data/train'
annotate_file = open('annotate.txt', 'w')
plot = False


def add_to_file(image, xmin, ymin, xmax, ymax):
    if plot:
        fig = plt.figure()
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
        plt.imshow(image)
        plt.show()

    ann_object = f'{os.path.abspath(image_file)},{xmin},{ymin},{xmax},{ymax},button\n'
    annotate_file.write(ann_object)


# for r, d, files in os.walk(data_dir):
for i, f in enumerate(os.listdir(data_dir)):
    if '.xml' in f:
        file_path = os.path.join(data_dir, f)
        xml_file = open(file_path)

        image_file = file_path[:-3] + 'png'
        image = plt.imread(image_file)

        done_rectangle = False
        xmin, ymin, xmax, ymax = 0, 0, 0, 0
        for row in xml_file:
            if 'xmin' in row:
                xmin = int(row.split('>')[1].split('<')[0])
            elif 'ymin' in row:
                ymin = int(row.split('>')[1].split('<')[0])
            elif 'xmax' in row:
                xmax = int(row.split('>')[1].split('<')[0])
            elif 'ymax' in row:
                ymax = int(row.split('>')[1].split('<')[0])
                done_rectangle = True

            if done_rectangle:
                add_to_file(image, xmin, ymin, xmax, ymax)
                done_rectangle = False


