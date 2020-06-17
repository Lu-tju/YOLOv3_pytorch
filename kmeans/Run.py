import glob
import xml.etree.ElementTree as ET

import numpy as np

from kmeans import kmeans, avg_iou

'''K-mean聚类算法'''

ANNOTATIONS_PATH = "D:\PyCharm Professional\projects\yolov3_older\data\Annotations"
CLUSTERS = 6


def load_dataset(path):
    dataset = []
    for xml_file in glob.glob("{}/*xml".format(path)):
        tree = ET.parse(xml_file)

        height = int(tree.findtext("./size/height"))
        width = int(tree.findtext("./size/width"))

        for obj in tree.iter("object"):
            xmin = int(obj.findtext("bndbox/xmin")) / width
            ymin = int(obj.findtext("bndbox/ymin")) / height
            xmax = int(obj.findtext("bndbox/xmax")) / width
            ymax = int(obj.findtext("bndbox/ymax")) / height

            dataset.append([xmax - xmin, ymax - ymin])

    return np.array(dataset)


data = load_dataset(ANNOTATIONS_PATH)
out, indix = kmeans(data, k=CLUSTERS)
print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
print("Boxes:\n {}".format(out))

ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
print("Ratios:\n {}".format(sorted(ratios)))

import matplotlib.pyplot as plt
plt.scatter(data[:, 0], data[:, 1], c=indix.T, cmap=plt.cm.Spectral)
