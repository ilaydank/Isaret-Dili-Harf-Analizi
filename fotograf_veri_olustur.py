import cv2
from glob import glob
import numpy as np
import random
from sklearn.utils import shuffle
import pickle
import os

def pickle_images_labels():
	images_labels = []
	images = glob("images/*/*.jpg")
	images.sort()
	for image in images:
		print(image)
		label = image[image.find(os.sep)+1: image.rfind(os.sep)]
		img = cv2.imread(image, 0)
		images_labels.append((np.array(img, dtype=np.uint8), int(label)))
	return images_labels

images_labels = pickle_images_labels()
images_labels = shuffle(shuffle(shuffle(shuffle(images_labels))))
images, labels = zip(*images_labels)
print("images_labels uzunluğu:", len(images_labels))

train_images = images[:int(5/6*len(images))]
print("train_images uzunluğu:", len(train_images))
with open("veriler/train_images", "wb") as f:
	pickle.dump(train_images, f)
del train_images

train_labels = labels[:int(5/6*len(labels))]
print("train_labels uzunluğu:", len(train_labels))
with open("veriler/train_labels", "wb") as f:
	pickle.dump(train_labels, f)
del train_labels

test_images = images[int(5/6*len(images)):int(11/12*len(images))]
print("test_images uzunluğu:", len(test_images))
with open("veriler/test_images", "wb") as f:
	pickle.dump(test_images, f)
del test_images

test_labels = labels[int(5/6*len(labels)):int(11/12*len(images))]
print("test_labels uzunluğu:", len(test_labels))
with open("veriler/test_labels", "wb") as f:
	pickle.dump(test_labels, f)
del test_labels

val_images = images[int(11/12*len(images)):]
print("test_images uzunluğu: ", len(val_images))
with open("veriler/val_images", "wb") as f:
	pickle.dump(val_images, f)
del val_images

val_labels = labels[int(11/12*len(labels)):]
print("val_labels uzunluğu:", len(val_labels))
with open("veriler/val_labels", "wb") as f:
	pickle.dump(val_labels, f)
del val_labels