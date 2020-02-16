"""
Visualization of morphological functions in opencv
"""
import argparse
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
import numpy as np
import cv2

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--image", required=True)
args = parser.parse_args()

img = cv2.imread(args.image, 0)
assert img is not None, "Failed to read image: {}".format(args.image)

# resize if the image is too big
if img.shape[0]*img.shape[1] > 1.2E6:
    aspect_ratio = img.shape[1]/img.shape[0]
    img = cv2.resize(
        img, (int(1024*aspect_ratio), 1024),
        interpolation=cv2.INTER_CUBIC)

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
ax.imshow(img, cmap="gray")
ax.set_title("Morphological Transformations")

# Define slider and axes and set initial visibility
# kernel size
ksize = 5
def update_ksize(val=None):
    global ksize
    ksize = int(ksize_sl.val)
    update()

ksize_ax = plt.axes([0.25, 0.15, 0.65, 0.03])
ksize_sl = Slider(
    ksize_ax, 'kernel size', 3, 51, valinit=5, valstep=2)
ksize_sl.on_changed(update_ksize)

# iterations
iterations = 1
def update_iterations(val=None):
    global iterations
    iterations = int(iterations_sl.val)
    update()

iterations_ax = plt.axes([0.25, 0.1, 0.15, 0.03])
iterations_sl = Slider(
    iterations_ax, 'iterations', 1, 5, valinit=1, valstep=1)
iterations_sl.on_changed(update_iterations)

mtype = "ORIGINAL"
def set_mtype(label):
    global mtype
    mtype = label
    update()

mshape = "RECT"
def set_mshape(label):
    global mshape
    mshape = label
    update()

# radio buttons for morphological operation types
rax = plt.axes([0.025, 0.65, 0.15, 0.15])
mtypes = RadioButtons(
    rax, ('ORIGINAL', 'ERODE', 'DILATE', 'OPEN',
          'CLOSE', 'GRADIENT', 'TOPHAT', 'BLACKHAT'))

# radio buttons for morphological operation kernel shapes
rax_2 = plt.axes([0.025, 0.45, 0.15, 0.15])
mshapes = RadioButtons(
    rax_2, ('RECT', 'ELLIPSE', 'CROSS'))

mtypes.on_clicked(set_mtype)
mshapes.on_clicked(set_mshape)

def get_kernel():
    if mshape == "RECT":
        kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (ksize, ksize))
    elif mshape == "ELLIPSE":
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (ksize, ksize))
    elif mshape == "CROSS":
        kernel = cv2.getStructuringElement(
            cv2.MORPH_CROSS, (ksize, ksize))
    return kernel

def update(val=None):
    kernel = get_kernel()
    if mtype == "ORIGINAL":
        morphed = img
    elif mtype == "ERODE":
        morphed = cv2.erode(img, kernel, iterations=iterations)
    elif mtype == "DILATE":
        morphed = cv2.dilate(img, kernel, iterations=iterations)
    elif mtype == "OPEN":
        morphed = cv2.morphologyEx(
            img, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif mtype == "CLOSE":
        morphed = cv2.morphologyEx(
            img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif mtype == "GRADIENT":
        morphed = cv2.morphologyEx(
            img, cv2.MORPH_GRADIENT, kernel, iterations=iterations)
    elif mtype == "TOPHAT":
        morphed = cv2.morphologyEx(
            img, cv2.MORPH_TOPHAT, kernel, iterations=iterations)
    elif mtype == "BLACKHAT":
        morphed = cv2.morphologyEx(
            img, cv2.MORPH_BLACKHAT, kernel, iterations=iterations)

    ax.imshow(morphed, cmap="gray")
    plt.draw()

plt.show()
