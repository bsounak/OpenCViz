"""
Visualization of different thresholding functions in opencv
"""
import argparse
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
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
ax.set_title("Image Thresholding")

# Define slider and axes and set initial visibility
# threshold value
th_ax = plt.axes([0.25, 0.15, 0.65, 0.03])
th_sl = Slider(th_ax, 'thresh val', 0, 255, valinit=127, valstep=1)
th_ax.set_visible(False)
# block size (adaptive)
bs_ax = plt.axes([0.25, 0.1, 0.65, 0.03])
bs_sl = Slider(bs_ax, 'block size', 3, 49, valinit=11, valstep=2)
bs_ax.set_visible(False)
# constant (adaptive)
c_ax = plt.axes([0.25, 0.05, 0.65, 0.03])
c_sl = Slider(c_ax, 'constant', 0, 50, valinit=2, valstep=1)
c_ax.set_visible(False)

method = "original"
def update(val=None):
    if method == "original":
        th = img
    elif method == "binary":
        _, th = cv2.threshold(
            img, th_sl.val, 255, cv2.THRESH_BINARY)
    elif method == "adaptive mean":
        th = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, int(bs_sl.val), int(c_sl.val))
    elif method == "adaptive gaussian":
        th = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, int(bs_sl.val), int(c_sl.val))
    else:  # otsu
        _, th = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ax.imshow(th, cmap="gray")

th_sl.on_changed(update)
bs_sl.on_changed(update)
c_sl.on_changed(update)

def set_axes_visibility(label):
    global method
    method = label
    if label == "binary":
        th_ax.set_visible(True)
        bs_ax.set_visible(False)
        c_ax.set_visible(False)
    elif label in ["adaptive mean", "adaptive gaussian"]:
        th_ax.set_visible(False)
        bs_ax.set_visible(True)
        c_ax.set_visible(True)
    elif label in ["otsu", "original"]:
        th_ax.set_visible(False)
        bs_ax.set_visible(False)
        c_ax.set_visible(False)

    fig.canvas.draw_idle()
    update()

rax = plt.axes([0.025, 0.5, 0.15, 0.15])
radio = RadioButtons(
    rax, ('original', 'binary', 'otsu', 'adaptive mean', 'adaptive gaussian'))
radio.on_clicked(set_axes_visibility)
plt.show()
