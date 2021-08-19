"""
Visualization of different thresholding functions in opencv
"""
import argparse
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True)
args = parser.parse_args()

image = cv2.imread(args.image, 0)
assert image is not None, "Failed to read image: {}".format(args.image)

# resize if the image is too big
if image.shape[0] * image.shape[1] > 1.2e6:
    aspect_ratio = image.shape[1] / image.shape[0]
    image = cv2.resize(
        image, (int(1024 * aspect_ratio), 1024), interpolation=cv2.INTER_CUBIC
    )

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
ax.imshow(image, cmap="gray")
ax.set_title("Image Thresholding")

# Define slider and axes and set initial visibility
# threshold value
threshold_axis = plt.axes([0.25, 0.15, 0.65, 0.03])
threshold_slider = Slider(threshold_axis, "thresh val", 0, 255, valinit=127, valstep=1)
threshold_axis.set_visible(False)
# block size (adaptive)
block_size_axis = plt.axes([0.25, 0.1, 0.65, 0.03])
block_size = int(max(image.shape[0], image.shape[1]) / 4.0)
if block_size % 2 == 0:
    block_size += 1
block_size_slider = Slider(
    block_size_axis, "block size", 3, block_size, valinit=11, valstep=2
)
block_size_axis.set_visible(False)
# constant (adaptive)
constant_axis = plt.axes([0.25, 0.05, 0.65, 0.03])
constant_slider = Slider(constant_axis, "constant", 0, 50, valinit=2, valstep=1)
constant_axis.set_visible(False)

method = "original"


def update(val=None):
    if method == "original":
        th = image
    elif method == "binary":
        _, th = cv2.threshold(image, threshold_slider.val, 255, cv2.THRESH_BINARY)
    elif method == "adaptive mean":
        th = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            int(block_size_slider.val),
            int(constant_slider.val),
        )
    elif method == "adaptive gaussian":
        th = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            int(block_size_slider.val),
            int(constant_slider.val),
        )
    elif method == "adaptive gaussian AND otsu":
        th_1 = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            int(block_size_slider.val),
            int(constant_slider.val),
        )
        _, th_2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = cv2.bitwise_and(th_1, th_2)
    elif method == "adaptive gaussian OR otsu":
        th_1 = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            int(block_size_slider.val),
            int(constant_slider.val),
        )
        _, th_2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = cv2.bitwise_or(th_1, th_2)
    else:  # otsu
        _, th = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ax.imshow(th, cmap="gray")
    plt.draw()


threshold_slider.on_changed(update)
block_size_slider.on_changed(update)
constant_slider.on_changed(update)


def set_axes_visibility(label):
    global method
    method = label
    if label == "binary":
        threshold_axis.set_visible(True)
        block_size_axis.set_visible(False)
        constant_axis.set_visible(False)
    elif label in [
        "adaptive mean",
        "adaptive gaussian",
        "adaptive gaussian AND otsu",
        "adaptive gaussian OR otsu",
    ]:
        threshold_axis.set_visible(False)
        block_size_axis.set_visible(True)
        constant_axis.set_visible(True)
    elif label in ["otsu", "original"]:
        threshold_axis.set_visible(False)
        block_size_axis.set_visible(False)
        constant_axis.set_visible(False)
    update()


rax = plt.axes([0.025, 0.5, 0.15, 0.15])
radio = RadioButtons(
    rax,
    (
        "original",
        "binary",
        "otsu",
        "adaptive mean",
        "adaptive gaussian",
        "adaptive gaussian AND otsu",
        "adaptive gaussian OR otsu",
    ),
)
radio.on_clicked(set_axes_visibility)
plt.show()
