import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import os
from glob import glob
import glob2
import pydicom as pm
import imagej
import numpy as np
import simple_slice_viewer as ssv
from simple_slice_viewer import controller
from simple_slice_viewer.controller_base import ControllerBase


#pth do dcms
path = '/Users/sandrawieczorek/Library/CloudStorage/OneDrive-PolitechnikaŚląska/POMwJO/S8010'

#anonymize patient data

# show images
def Show(array):
    fig = plt.figure()
    for i in range(len(array)):
        plt.subplot(5, 5, i + 1)
        plt.imshow(array[i], cmap='gray')
    plt.show()


# read files
reader = sitk.ImageSeriesReader()
dcm_Names = reader.GetGDCMSeriesFileNames(path)
reader.SetFileNames((dcm_Names))
image = reader.Execute()


#display
o = ssv.display(image = image)
seed = o.slice_controller.mouse.get_image_data(o.view.image_view.image_view.get_position())['position_index']
lower_thresh = (o.slice_controller.mouse.get_image_data(o.view.image_view.image_view.get_position())['image_value'])
lower_thresh = float(lower_thresh) - 200
upper_thresh = (o.slice_controller.mouse.get_image_data(o.view.image_view.image_view.get_position())['image_value'])
upper_thresh = float(upper_thresh) + 200


print(seed)


size = image.GetSize()
image_array = sitk.GetArrayViewFromImage(image)

print("Image size:", size[0], size[1], size[2])


plt.hist(sitk.GetArrayViewFromImage(image).flatten(), bins = 200)
plt.show()


########    START SEGMENTATION     ########
#threshold filters - divide from bckground

threshold_filters = {
    "Otsu": sitk.OtsuThresholdImageFilter(),
    "Triangle": sitk.TriangleThresholdImageFilter(),
    "Huang": sitk.HuangThresholdImageFilter(),
    "MaxEntropy": sitk.MaximumEntropyThresholdImageFilter(),
}

filter_selection = "Huang"

try:
    thresh_filter = threshold_filters[filter_selection]
    thresh_filter.SetInsideValue(0)
    thresh_filter.SetOutsideValue(1)
    thresh_img = thresh_filter.Execute(image)
    thresh_value = thresh_filter.GetThreshold()
except KeyError:
    thresh_value = 120
    thresh_img = image > thresh_value

print("Threshold used: " + str(thresh_value))
threshed_array = sitk.GetArrayViewFromImage(thresh_img)

# opening & closing
cleaned_thresh_img = sitk.BinaryOpeningByReconstruction(thresh_img, [10, 10, 10])
cleaned_thresh_img = sitk.BinaryClosingByReconstruction(cleaned_thresh_img, [5, 5, 5])

# ssv.display(cleaned_thresh_img)
#### FAST-MARCHING SEGENTATION
# seed = (355, 300, 11)
# feature_img = sitk.GradientMagnitudeRecursiveGaussian(image, sigma=1.5)
# speed_img = sitk.BoundedReciprocal(feature_img)
# fm_filter = sitk.FastMarchingBaseImageFilter()
# fm_filter.SetTrialPoints([seed])
# fm_filter.SetStoppingValue(1000)
# fm_img = fm_filter.Execute(speed_img)
#
# ssv.display(fm_img)
#### REGION GROWING SEGM.
# seed = (364, 273, 11)
# seed = (355, 300, 11)

seg = sitk.Image(image.GetSize(), sitk.sitkUInt8)
seg.CopyInformation(image)
# seg[seed] = 1

# vector = ([3, 3, 3], sitk.sitkVectorUInt32)
seg = cleaned_thresh_img
#myshow(sitk.LabelOverlay(img_T1_255, seg), "Initial Seed")



seg = sitk.ConnectedThreshold(image, seedList=[seed], lower=lower_thresh, upper=upper_thresh)

# seg = sitk.ConfidenceConnected(image, seedList=[seed],
#                                  numberOfIterations=100,
#                                  multiplier=.5,
#                                  initialNeighborhoodRadius=5,
#                                  replaceValue=1)
# ssv.display(seg)


#
final = sitk.LabelOverlay(image, seg)

pink = [255, 105, 180]
green = [0, 255, 0]
gold = [255, 215, 0]

combined = sitk.LabelOverlay(
    image=image,
    labelImage=seg,
    opacity=0.5,
    backgroundValue=40,
)


Show(sitk.GetArrayViewFromImage(seg))
# Show(sitk.GetArrayViewFromImage(combined))
# ssv.display(seg)

