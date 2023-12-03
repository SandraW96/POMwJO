import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import os
from glob import glob
import glob2
import pydicom as pm
import imagej
import simple_slice_viewer as ssv


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
#ssv.display(image = image)

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
cleaned_thresh_img = sitk.BinaryClosingByReconstruction(cleaned_thresh_img, [15, 15, 15])

seed = (358, 247, 12)
feature_img = sitk.GradientMagnitudeRecursiveGaussian(image, sigma=.5)
speed_img = sitk.BoundedReciprocal(feature_img)
fm_filter = sitk.FastMarchingBaseImageFilter()
fm_filter.SetTrialPoints([seed])
fm_filter.SetStoppingValue(800)
fm_img = fm_filter.Execute(speed_img)

ssv.display(fm_img)



