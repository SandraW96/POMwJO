import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import os

# show images
def Show(array):
    fig = plt.figure()
    for i in range(len(array)):
        plt.subplot(5, 5, i + 1)
        plt.imshow(array[i], cmap='gray')
    plt.show()


# read files
path = '/Users/sandrawieczorek/Library/CloudStorage/OneDrive-PolitechnikaŚląska/POMwJO/S8010'
reader = sitk.ImageSeriesReader()
dcm_Names = reader.GetGDCMSeriesFileNames(path)
reader.SetFileNames((dcm_Names))
image = reader.Execute()

size = image.GetSize()
image_array = sitk.GetArrayViewFromImage(image)

print("Image size:", size[0], size[1], size[2])

Show(image_array)

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
Show(threshed_array)

# opening & closing
cleaned_thresh_img = sitk.BinaryOpeningByReconstruction(thresh_img, [10, 10, 10])
cleaned_thresh_img = sitk.BinaryClosingByReconstruction(cleaned_thresh_img, [10, 10, 10])

#watershed segmentation
dist_img = sitk.SignedMaurerDistanceMap(
    cleaned_thresh_img != 0,
    insideIsPositive=False,
    squaredDistance=False,
    useImageSpacing=False,
)
radius = 10
# Seeds have a distance of "radius" or more to the object boundary, they are uniquely labelled.
seeds = sitk.ConnectedComponent(dist_img < -radius)
# Relabel the seed objects using consecutive object labels while removing all objects with less than 15 pixels.
seeds = sitk.RelabelComponent(seeds, minimumObjectSize=15)
# Run the watershed segmentation using the distance map and seeds.
ws = sitk.MorphologicalWatershedFromMarkers(dist_img, seeds, markWatershedLine=True)
ws = sitk.Mask(ws, sitk.Cast(cleaned_thresh_img, sitk.sitkUInt8))

plt.figure()
plt.subplot(1, 3, 3)
plt.imshow(threshed_array)
plt.imshow(sitk.LabelOverlay(image, ws))
plt.imshow(sitk.LabelOverlay(image, seeds))
plt.show()