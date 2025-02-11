import numpy as np
from glob import glob
from tqdm import tqdm
import h5py
import nrrd
import nibabel as nib
import pandas as pd
import xlrd
import pdb
import SimpleITK as sitk
from skimage import transform, measure
import os
import pydicom
import matplotlib.pyplot as plt


def load_scan(path):
    temp = [pydicom.dcmread(path + f) for f in os.listdir(path)]
    slices = [t for t in temp if t.Modality == 'CT']
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices


def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.float32)
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float32)
        image = image.astype(np.float32)

    image += np.float32(intercept)

    return np.array(image, dtype=np.float32)


listt = glob('E:/PyCharmProjects/UA-MT-master/data/Pancreas-CT/*/*/PANCREAS_*/')
base_dir = "E:/PyCharmProjects/UA-MT-master/data/TCIA_pancreas_labels-02-05-2017/"

for item in tqdm(listt):
    name = str(item)
    name_id = name[-5:-1]
    print(name, name_id)
    patient_ct = load_scan(name)
    imgs_ct = get_pixels_hu(patient_ct)

    itk_img = sitk.ReadImage(base_dir +'label'+ name_id + '.nii.gz')
    origin = itk_img.GetOrigin()
    direction = itk_img.GetDirection()
    space = itk_img.GetSpacing()
    label = sitk.GetArrayFromImage(itk_img)

    image_gz = sitk.GetImageFromArray(imgs_ct)
    image_gz.SetOrigin(origin)
    image_gz.SetDirection(direction)
    image_gz.SetSpacing(space)
    sitk.WriteImage(image_gz, "E:/PyCharmProjects/UA-MT-master/data/image/" + 'image' + name_id+".nii.gz")

#     plt.figure(figsize=(10, 10))
#     plt.title('CT Slice_10')
#     plt.imshow(imgs_ct[9],cmap='gray')
#     plt.axis('off')
#     plt.show()