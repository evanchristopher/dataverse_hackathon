# Reference: https://alex-vaith.medium.com/save-precious-time-with-image-augmentation-in-object-detection-tasks-2f9111abb851
# Reference: https://imgaug.readthedocs.io/en/latest/index.html
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import cv2
import pandas as pd
import os


# prepare sequentual object of select image augmenters to be applied in
# random order
seq = iaa.Sequential([
    # flip horizontally from left to right the image
    iaa.Fliplr(0.5), 
    # random crops up to 30%
    iaa.Crop(percent=(0, 0.3)), 
    # sometimes apply GaussianBlur 50% of the time up to 0.5 sigma
    iaa.Sometimes(0.5, 
                  iaa.GaussianBlur(sigma=(0,0.5))
                  ), 
    # change contrast slightly
    iaa.LinearContrast((0.95, 1.05)), 
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), 
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # apply affine transformation, specifically rotation. 
    # We limit this to a max of 3 degrees as studies show beyond this 
    # and the bounding box will no longer fit
    iaa.Affine(rotate=(-3,3))
    ], random_order=True)

def augment_image(path: str, 
                  filename: str, 
                  df: pd.DataFrame, 
                  augmentations: int=10) -> tuple:
    """
    Applies augmentations to a supplied image file and bounding boxes. The 
    augmented image data and augmented bounding box coordinates are returned.

    Args:
        - path (str) = directory path to image file.
        - filename (str) = name of image file.
        - df (pd.DataFrame) = dataframe object containing bounding box 
                                coordinates.
        - augmentations (int) = number of times the original image and bounding
                                boxes should be copied and 
                                apply augmentations to. Defaults to 10.
    Returns:
        - (tuple) object containing 2 lists consisting of the augmented image
            data and respective augmented bounding boxes.
    """
    img_bbs = []
    # read image and retrieve image data from OpenCV
    img = cv2.imread(os.path.join(path, filename))

    # retrieve bounding box targets and store bounding boxes for augmentation
    for _, row in df[df.ImageID == filename].iterrows():
        x1 = row.XMin
        y1 = row.YMin
        x2 = row.XMax
        y2 = row.YMax
        img_bbs.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=row.LabelName))
    bbs = BoundingBoxesOnImage(img_bbs, shape=img.shape[:-1])
    # copy file and associated bounding boxes (augmentation) times
    images = [img for _ in range(augmentations)]
    bbss = [bbs for _ in range(augmentations)]
    # generate augmentations
    images, bbss_aug = seq(images=images, bounding_boxes=bbss)
    return images, bbss_aug

def save_augmentation(images: list, 
                      bbss: list, 
                      filename: str, 
                      df: pd.DataFrame, 
                      folder: str) -> pd.DataFrame:
    """
    Appends generated augmented bounding boxes to dataframe and saves 
    augmented file if at least one box is acceptably within frame.

    Args:
        - images (list) = list of augmented image data.
        - bbss (list) = list of augmented bounding boxes.
        - filename (str) = name of original image file.
        - df (pd.DataFrame) = dataframe containing original bounding box
                                coordinates to append new augmented boxes to.
        - folder (str) = directory to save augmented file to if at least 1 
                            augmented bounding box exists within the augmented
                            image.
    Returns:
        - (pd.DataFrame) dataframe object with appended bounding box 
            coordinates.
    """
    for (i, img_aug), bbs_aug in zip(enumerate(images), bbss):
        # set new filename
        file_info = os.path.splitext(filename)
        filename_aug =  f'{file_info[0]}_aug_{i}{file_info[1]}'
        one_box_present = False
        # show augmented image with augmented bounding boxes
        for bbs in bbs_aug:
            # skip bounding box if more than 80 percent is outside the image
            if bbs.compute_out_of_image_fraction(img_aug) > 0.8:
                continue
            # clip any bounding boxes stretching outside image after cropping
            bbs = bbs.clip_out_of_image(img_aug)
            one_box_present = True
            x1_aug = bbs.x1
            y1_aug = bbs.y1
            x2_aug = bbs.x2
            y2_aug = bbs.y2
            # add augmented bounding box to dataframe
            df = df.append(
                pd.DataFrame(data=[[filename_aug, 
                                    bbs.label, 
                                    x1_aug, 
                                    x2_aug, 
                                    y1_aug, 
                                    y2_aug]],
                columns=df.columns.tolist()), 
                ignore_index=True)
        # save augmented image if at least 1 bounding box was saved
        if one_box_present:
            cv2.imwrite(os.path.join(folder, filename_aug), img_aug)
    # return appended dataframe
    return df
