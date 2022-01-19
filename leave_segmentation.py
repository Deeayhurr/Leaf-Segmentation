# import the necessary packages
import glob
import numpy as np
import cv2
import os
from math import sqrt


def save_image(save_path,image, name,):
    """Function for saving the image"""
    name = os.path.basename(name)
    cv2.imwrite(save_path + '/' + name, image)

def create_mask_and_box(path):
    """Fuction to create mask and bounding box"""
      # load the image 
    image = cv2.imread(path,cv2.IMREAD_COLOR)

    # remove all colors except of particular shades of green
    hsv = cv2.cvtColor(image , cv2.COLOR_BGR2HSV)
    lower_green = np.array ([30 , 30, 10])
    upper_green = np.array ([110, 255 , 130])
    mask = cv2.inRange(hsv , lower_green , upper_green)

    #denoise blur image
    mask = cv2.medianBlur(mask, 7)

    #threshold the image
    ret,thresh = cv2.threshold(mask, 127, 255, 0, cv2.THRESH_BINARY)

    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # perform a series of erosions and dilations
    closed_eroded= cv2.erode(closed, None, iterations = 4)
    closed_eroded= cv2.dilate(closed_eroded, None, iterations = 4)
    
    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    countour_image = cv2.findContours(closed_eroded.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    countour_image  = countour_image[0]
    if len(countour_image) == 1:
      c = sorted(countour_image, key = cv2.contourArea, reverse = True)[0]
    else:
      min=image.shape[0]
      for i in range(len(countour_image)):
          if cv2.contourArea(countour_image[i]) > 500: 
              Moments = cv2.moments(countour_image[i])
              cont_X = int(Moments["m10"] / Moments["m00"])
              cont_Y = int(Moments["m01"] / Moments["m00"])
              Z = sqrt((cont_X - image.shape[1]/2)**2 + (cont_Y - image.shape[0]/2)**2)
              if min > Z: 
                  min=Z
                  c = countour_image[i]
      


    rect_image = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect_image)
    box = np.int0(box)


    # draw a bounding box arounded the detected leaf and display the
    # image
    box_image = image.copy()
    cv2.drawContours(box_image, [box], -1, (0, 255, 0), 3)
    return box_image,mask
 
          


def read_write_path():
    """ Function for creating and saving masks and bounding boxes for each leaf image"""
    read_leaf_path ="leaves_testing_set_1/color_images"
    write_leaf_masks_path = 'solution/masks'
    write_leaf_box_path = 'solution/boxes'

    files = [f for f in glob.glob(read_leaf_path + "**/*.png", recursive=True)] 
    for f in files:
        box_image,mask_image = create_mask_and_box(f)
        save_image(write_leaf_masks_path,mask_image, f)
        save_image(write_leaf_box_path,box_image, f.replace('rgb', 'box'))
        name = os.path.basename(f)
    
    print("All files saved")
    cv2.waitKey()



    

def check_result():
    """Function for chcecking if our predicted masks is similar to sample masks"""
    print("************** checking masks *****************")

    sample_mask_image_path= "leaves_testing_set_1/ground_truth"
    my_leaf_masks_path = "solution/masks"

    masks_files = [f for f in glob.glob(sample_mask_image_path + "**/*.png", recursive=True)] 
    mean_IoU = 0
    mean_Dice = 0
    min_IoU = 1
    max_IoU = 0
    min_Dice = 1
    max_Dice = 0
    results = []

    for f in masks_files:
        name = os.path.basename(f)
        my_mask = cv2.imread(my_leaf_masks_path + '/' + name)[:, :, 0]
        sample_mask = cv2.imread(sample_mask_image_path + '/' + name)[:, :, 0]

        predicted_mask_binary = my_mask.flatten().astype(np.bool)
        true_mask_binary = sample_mask.flatten().astype(np.bool)

        overlap = np.logical_and(true_mask_binary, predicted_mask_binary)
        union = np.logical_or(true_mask_binary, predicted_mask_binary)
        IOU = overlap.sum() / float(union.sum())

        Dice = (2.0 * overlap.sum()) / (
            np.sum(predicted_mask_binary) + np.sum(true_mask_binary)
        )
        results.append((name, IOU, Dice))
        #print(name + ' :  IoU: ' + str(IOU) + '   Dice: ' + str(Dice))
        mean_IoU += IOU
        mean_Dice += Dice
        if IOU > max_IoU:
            max_IoU = IOU
            nameMaxIoU = os.path.basename(f)
        if IOU < min_IoU:
            min_IoU = IOU
            nameMinIoU = os.path.basename(f)
        if Dice > max_Dice:
            max_Dice = Dice
            nameMaxDice = os.path.basename(f)
        if Dice < min_Dice:
            min_Dice = Dice
            nameMinDice = os.path.basename(f)

    mean_IoU /= len(masks_files)
    mean_Dice /= len(masks_files)
    print("\n ********** Jaccard index *************** \n")
    print("Mean IoU: " + str(mean_IoU))
    print("Min IoU: " + nameMinIoU + ' - ' + str(min_IoU))
    print("Max IoU: " + nameMaxIoU + ' - ' + str(max_IoU))
    print("\n ********** Dice coefficient *************** \n")
    print("Mean Dice: " + str(mean_Dice))
    print("Min Dice: " + nameMinDice + ' - ' + str(min_Dice))
    print("Max Dice: " + nameMaxDice + ' - ' + str(max_Dice) + "\n")
    with open('results.txt', 'w') as f:
        f.write("\n ************** checking masks ***************** \n ")
        f.write("	Filename: 	      IoU per leaf score:         IoU score:			Dice per leaf score:				Dice score:\n")
        for item in results:
            f.write("%s\n" % str(item))
        f.write("\n%s\n\n" % "\n ********** Jaccard index *************** \n")
        f.write("%s" % "Mean IoU: " + str(mean_IoU) + "\n")
        f.write("%s" % "Min IoU: " + nameMinIoU + ' - ' + str(min_IoU) + "\n")
        f.write("%s" % "Max IoU: " + nameMaxIoU + ' - ' + str(max_IoU) + "\n")
        f.write("\n%s\n\n" % "\n ********** Dice coefficient *************** \n")
        f.write("%s" % "Mean Dice: " + str(mean_Dice) + "\n")
        f.write("%s" % "Min Dice: " + nameMinDice + ' - ' + str(min_Dice) + "\n")
        f.write("%s" % "Max Dice: " + nameMaxDice + ' - ' + str(max_Dice) + "\n")
    cv2.waitKey()



def main():
    print("Computer Vision project 1 - Leaf")

    #Main code:
    read_write_path()
    check_result()
    print("Thank you. Goodbye")


main()