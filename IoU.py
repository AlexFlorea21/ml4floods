import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.metrics import MeanIoU
import torch


def calculate_iou(confusions, labels):
    """
    Caculate IoU for a list of confusion matrices 
    
    Args:
        confusions: List with shape (batch_size, len(labels), len(labels))
        labels: List of class names
        
        returns: dictionary of class names and iou scores for that class (summed across whole matrix list)
    """
    conf_matrix = np.array(confusions)
    conf_matrix = np.sum(confusions, axis=0)
    true_positive = np.diag(conf_matrix) + 1e-6
    # true_positive = np.diag(conf_matrix)
    false_negative = np.sum(conf_matrix, 0) - true_positive
    false_positive = np.sum(conf_matrix, 1) - true_positive
    iou = true_positive / (true_positive + false_positive + false_negative)
    
    iou_dict = {}
    for i, l in enumerate(labels):
        iou_dict[l] = iou[i]
    return iou_dict


def calculate_recall(confusions, labels):
    confusions = np.array(confusions)
    conf_matrix = np.sum(confusions, axis=0)
    true_positive = np.diag(conf_matrix) + 1e-6
    false_negative = np.sum(conf_matrix, 0) - true_positive
    recall = true_positive / (true_positive + false_negative  + 1e-6)

    recall_dict = {}
    for i, l in enumerate(labels):
        recall_dict[l] = recall[i]
    return recall_dict



image_folder_dir = r'/mmfs1/scratch/hpc/00/zhangz65/train/Worldfloods/Code/model/1000/prediction/' #This one needs to be changed into the prediction folder
image_gt_folder_dir = r'/mmfs1/scratch/hpc/00/zhangz65/train/Worldfloods/Data/test/gt_mask/' #This one needs to be changed into the ground truth folder called gt_mask that I sent you
contents = os.listdir(image_folder_dir)
labels = ["land", "water", "cloud"]
contents.sort()
iou1 = []
iou2 = []
iou3 = []
iou4 = []
mets = []
iou_score_list = []
for each in contents: #Loop for the folders
  print("Starting {} images".format(each))
  image_dir = image_folder_dir + each
  image_gt_dir = image_gt_folder_dir + each
  print(image_dir)
  print(image_gt_dir)


  
  y_pred=cv2.imread(image_dir)
  image_gt = cv2.imread(image_gt_dir)


#   plt.figure()
#   plt.imshow(cv2.cvtColor(y_pred, cv2.COLOR_BGR2RGB))
#   plt.axis('off')
#   plt.show()


  print()

  y_pred_ = y_pred[:,:,0]
  image_gt_ = image_gt[:,:,0]
  print(y_pred_.shape)
  print(image_gt_.shape)

  # 0-land, 1-water, 2-Cloud, 3-Invalid 
  y_pred_ = np.where(y_pred_ == 120, 0, y_pred_)
  y_pred_ = np.where(y_pred_ == 141, 1, y_pred_)
  y_pred_ = np.where(y_pred_ == 36, 2, y_pred_)
  y_pred_ = np.where(y_pred_ == 84, 3, y_pred_)

  print(np.unique(y_pred_))

 

  image_gt_ = np.where(image_gt_ == 120, 0, image_gt_)
  image_gt_ = np.where(image_gt_ == 141, 1, image_gt_)
  image_gt_ = np.where(image_gt_ == 36, 2, image_gt_)
  image_gt_ = np.where(image_gt_ == 84, 3, image_gt_)

 
  print(np.unique(image_gt_))

  # print(f'prediction_pixel {c}')
  # print(f'image_gt_ {d}')

    #Using built in keras function for IoU
  
  num_classes = 4
  IOU = MeanIoU(num_classes=num_classes)  
  IOU.update_state(y_pred_, image_gt_)
 
  print("Mean IoU  = ", IOU.result().numpy())  #Should be same as the one from sklearn with average='macro'


  #To calculate I0U for each class...
  values = np.array(IOU.get_weights()).reshape(num_classes, num_classes)

  print(values)
  values= values[0:3,0:3]
  values = values.astype(int).tolist()
  mets.append(values)




mets = np.array(mets)

original_method = calculate_iou(mets,labels)
print(original_method)

recall_water = calculate_recall(mets,labels)
print(recall_water)


