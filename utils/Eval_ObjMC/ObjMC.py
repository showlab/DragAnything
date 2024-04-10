import math
import json
import os 
from numpy import *

def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

gt_json = "./data/VIPSeg_Test/trajectories"
prediction_json = "./Output/prediction_trajectories"

gt_list = os.listdir(gt_json)
pred_list = os.listdir(prediction_json)

json_list = []
for i in gt_list:
    if i in pred_list:
        json_list.append(i)
        

ED_list = []
for json_one in json_list:
    with open(os.path.join(gt_json,json_one), 'r') as json_file:
        trajectory_gt = json.load(json_file)
        
    with open(os.path.join(prediction_json,json_one), 'r') as json_file:
        trajectory_pred = json.load(json_file)
        
    for index in trajectory_gt:
#         print(index)
        gt_points = trajectory_gt[index]
        pred_points = trajectory_pred[index]  
        
        for point1,point2 in zip(gt_points,pred_points):
#             print(point1,point2)
            ED = euclidean_distance(point1,point2)
            ED_list.append(ED)
            
#         break
#     break
print("mean euclidean distance",mean(ED_list))
