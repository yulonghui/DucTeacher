import json
import random
import os
import numpy as np

####### the id of images from the same domain included in the same key in res_id
res_id = {}
avg_score = {}
class_dist = {}

avg_score_num = {}
estimated_distribution = {}
with open('./cache/data/haitian/annotations/instance_unlabel_0_100000.json','r',encoding='utf8')as fp:
    unlabel_data = json.load(fp)

set_location = set()
set_period = set()
set_weather = set()

for image in unlabel_data:
    set_location.add(image['location'])
    set_period.add(image['period'])
    set_weather.add(image['weather'])

for item in set_period:
    for item1 in set_location:
        for item2 in set_weather:
            res_id[item][item1][item2] = []
            avg_score[item][item1][item2] = []
            class_dist[item][item1][item2] = [0, 0, 0, 0, 0, 0]
            avg_score_num[item][item1][item2] = 0
            estimated_distribution[item][item1][item2] = [0, 0, 0, 0, 0, 0]

for data in unlabel_datap['images']:
    res_id[data['period']][data['location']][data['weather']].append(data['id'])


####### after get image ids for each domain, we compute the similarity and estimated class distribution for each domain

with open('./output/coco_instances_results.json','r',encoding='utf8')as fp:
    res_data = json.load(fp)
    for item in set_period:
        for item1 in set_location:
            for item2 in set_weather:
                for res in res_data:
                    if res['image_id'] in res_id[item][item1][item2]:
                        avg_score[item][item1][item2].append(res['score'])      # score append in list , then average 
                        class_dist[item][item1][item2][res['category_id']] += 1 # predicted class number for this domain +1

for item in set_period:
    for item1 in set_location:
        for item2 in set_weather:
            ### the domain similarity computed for each domain, then we can selcet high domain similarity to learn first.
            avg_score_num[item][item1][item2] = sum(avg_score[item][item1][item2])/len(avg_score[item][item1][item2])

####### get the labeled distribution for estimation
labeld_dist = [0, 0, 0, 0, 0, 0]
with open('./cache/data/haitian/annotations/instance_train.json','r',encoding='utf8')as fp:
    labeld_data = json.load(fp)
    for label in labeld_data['annotations']:
        labeld_dist[label['category_id']] += 1

for item in set_period:
    for item1 in set_location:
        for item2 in set_weather:
            estimated_distribution = np.array(class_dist[item][item1][item2]) / np.array(class_dist['daytime']['citystreet']['clear']) * np.array(labeld_dist)/np.sum(np.array(labeld_dist))


print(avg_score_num)            # the domain similarity
print(estimated_distribution)   # the estimated class distribution