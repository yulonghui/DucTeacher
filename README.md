# DucTeacher

## Prepare for SODA10M

```shell
DucTeacher/
└── datasets/
    └── haitian/
        ├── train/
        ├── unlabel/
        ├── val/
        └── annotations/
        	├── instances_train.json
            ├── instances_unlabel_0.json
        	└── instances_val.json
```
install detectron2 in ./detectron2
## Training

bash.sh for DucTeacher

################## pre-train stage ##################

get the pre-train model in ./output
```shell
mv 'merge_domain_8/merge_0.json' '/cache/data/haitian/annotations/instance_unlabel_0.json'
python train_net.py --num-gpus 8 --config configs/haitian_supervision/faster_rcnn_R_50_FPN_sup_run1.yaml SOLVER.IMG_PER_BATCH_LABEL 16 SOLVER.IMG_PER_BATCH_UNLABEL 16 SEMISUPNET.PARA_MU 0.1 SEMISUPNET.PARA_T 0.7
```

################## Evaluation on Unlabeled data ##################

get the Evaluation results in ./output/infrence/coco_instances_results.json
```shell
python train_net.py --num-gpus 8 --eval-only --config configs/haitian_supervision/faster_rcnn_R_50_FPN_sup_run1.yaml SOLVER.IMG_PER_BATCH_LABEL 16 SOLVER.IMG_PER_BATCH_UNLABEL 16
```

################## Get 1. Domain Similarity 2. Estimated Class Distribution ##################

input : ./output/infrence/coco_instances_results.json
output : Domain Similarity & Estimated Class Distribution
```shell
python get_domain_similarity_class_distribution.py
```
################## DucTeacher stage ##################

after getting the domain similarity and class distribution, we can train the DucTeacher.
```shell
python DucTeacher_domain_evolve_sh_0_10.py
```