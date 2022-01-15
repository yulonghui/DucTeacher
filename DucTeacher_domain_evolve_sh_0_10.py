import os
import sys
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--init_method', type=str, default=None, help='master address')
    parser.add_argument('--rank', type=int, default=0, help='Index of current task')
    parser.add_argument('--world_size', type=int, default=1, help='Total number of tasks')

    parser.add_argument('--data_url', type=str, default=None, help='s3 path of dataset')
    parser.add_argument('--train_url', type=str, default=None, help='s3 path of outputs')

    parser.add_argument('--dataset', type=str, default="Haitian", help='dataset')
    parser.add_argument('--config', type=str,
                        default="configs/haitian_supervision/faster_rcnn_R_50_FPN_sup_run1.yaml", help='config')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    for domain_i in [0, 1, 4, 7]:   #domain evolving
        os.system("mv ./merge_domain_8/merge_%s.json /cache/data/haitian/annotations/instance_unlabel_0.json" %domain_i )

        print(os.listdir('/cache/data/haitian/annotations'))
        print(os.listdir('/cache/data/haitian/'))
        print(os.listdir('/cache/data/haitian/unlabel/dataset/unlabel'))

        train_data = json.load(open('/cache/data/haitian/annotations/instance_train.json'))
        train_data['annotations'] = []
        train_data['images'] = []

        unlabel_data = json.load(open('/cache/data/haitian/annotations/instance_unlabel_0.json')
        train_data['images'] += unlabel_data.copy()

        with open('/cache/data/haitian/annotations/instance_train_unlabel.json', 'w') as f:
            json.dump(train_data, f)
        print(os.listdir('/cache/data/haitian/annotations'))

        # install enviroment
        os.system("pip install torch==1.7.0 torchvision==0.8.1")
        os.system('pip install black==21.4b2')
        os.system("cd /home/ma-user/modelarts/user-job-dir/unbiased-teacher/detectron2; pip install -e .")
        os.system("pip install pyyaml==5.1")

        # train_model
        os.system('cd unbiased-teacher; '
                    'rm -rf datasets;'
                'ln -s /cache/data datasets')       # soft-link

        os.system('mv datasets/haitian/annotations/instance_train_unlabel.json datasets/haitian/annotations/instance_unlabel.json')
        print(os.listdir('datasets/haitian'))
        if domain_i == 0:
            os.system('python train_net.py --num-gpus 8 --config %s SOLVER.IMG_PER_BATCH_LABEL 16 SOLVER.IMG_PER_BATCH_UNLABEL 16 SEMISUPNET.PARA_MU 0.20 SEMISUPNET.PARA_T 0.7'  % args.config)
        else:
            os.system('python train_net.py --num-gpus 8 --resume --config %s SOLVER.IMG_PER_BATCH_LABEL 16 SOLVER.IMG_PER_BATCH_UNLABEL 16 SEMISUPNET.PARA_MU 0.20 SEMISUPNET.PARA_T 0.7' % args.config)

if __name__ == "__main__":
    main()
