#!/usr/bin/python3
# coding=utf-8
from __future__ import absolute_import, division, print_function
import glob
import logging
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import cv2
from models.model_densenet import densenet121
from PIL import Image
import torchvision.transforms as transforms
logger = logging.getLogger(__name__)

def process_and_stack_images(folder1, folder2, output_folder, image_size=(640, 640)):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取两个文件夹中的图片文件名
    images1 = [img for img in os.listdir(folder1) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images2 = [img for img in os.listdir(folder2) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 按文件名排序以匹配图片
    images1.sort()
    images2.sort()

    # 迭代图片文件名，按序号处理和叠加图片
    for img1, img2 in zip(images1, images2):
        img_path1 = os.path.join(folder1, img1)
        img_path2 = os.path.join(folder2, img2)

        # 读取并调整两张图片的大小
        image1 = cv2.imread(img_path1)
        image2 = cv2.imread(img_path2)
        image1 = cv2.resize(image1, image_size)
        image2 = cv2.resize(image2, image_size)

        # 垂直叠加两张图片
        stacked_image = cv2.vconcat([image1, image2])

        # 保存叠加后的图片
        output_path = os.path.join(output_folder, f"stacked_{img1}")
        cv2.imwrite(output_path, stacked_image)

# 图片存放的文件夹路径
folder1 = "F:/Actcls/redata/assemble"
folder2 = "F:/Actcls/redata/part"

# 处理后的文件保存路径
output_folder = "F:/Actcls/redata/predict"

# 调用函数处理图片并叠加
process_and_stack_images(folder1, folder2, output_folder)

def setup(args):
    # Prepare model
    num_classes = 7
    if args.method == 'densenet':
        model = densenet121(num_classes=num_classes)
        model_weight_path = args.test_pretrain_weights
        assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
        pretrained_dict = torch.load(model_weight_path, map_location=args.device)['model']
        model.load_state_dict(pretrained_dict)
    else:
        print('model chose error！')
    model.to(args.device)
    num_params = count_parameters(model)
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


from pathlib import Path

def valid(args, model):
    # 获取目录中所有图像文件
    image_files = list(Path(args.data_path).glob('*.jpg'))  # 假设图片是jpg格式，您可以根据需要调整
    classes = ['end', 'hand', 'move1', 'move2', 'move3', 'move4', 'start']

    model.eval()

    for image_file in image_files:
        # 打开图像文件
        x = Image.open(image_file)
        x = x.resize((256, 512))

        with torch.no_grad():
            transf = transforms.ToTensor()
            x = transf(x)
            x = x.to(args.device)
            x = x.unsqueeze(0)
            logits = model(x)
            preds = torch.argmax(logits, dim=-1)
            print(str(image_file) + ' image type: ', classes[int(preds.item())])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=r'F:\Actcls\redata\predict')
    parser.add_argument('--method', type=str, default="densenet", choices=["densenet"])
    parser.add_argument("--test_pretrain_weights", type=str, default=r"F:\Actcls\trainingrecords\densenet_sgd_b8\_checkpoint_2800.bin", help="test_pretrain_weights path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    args.nprocs = torch.cuda.device_count()
    args, model = setup(args)
    valid(args, model)
