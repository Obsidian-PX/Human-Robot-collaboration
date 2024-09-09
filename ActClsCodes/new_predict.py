import cv2
import os
import time
from datetime import datetime
import torch
from models.model_densenet import densenet121
from PIL import Image
import torchvision.transforms as transforms
import logging
import argparse
import glob  # 用于获取目录下的文件列表

logger = logging.getLogger(__name__)

def process_and_stack_images(folder1, folder2, output_folder, image_size=(640, 640)):
    while True:
        images1 = [img for img in os.listdir(folder1) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        images2 = [img for img in os.listdir(folder2) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if images1 and images2:
           images1.sort()
           images2.sort()
           for img1, img2 in zip(images1, images2):
               img_path1 = os.path.join(folder1, img1)
               img_path2 = os.path.join(folder2, img2)
               image1 = cv2.imread(img_path1)
               image2 = cv2.imread(img_path2)
               image1 = cv2.resize(image1, image_size)
               image2 = cv2.resize(image2, image_size)
               stacked_image = cv2.vconcat([image1, image2])
               timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
               output_filename = f"stacked_{timestamp}.jpg"  # 使用时间戳保证文件名唯一
               output_path = os.path.join(output_folder, output_filename)
               cv2.imwrite(output_path, stacked_image)
           break
        else:
            time.sleep(1)

def clean_folders(folders):
    for folder in folders:
        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)
            os.remove(file_path)
            print(f"Removed {file_path}")

def setup(args):
    num_classes = 7
    if args.method == 'densenet':
        model = densenet121(num_classes=num_classes)
        model_weight_path = args.test_pretrain_weights
        assert os.path.exists(model_weight_path), "file does not exist.".format(model_weight_path)
        pretrained_dict = torch.load(model_weight_path, map_location=args.device)['model']
        model.load_state_dict(pretrained_dict)
    else:
        print('model choice error！')
    model.to(args.device)
    num_params = count_parameters(model)
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000

label_to_number = {'hand': 0, 'move1': 1, 'move2': 2, 'move3': 3, 'move4': 4, 'start': 5, 'end': 5}


def valid(args, model, output_path="F:/Actcls/output.txt"):
    files = glob.glob(os.path.join(args.data_path, '*.jpg'))
    for file in files:
        x = Image.open(file)
        x = x.resize((256, 512))
        classes = ['end', 'hand', 'move1', 'move2', 'move3', 'move4', 'start']
        model.eval()
        with torch.no_grad():
            transf = transforms.ToTensor()
            x = transf(x)
            x = x.to(args.device)
            x = x.unsqueeze(0)
            logits = model(x)
            preds = torch.argmax(logits, dim=-1)
            result_label = classes[int(preds.item())]
            result_number = label_to_number[result_label]
            print(f'{file} image type: {result_label} ({result_number})')

        #直接覆盖文件内容，离开这个with块时，内容自动保存到文件
        with open(output_path, 'w') as f:
            f.write(str(result_number))

        os.remove(file)  # 删除处理过的文件


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=r'F:\Actcls\redata\predict')
    parser.add_argument('--method', type=str, default="densenet", choices=["densenet"])
    parser.add_argument("--test_pretrain_weights", type=str, default=r"F:\Actcls\trainingrecords\densenet_sgd_b8\_checkpoint_2800.bin")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    args.nprocs = torch.cuda.device_count()
    args, model = setup(args)
    folder1 = "F:/Actcls/redata/assemble"
    folder2 = "F:/Actcls/redata/part"
    output_folder = "F:/Actcls/redata/predict"
    while True:
        process_and_stack_images(folder1, folder2, output_folder)
        valid(args, model, output_path="F:/Actcls/output.txt")
        clean_folders([folder1, folder2])  # 清理源文件夹
        print("Waiting for new images...")
        print("Using device:", args.device)
