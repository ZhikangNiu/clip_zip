# -*- coding: utf-8 -*-
# @Time    : 2022-08-08 18:41
# @Author  : Zhikang Niu
# @FileName: evaluate_mobilevitv2_clip.py
# @Software: PyCharm

import torch
import timm
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

import clip
from PIL import Image

from ClipDataset import JsonDataset
from clip.model import CLIP

"""
# 遍历数据集，对每一张图片进行测试
# 使用预测接口，每次遍历一张图片，传进去的是图片路径，读取图片
# transformer传进去的是一个f"a photo in {label_name}" 使用列表生成式进行生成一个token_list
"""

def token_list():
    # 先读取文件
    label_list = []
    file = open("/home/public/datasets/place365/filelist/categories_places365.txt","r")
    for line in file:
        label_name = line.split()[0][3:]
        label_list.append(label_name)
    token_list = ["a photo of "+label_name for label_name in label_list]
    return token_list


def evaluate_top3():
    pass
def evaluate_top5():
    pass
def evaluate_top10():
    pass

@torch.no_grad()
def evaluate_mobilevitv2_clip(token_list):
    # 设置目录
    TRAIN_JSON_PATH = '/home/niuzhikang/src/SceneClassification/filelist/json_filelist/train.json'
    TRAIN_IMAGE_PATH = '/home/public/datasets/place365/train/data_256'
    #checkpoint_path = "./checkpoint/KD_BEST_student.pth"

    # 加载模型
    #student_model = timm.create_model('mobilevitv2_050',pretrained=True,num_classes=512)
    # 加载参数，从多卡转化为单卡
    #state_dict = torch.load(checkpoint_path)
    #student_model.load_state_dict({k.replace('module.',''):v for k,v in state_dict.items()})
    #student_model = student_model.cuda().half() # 将权重从float32 --> float16

    # 将视觉模型转换为mobilevitv2的模型
    model,preprocess = clip.load("ViT-L/14",device='cuda')
    #model.visual = student_model.eval()

    # 读取数据集
    process = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = JsonDataset(TRAIN_JSON_PATH, TRAIN_IMAGE_PATH, process)

    loader = DataLoader(dataset,batch_size=64,)

    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    top10_correct = 0
    total = len(loader.dataset)
    for data,label in tqdm(loader):
        image = data.cuda()
        label = label.cuda()

        text = clip.tokenize(token_list).cuda()
        logits_per_image, logits_per_text = model(image, text)
        # 计算top1
        pred = logits_per_image.argmax(dim=1)
        top1_correct += torch.eq(pred, label).sum().float().item()

        label_resize = label.view(-1,1)
        _,pred = logits_per_image.topk(3,1,True,True)
        top3_correct += torch.sum(torch.eq(pred,label_resize).float()).item()


        maxk_5 = max((1,5))
        label_resize = label.view(-1,1)
        _,pred = logits_per_image.topk(maxk_5,1,True,True)
        top5_correct += torch.sum(torch.eq(pred,label_resize).float()).item()


        maxk_10 = max((1,10))
        label_resize = label.view(-1,1)
        _,pred = logits_per_image.topk(maxk_10,1,True,True)
        top10_correct += torch.sum(torch.eq(pred,label_resize).float()).item()

        print(f"Top1预测准确: {top1_correct}个")
        print(f"top1_correct: {top1_correct / total}")
        print(f"Top3预测准确: {top3_correct}个")
        print(f"top3_correct: {top3_correct / total}")
        print(f"Top5预测准确: {top5_correct}个")
        print(f"top5_correct: {top5_correct / total}")
        print(f"Top10预测准确: {top10_correct}个")
        print(f"top10_correct: {top10_correct / total}")
    with open("./clip_correct.txt", "w") as f:
        f.write(f"Top1预测准确: {top1_correct}个\n")
        f.write(f"top1_correct: {top1_correct / total}\n")
        f.write(f"Top3预测准确: {top3_correct}个\n")
        f.write(f"top3_correct: {top3_correct / total}\n")
        f.write(f"Top5预测准确: {top5_correct}个\n")
        f.write(f"top5_correct: {top5_correct / total}\n")
        f.write(f"Top10预测准确: {top10_correct}个\n")
        f.write(f"top10_correct: {top10_correct / total}\n")


if __name__ == '__main__':
    torch.manual_seed(66)
    torch.cuda.manual_seed_all(66)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    token_list = token_list()
    # with open('./token_list.txt','w') as f:
    #     for i in token_list:
    #         f.write(f"{i}\n")
    evaluate_mobilevitv2_clip(token_list)
