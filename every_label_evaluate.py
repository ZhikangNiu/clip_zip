# -*- coding: utf-8 -*-
# @Time    : 2022-08-15 8:37
# @Author  : Zhikang Niu
# @FileName: every_label_evaluate.py
# @Software: PyCharm

import os
import torch
import timm
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

import clip
from PIL import Image

from ClipDataset import JsonDataset, single_file_dataset
from clip.model import CLIP

"""
# 遍历数据集，对每一张图片进行测试
# 使用预测接口，每次遍历一张图片，传进去的是图片路径，读取图片
# transformer传进去的是一个f"a photo in {label_name}" 使用列表生成式进行生成一个token_list
"""

def token_list(file_path):
    # 先读取文件

    """
    label_list = ["living room","dining room","toilet","bedroom","garden","shopping center","restaurant","Walking street","amusement park","high way","bridge","high-rise","city night","classroom","laboratory","library","office workstation","meeting room","stadium","stage","music hall","disco","Exhibition hall","aquarium","airport","train","subway","airplane","countryside","farmland","grassland","desert","mountain","river","sea","beach","snow","Blue sky","night","moon"]
    """
    label_list =  ["living room","dining room","bedroom","garden","shopping center","restaurant","Walking street","amusement park","high way","bridge","classroom","laboratory","library","office workstation","stadium","stage","music hall","disco","aquarium","airport","farmland","desert","mountain","river","beach","blue sky"]
    place365_label_list = ["l/living_room","d/dining_room","b/bedroom","f/formal_garden","s/shopping_mall/indoor","r/restaurant","s/street","a/amusement_park","h/highway","b/bridge","c/classroom","b/biology_laboratory","l/library/indoor","o/office","s/stadium/football","s/stage/indoor","m/music_studio","d/discotheque","a/aquarium","a/airport_terminal","f/farm","d/desert/sand","m/mountain","r/river","b/beach","s/sky",]
    category_num = len(label_list)
    print("category_num: ",category_num)

    # BUG:看下这个列表生成式为什么不对
    #place365_label_path = [os.path.join(file_path,label) if isinstance(sublabel,str) and isinstance(label,str) else os.path.join(file_path,sublabel) for label in place365_label_list for sublabel in label]

    place365_label_path = []
    for label in place365_label_list:
        if isinstance(label,str):
            place365_label_path.append(os.path.join(file_path,label))
        elif isinstance(label,list):
            for sublabel in label:
                place365_label_path.append(os.path.join(file_path,sublabel))
    token_list = ["a photo of "+label_name for label_name in label_list]
    token_path_list = zip(token_list,place365_label_path)
    return token_path_list

@torch.no_grad()
def evaluate_mobilevitv2_clip(token_list,file_path):
    checkpoint_path = "./checkpoint/KD_BEST_student.pth"
    # 加载模型
    mobilevitv2_model = timm.create_model('mobilevitv2_050',pretrained=True,num_classes=512)
    # 加载参数，从多卡转化为单卡
    state_dict = torch.load(checkpoint_path)
    mobilevitv2_model.load_state_dict({k.replace('module.',''):v for k,v in state_dict.items()})
    mobilevitv2_model = mobilevitv2_model.cuda().half() # 将权重从float32 --> float16

    # 将视觉模型转换为mobilevitv2的模型
    model,preprocess = clip.load("ViT-B/32",device='cuda')
    model.visual = mobilevitv2_model.eval()

    # 读取数据集
    process = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    for single_file in file_path:
        index = file_path.index(single_file)
        print("label: ",index)
        print("single_file: ",single_file)
        dataset = single_file_dataset(single_file,index,process)
        loader = DataLoader(dataset,batch_size=128,)
        acc = 0
        for data,label in tqdm(loader):
            image = data.cuda()
            label = label.cuda()
            text = clip.tokenize(token_list).cuda()
            logits_per_image, logits_per_text = model(image,text)
            # 返回最大的位置
            pred = logits_per_image.argmax(dim=1)
            acc += torch.eq(pred, label).sum().float().item()

        with open('./26_bs128_correct.txt','a') as f:
            f.write(f"{token_list[index]}:{acc} \n")
            f.write(f"{token_list[index]}:{acc/len(loader.dataset)} \n")




if __name__ == '__main__':
    torch.manual_seed(66)
    torch.cuda.manual_seed_all(66)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    token_path_list = token_list("/home/public/datasets/place365/train/data_256")
    # with open('./token_list.txt','w') as f:
    #     for i in token_list:
    #         f.write(f"{i}\n")
    token_list,path = zip(*token_path_list)
    token_list = list(token_list)
    print("token_list: ",token_list)
    path = list(path)
    evaluate_mobilevitv2_clip(token_list,path)

