# -*- coding: utf-8 -*-
# @Time    : 2022-07-12 14:52
# @Author  : Zhikang Niu
# @FileName: Knowledge_Distillation.py
# @Software: PyCharm

import os

from torch.optim.lr_scheduler import CosineAnnealingLR

os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from config import get_option
import clip
import timm
from PIL import Image
from ClipDataset import JsonDataset
import logging
import warnings
warnings.filterwarnings("ignore")



opt = get_option()


batch_size = opt.batch_size
epochs = opt.epochs
LOG_FILE = opt.log_file
save_folder = opt.checkpoint_dir
lr = opt.lr
gpus = opt.GPUS

print(f"{batch_size}/"
      f"{epochs}/"
      f"{lr}/"
      f"{gpus}")

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(LOG_FILE)
formatter = logging.Formatter('%(asctime)s: %(levelname)s: [%(filename)s: %(lineno)d]: %(message)s')
file_handler.setFormatter(formatter)

# print to screen
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)

# add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


# 设置目录
TRAIN_JSON_PATH = '/home/niuzhikang/src/SceneClassification/filelist/json_filelist/train.json'
TRAIN_IMAGE_PATH = '/home/public/datasets/place365/train/data_256'

# 返回模型和处理方法
model,preprocess = clip.load("ViT-B/32",device='cuda')
"""
  model          output_size
ViT-B/16             512
ViT-B/32             512
ViT-L/14             512
ViT-L/14@336px       512
"""

torch.manual_seed(66)
torch.cuda.manual_seed_all(66)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# image.shape = [1,3,224,224]
# text.shape = [3,77]
# image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
# text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

# with torch.no_grad():
#     # image_features.shape = [1,512]
#     image_features = model.encode_image(image)
#     # text_features.shape = [3,512]
#     text_features = model.encode_text(text)
#     # logits_per_image.shape = [1,3]
#     # logits_per_text.shape = [3,1]
#     # logits_per_text = logits_per_image的转置
#     logits_per_image, logits_per_text = model(image, text)
#
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# 获取视觉模型ViT
# visual_model = model.visual
# visual_model.eval()
# # 数据类型为半精度
# image = image.half()
# # 获取视觉模型的输出
# output = visual_model(image)
# # output的输出与encode_image的输出相同
# print(output.shape)

# 定义教师网络
teacher_model = model.visual.eval()
# 蒸馏温度
temp = 4
# loss = KLD(x_s.softmax(), x_t.softmax()) + smoothL1(x_s, x_t)
L1 = nn.L1Loss()
smoothL1 = nn.SmoothL1Loss()
cos_loss = nn.CosineEmbeddingLoss()
# TODO:尝试使用smoothL1 Loss或者余弦损失
# smoothL1 Loss 太平滑！
alpha = 0.3
soft_loss = nn.KLDivLoss()
# KL损失的reduction这玩意是个啥

# 创建学生模型
student_model = timm.create_model('mobilevitv2_050',pretrained=True,num_classes=512).cuda()
# 优化器
params = [p for p in student_model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params,lr=lr)
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0)


# 读取数据集
train_process = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

train_dataset = JsonDataset(TRAIN_JSON_PATH,TRAIN_IMAGE_PATH,train_process)

train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,)

if opt.GPUS > 1:
    teacher_model = nn.DataParallel(teacher_model)
    student_model = nn.DataParallel(student_model)
    teacher_model.cuda()
    student_model.cuda()

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

#bug：val经常报错的那个问题好像是最后一个判断引起的！！！！
tar = torch.ones(batch_size).cuda()
print(f"tar shape: {tar.shape}")
min_loss = 10000
for epoch in range(epochs):
    for data,_ in tqdm(train_loader):
        data = data.cuda()
        # print(data.shape)
        # 教师网络的预测
        with torch.no_grad():
            teacher_preds = teacher_model(data.half())
            # teacher_preds.shape = [batch,512]
            # print(teacher_preds.shape)

        # 学生模型的预测
        student_preds = student_model(data).half()
        #print(student_preds.shape)
        # 计算蒸馏后的预测结果及soft_loss
        ditillation_loss = soft_loss(
            F.softmax(student_preds/temp,dim=1),
            F.softmax(teacher_preds/temp,dim=1)
        )

        loss = ditillation_loss+smoothL1(student_preds,teacher_preds)
        # bug:加上cos_loss会报错 RuntimeError: The size of tensor a (1024) must match the size of tensor b (196) at non-singleton dimension 0


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    logger.info('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, loss.item()))
    if loss.item() < min_loss:
        min_loss = loss.item()
        torch.save(student_model.state_dict(), os.path.join(save_folder, 'KD_BEST_student.pth'))

torch.save(student_model.state_dict(),os.path.join(save_folder,'last_KD_student_model.pth'))
