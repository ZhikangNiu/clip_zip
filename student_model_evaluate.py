# -*- coding: utf-8 -*-
# @Time    : 2022-07-21 19:47
# @Author  : Zhikang Niu
# @FileName: student_model_evaluate.py
# @Software: PyCharm

import torch
import timm
import clip
import time
from PIL import Image
from clip.model import CLIP


checkpoint_path = "./checkpoint/KD_BEST_student.pth"

# 加载模型
student_model = timm.create_model('mobilevitv2_050',pretrained=True,num_classes=512)
# 加载参数，从多卡转化为单卡
state_dict = torch.load(checkpoint_path)
student_model.load_state_dict({k.replace('module.',''):v for k,v in state_dict.items()})
student_model = student_model.cuda().half() # 将权重从float32 --> float16

model,preprocess = clip.load("ViT-B/32",device='cuda')

model.visual = student_model.eval()

start = time.time()
image = preprocess(Image.open("cat.png")).unsqueeze(0).cuda()
text = clip.tokenize(["Cat", "a photo of small Gat","a image of small cat in it", "A photo of small cat","a brown cat"]).cuda()

with torch.no_grad():
    # image_features.shape = [1,512]
    image_features = model.visual(image.half())
    # text_features.shape = [3,512]
    text_features = model.encode_text(text)
    # logits_per_image.shape = [1,3]
    # logits_per_text.shape = [3,1]
    # logits_per_text = logits_per_image的转置
    logits_per_image, logits_per_text = model(image, text)

    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

end = time.time()
print(f"各个物体的概率：{probs}")
print(f"所消耗时间：{end-start}s")