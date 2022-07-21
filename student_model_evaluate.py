# -*- coding: utf-8 -*-
# @Time    : 2022-07-21 19:47
# @Author  : Zhikang Niu
# @FileName: student_model_evaluate.py
# @Software: PyCharm

import torch
import timm

checkpoint_path = "./checkpoint/KD_BEST_epoch_29.pth"
print(checkpoint_path)
student_model = timm.create_model('mobilevitv2_050',pretrained=True,num_classes=512)
state_dict = torch.load(checkpoint_path)
print(state_dict)