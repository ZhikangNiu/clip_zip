# -*- coding: utf-8 -*-
# @Time    : 2022-07-25 16:13
# @Author  : Zhikang Niu
# @FileName: export_onnx.py
# @Software: PyCharm

import torch.onnx
import timm
import torch

def convert_onnx(model,
                 checkpoint_path:str,
                 parallel_trained:bool=True,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
    # 加载参数，从多卡转化为单卡
    state_dict = torch.load(checkpoint_path)
    if parallel_trained:
        model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
        model = model.half().to(device)
    else:
        model.load_state_dict(state_dict)
        model = model.half.to(device)
    # 将模型开启测试
    model.eval()
    dummy_input = torch.randn(1, 3,256,256, requires_grad=True).type(torch.float16).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        "mobilevitv2_050_place365.onnx",
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=['modelInput'],  # the model's input names
        output_names=['modelOutput'],  # the model's output names
        dynamic_axes={'modelInput': {0: 'batch_size'},
                      'modelOutput':{0:'batch_size'}}
    )

    print('Model has been converted to ONNX')

if __name__ == '__main__':
    model = timm.create_model('mobilevitv2_050', num_classes=512)
    checkpoint_path = './checkpoint/KD_BEST_student.pth'
    convert_onnx(model,checkpoint_path)