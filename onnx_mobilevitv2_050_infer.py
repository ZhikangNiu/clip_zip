import onnx
import onnxruntime
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision.transforms import transforms


class Mobilevitv2():
    def __init__(self,
                 onnx_path:str,
                onnx_img_size = 256
                 ):
        """

        Args:
            onnx_path: 传入的onnx文件的位置
            onnx_img_size: 在导出onnx格式时，张量的大小
        """
        self.onnx_path = onnx_path
        self.onnx_model = onnx.load(onnx_path)
        self.ort_session = onnxruntime.InferenceSession(self.onnx_path)
        self.img_size = (onnx_img_size,onnx_img_size)
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
        ])

    def _set_type(self):
        """
        clip的数据格式是torch.float16
        Returns: torch.float16

        """
        return torch.float16

    def _to_numpy(self,tensor):
        """
        转化为numpy格式
        Args:
            tensor:

        Returns:

        """
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def infer(self,img_path):
        """

        Args:
            img_path:传入图像

        Returns:

        """
        # 检验onnx文件是否可用
        try:
            onnx.checker.check_model(self.onnx_model)
        except onnx.checker.ValidationError as e:
            print("The model is invalid: %s"%e)
        else:
            print("The model is valid!")

        img = Image.open(img_path).convert("RGB")
        x = self.transform(img).type(self._set_type())
        x = x.unsqueeze(0)
        ort_inputs = {self.ort_session.get_inputs()[0].name:self._to_numpy(x)}
        output_name = self.ort_session.get_outputs()[0].name
        ort_outs = self.ort_session.run([output_name],ort_inputs)
        return ort_outs




#
# onnx_model = onnx.load(onnx_path)
# onnx.checker.check_model(onnx_model)
#
# ort_session = onnxruntime.InferenceSession(onnx_path)
#
# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
#
# x = torch.rand(1,3,256,256).type(torch.float16)
# ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
# print(ort_inputs)
# ort_outs = ort_session.run(None, ort_inputs)
# # compare ONNX Runtime and PyTorch results
# # np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

if __name__ == '__main__':
    onnx_path = "./mobilevitv2_050_place365.onnx"
    img_path = "./test.png"
    onnx_infer = Mobilevitv2(onnx_path)
    ort_puts = onnx_infer.infer(img_path)
    print(type(ort_puts))
    print(type(ort_puts[0]))
    print(ort_puts)
    #print(ort_puts[0].shape)