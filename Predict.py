"""
This script is used to pred the results
Created by chenmingliang in 2020/12/16
"""
import torch
import os
from torch import nn
import pydicom
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from natsort import ns, natsorted

class gen(torch.nn.Module):

    def __init__(self, input_dim, n_layers=3):
        super(gen, self).__init__()
        self.ch_in = input_dim
        self.fea = 64
        self.layer1 = nn.Sequential(nn.Conv2d(self.ch_in, self.fea, 3), nn.ReLU(inplace=True))  # 62
        self.layer2 = nn.Sequential(nn.Conv2d(self.fea, self.fea, 3), nn.ReLU(inplace=True))  # 62
        self.layer3 = nn.Sequential(nn.Conv2d(self.fea, self.fea, 3), nn.ReLU(inplace=True))  # 62
        self.layer4 = nn.Sequential(nn.Conv2d(self.fea, self.fea, 3), nn.ReLU(inplace=True))  # 62
        self.layer5 = nn.Sequential(nn.Conv2d(self.fea, self.fea, 3), nn.ReLU(inplace=True))  # 62
        self.layer6 = nn.Sequential(nn.Conv2d(self.fea, self.fea, 3), nn.ReLU(inplace=True))  # 62
        self.layer7 = nn.Sequential(nn.Conv2d(self.fea, self.fea, 3), nn.ReLU(inplace=True))  # 62
        self.layer8 = nn.Sequential(nn.Conv2d(self.fea, self.fea, 3), nn.ReLU(inplace=True))  # 62


        self.layer8_r = nn.ConvTranspose2d(self.fea, self.fea, 3)
        self.layer7_r = nn.ConvTranspose2d(self.fea, self.fea, 3)
        self.layer6_r = nn.ConvTranspose2d(self.fea, self.fea, 3)
        self.layer5_r = nn.ConvTranspose2d(self.fea, self.fea, 3)
        self.layer4_r = nn.ConvTranspose2d(self.fea, self.fea, 3)
        self.layer3_r = nn.ConvTranspose2d(self.fea, self.fea, 3)
        self.layer2_r = nn.ConvTranspose2d(self.fea, self.fea, 3)
        self.layer1_r = nn.ConvTranspose2d(self.fea, 1, 3)
        # self.output_layer = nn.Conv2d(self.fea, 1, 3, padding=1)

    def forward(self, x):
        layer1 = self.layer1(x)  # 62
        layer2 = self.layer2(layer1)  #  60
        layer3 = self.layer3(layer2)  #  58
        layer4 = self.layer4(layer3)  #  56
        layer5 = self.layer5(layer4)  #  54
        layer6 = self.layer6(layer5)  #  52
        layer7 = self.layer7(layer6)  #  50
        layer8 = self.layer8(layer7)  #  48

        outputs_8_r = self.layer8_r(layer8)  # 50
        outputs_8_r = outputs_8_r + layer7
        outputs_8_r = nn.ReLU(inplace=True)(outputs_8_r)

        outputs_7_r = self.layer7_r(outputs_8_r)  # 52
        outputs_7_r = outputs_7_r + layer6
        outputs_7_r = nn.ReLU(inplace=True)(outputs_7_r)

        outputs_6_r = self.layer6_r(outputs_7_r)  # 54
        outputs_6_r = outputs_6_r + layer5
        outputs_6_r = nn.ReLU(inplace=True)(outputs_6_r)

        outputs_5_r = self.layer5_r(outputs_6_r)  # 56
        outputs_5_r = outputs_5_r + layer4
        outputs_5_r = nn.ReLU(inplace=True)(outputs_5_r)

        outputs_4_r = self.layer4_r(outputs_5_r)  # 58
        outputs_4_r = outputs_4_r + layer3
        outputs_4_r = nn.ReLU(inplace=True)(outputs_4_r)

        outputs_3_r = self.layer3_r(outputs_4_r)  # 60
        outputs_3_r = outputs_3_r + layer2
        outputs_3_r = nn.ReLU(inplace=True)(outputs_3_r)

        outputs_2_r = self.layer2_r(outputs_3_r)  # 62
        outputs_2_r = nn.ReLU(inplace=True)(outputs_2_r)

        outputs_1_r = self.layer1_r(outputs_2_r)  # 64
        # outputs_1_r = self.output_layer(outputs_1_r)
        outputs_1_r = outputs_1_r + x
        outputs_1_r = nn.ReLU(inplace=True)(outputs_1_r)

        return outputs_1_r

def Tes2dicom(gen_a2b, test_folder="C:\Data\CTLung\Celeba_TA\chenqiufang",
              sav_folder="F:\AI_Projects\CTDenoise\LIR-for-Unsupervised-IR\Deblur_Real"):
    """
        This function is used to test and save as a whole dicom
        --Created by chenmingliang in 2020/9/10
    """
    test_folder = test_folder
    if not os.path.exists(test_folder):
        raise Exception('input path is not exists!')
    imglist = os.listdir(test_folder)
    sav_folder = os.path.join(sav_folder)
    if not os.path.exists(sav_folder):
        os.makedirs(sav_folder)
    gen_a2b.eval()
    for i, dicom_file in enumerate(imglist):
        if os.path.splitext(dicom_file)[-1] == '.dcm':
            ds = pydicom.dcmread(os.path.join(test_folder, dicom_file))
            arr_data = ds.pixel_array
            arr_data = arr_data[np.newaxis, np.newaxis, ...]
            arr_data = ((arr_data + 1000) * 1 / 4000).clip(0, 1)
            # pimg = Image.frombytes("L", arr_data.shape, arr_data.tostring())  # img is a PIL image
            # pimg = Image.fromarray(arr_data.astype('uint8'))  # img is a PIL image
            # image = transform(pimg).unsqueeze(0).cuda()
            image = torch.from_numpy(arr_data).type(torch.cuda.FloatTensor)
            # Start testing
            a2b = gen_a2b(image)
            tmp = a2b.data.cpu()
            ds.PixelData = (tmp.numpy() * 4000 - 1000).astype(np.int16)
            if not os.path.exists(sav_folder):
                os.makedirs(sav_folder)
            path = os.path.join(sav_folder, dicom_file)
            ds.save_as(path)
    print('Finish save dicom!')


def pred_test(gen_a2b, test_file, Sav_file='./pred'):
    gen_a2b.load_state_dict(torch.load("pred_weight/net.pt"))
    Tes2dicom(gen_a2b, test_file, Sav_file)


if __name__ == '__main__':
    # test_folder = ".\\train\Aliasing_artifacts\d8\D8_F1"
    # test_folder = "E:\卷叠伪影\XFFS\Jupiter\\2.16.840.1.114492.84191100109195117.19141122203.5540.213"
    ori_root = "E:\卷叠伪影\XFFS4\CT128_without_BPwgt"
    # path = natsorted(os.listdir(ori_root))
    count = 0
    for root, folder, files in natsorted(os.walk(ori_root)):
        if len(files) > 0:
            if os.path.splitext(files[0])[-1] == '.dcm':

                count += 1
                test_folder = root
                file_name = root.split('\\')
                Sav_folder = './pred_test/CT128_without_BPwgt/' + file_name[-1]
                net_a2b = gen(input_dim=1).cuda()
                # net_a2b = nn.DataParallel(net_a2b)
                pred_test(net_a2b, test_file=test_folder, Sav_file=Sav_folder)
