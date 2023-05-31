luna_path = r'E:\Lung cancer CT data set\LUNA16'
xml_file_path = r'E:\Lung cancer CT data set\LIDC-XML-only\tcia-lidc-xml'
annos_csv = r'E:\Lung cancer CT data set\LUNA16\CSVFILES\CSVFILES\annotations.csv'
new_bbox_annos_path = r'E:\datasets\sk_output\bbox_annos\bbox_annos.xls'
mask_path = r'E:\Lung cancer CT data set\LUNA16\seg-lungs-LUNA16\seg-lungs-LUNA16'
output_path = r'E:\datasets\sk_output'
bbox_img_path = r'E:\datasets\sk_output\bbox_image'
bbox_msk_path = r'E:\datasets\sk_output\bbox_mask'
wrong_img_path = r'E:\datasets\wrong_img.xls'
zhibiao_path = r'E:\datasets\sk_output\zhibiao'
model_path = r'E:\datasets\sk_output\model'
# "E:\Lung cancer CT data set\LUNA16"
from pathlib import Path

zhibiao_path1 = Path(zhibiao_path)
zhibiao_path1.mkdir(exist_ok=True, parents=True)

xitong = "windows"  # "linux"

# 训练设置
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 没gpu就用cpu
valid_epoch_each = 5  # 每几轮验证一次

if xitong == "linux":
    fengefu = r"/"
else:
    fengefu = r"\\"

##定义文件路径及不同系统的分隔符
