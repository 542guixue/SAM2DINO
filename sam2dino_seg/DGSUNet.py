import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import dinov2_extract, sam2hiera
from fusion import CGAFusion, sff
from modules import updown, wtconv, RFB
from torchinfo import summary


class DGSUNet(nn.Module):
    def __init__(self, dino_model_name=None, dino_hub_dir=None, sam_config_file=None, sam_ckpt_path=None):
        super(DGSUNet, self).__init__()
        if dino_model_name is None:
            print("No model_name specified, using default")
            dino_model_name = 'dinov2_vitl14'
        if dino_hub_dir is None:
            print("No dino_hub_dir specified, using default")
            dino_hub_dir = 'facebookresearch/dinov2'
        if sam_config_file is None:
            print("No sam_config_file specified, using default")
            sam_config_file = '/data2/users/donghang/SAM2DINO-Seg/sam2_configs/sam2.1_hiera_l.yaml'
        if sam_ckpt_path is None:
            print("No sam_ckpt_path specified, using default")
            sam_ckpt_path = '/data2/users/donghang/SAM2DINO-Seg/checkpoints/sam2.1_hiera_large.pt'

        # Backbone
        self.backbone_dino = dinov2_extract.DinoV2FeatureExtractor(dino_model_name, dino_hub_dir)
        self.backbone_sam = sam2hiera.sam2hiera(sam_config_file, sam_ckpt_path)

        # Feature Fusion
        self.fusion4 = CGAFusion.CGAFusion(1152)
        self.dino2sam_down4 = updown.interpolate_upsample(11)
        self.dino2sam_down14 = wtconv.DepthwiseSeparableConvWithWTConv2d(in_channels=1024, out_channels=1152)

        self.rfb1 = RFB.RFB_modified(144, 64)
        self.rfb2 = RFB.RFB_modified(288, 64)
        self.rfb3 = RFB.RFB_modified(576, 64)
        self.rfb4 = RFB.RFB_modified(1152, 64)

        self.decoder1 = sff.SFF(64)
        self.decoder2 = sff.SFF(64)
        self.decoder3 = sff.SFF(64)

        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(64, 1, kernel_size=1)
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x_dino, x_sam):
        x1, x2, x3, x4 = self.backbone_sam(x_sam)
        x_dino = self.backbone_dino(x_dino)
        x_dino4 = self.dino2sam_down4(x_dino)
        x_dino4 = self.dino2sam_down14(x_dino4)

        x4 = self.fusion4(x4, x_dino4)

        x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
        x = self.decoder1(x4, x3)
        out1 = F.interpolate(self.side1(x), scale_factor=16, mode='bilinear', align_corners=False)
        x = self.decoder2(x, x2)
        out2 = F.interpolate(self.side2(x), scale_factor=8, mode='bilinear', align_corners=False)
        x = self.decoder3(x, x1)
        out3 = F.interpolate(self.head(x), scale_factor=4, mode='bilinear', align_corners=False)

        return out1, out2, out3


######################################################################################################

if __name__ == "__main__":
    import subprocess

    def get_free_gpu():
        try:
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
                encoding='utf-8'
            )
            free_memory = [int(x) for x in result.strip().split('\n')]
            return free_memory.index(max(free_memory))
        except Exception as e:
            print("自动检测空闲 GPU 失败，默认使用 0 号 GPU。", e)
            return 0

    gpu_id = get_free_gpu()
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"使用 GPU: {gpu_id}")

    with torch.no_grad():
        model = DGSUNet()

        # 默认仅用单卡
        model = model.to(device)

        # 构造输入张量
        x_dino = torch.randn(1, 3, 518, 518).to(device)
        x_sam = torch.randn(1, 3, 352, 352).to(device)

        # 可选：支持多 GPU，只在 device_ids 显式设置时启用
        # device_ids = [0, 1]  # 如果你有多个 GPU 可用，解除注释
        # model = nn.DataParallel(model.to(torch.device("cuda:0")), device_ids=device_ids)

        # 打印模型结构
        summary(model, input_data=(x_dino, x_sam), device=str(device))

        out, out1, out2 = model(x_dino, x_sam)
        print(out.shape, out1.shape, out2.shape)
