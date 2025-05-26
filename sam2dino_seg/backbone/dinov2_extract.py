import torch
import torch.nn as nn
import numpy as np
from matplotlib import rcParams
from sam2dino_seg.self_transforms.preprocess_image import transforms_image
from dinov2.models.vision_transformer import vit_large  # 需要先安装 dinov2 包

# 安装 dinov2 包（如果尚未安装）
# pip install dinov2

# 设置全局字体为 SimHei（黑体）
rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示为方块的问题

class DinoV2FeatureExtractor(nn.Module):
    def __init__(self,dino_model_name=None, dino_hub_dir=None) -> None:
        super().__init__()
        # 加载模型架构
        self.dino_encoder = vit_large(
        patch_size=14,
        img_size=518,
        init_values=1.0,
        block_chunks=0)
        # 加载本地权重文件（修改为你实际的路径）
        ckpt_path = "/data2/users/donghang/SAM2DINO-Seg/checkpoints/dinov2_vitl14_pretrain.pth"
        state_dict = torch.load(ckpt_path)
        
        # 处理权重格式
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.dino_encoder.load_state_dict(state_dict, strict=False)
        self.patchsize = 14
        self.dino_encoder.eval()

        # 冻结所有参数
        for param in self.dino_encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        output = self.dino_encoder.forward_features(x)
        dino_feature = output['x_norm_patchtokens']
        
        # 转换为空间特征图
        img_size = x.shape[-1]
        batch_size = x.shape[0]
        feature_size = (img_size // self.patchsize) ** 2
        
        assert dino_feature.shape[1] == feature_size, \
            f"特征大小不匹配: {dino_feature.shape[1]} vs {feature_size}"
        
        side_length = int(np.sqrt(feature_size))
        dino_feature_map = dino_feature.reshape(
            batch_size, side_length, side_length, -1
        ).permute(0, 3, 1, 2)

        return dino_feature_map


if __name__ == "__main__":
    # 预处理图像示例
    # image_path = "your_image_path.jpg"
    # x = transforms_image(image_path, image_size=518)
    
    # 创建随机输入
    x = torch.randn(12, 3, 518, 518)
    
    # 初始化模型
    model = DinoV2FeatureExtractor()
    
    # GPU 支持
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
    
    # 前向传播
    with torch.no_grad():
        out = model(x)
        print("输出特征图形状:", out.shape)
        # print("特征图数值示例:", out[0, :3, 0, 0])
