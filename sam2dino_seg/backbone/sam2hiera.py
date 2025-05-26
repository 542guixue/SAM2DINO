import torch
import torch.nn as nn
from sam2.build_sam import build_sam2
from sam2dino_seg.self_transforms.preprocess_image import transforms_image
from sam2dino_seg.modules import adapter
from visualize.features_vis import visualize_feature_maps_mean, visualize_feature_maps_pca, visualize_feature_maps_tsne
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os
from subprocess import check_output

def set_chinese_font():
    """安全加载自定义中文字体"""
    font_path = "/data2/users/donghang/SAM2DINO-Seg/SimHei.ttf"
    try:
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)
        font_name = prop.get_name()
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache/matplotlib")
        if os.path.exists(cache_dir):
            os.system(f"rm -rf {cache_dir}/*")
            print("已清除 Matplotlib 缓存")
        plt.rcParams['font.sans-serif'] = [font_name]
        plt.rcParams['axes.unicode_minus'] = False
        return True
    except Exception as e:
        print(f"❌ 字体加载失败: {str(e)}")
        return False

if set_chinese_font():
    print("✅ 中文字体加载成功")
else:
    print("❌ 使用默认字体")
print("已加载字体列表:", [f.name for f in font_manager.fontManager.ttflist if 'SimHei' in f.name])

class sam2hiera(nn.Module):
    def __init__(self, config_file=None, ckpt_path=None, device=None) -> None:
        super().__init__()
        if config_file is None:
            config_file = "./sam2_configs/sam2.1_hiera_l.yaml"
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = build_sam2(config_file, ckpt_path, device=device)
        model.to(device)

        # 删除不需要的模块，保留image_encoder.trunk作为sam_encoder
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck

        self.sam_encoder = model.image_encoder.trunk

        for param in self.sam_encoder.parameters():
            param.requires_grad = False

        # Adapter
        blocks = []
        for block in self.sam_encoder.blocks:
            adapted_block = adapter.Adapter(block).to(device)
            blocks.append(adapted_block)
        self.sam_encoder.blocks = nn.Sequential(*blocks)

        self.to(device)

    def forward(self, x):
        # 直接调用sam_encoder，假设sam_encoder内部实现了完整前向
        print(f"Input image size: {x.shape}")
        out = self.sam_encoder(x)
        # out一般是list或者tuple，按需要返回
        return out

def get_free_gpu():
    try:
        result = check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        free_memory = [int(x) for x in result.strip().split('\n')]
        return free_memory.index(max(free_memory))
    except Exception as e:
        print("自动检测空闲 GPU 失败，默认使用 0 号 GPU。", e)
        return 0

if __name__ == "__main__":
    config_file = "/data2/users/donghang/SAM2DINO-Seg/sam2_configs/sam2.1_hiera_l.yaml"
    ckpt_path = "/data2/users/donghang/SAM2DINO-Seg/checkpoints/sam2.1_hiera_large.pt"
    image_path = "/data2/users/donghang/SAM2DINO-Seg/data/TrainDataset/image/1.png"

    x = transforms_image(image_path, image_size=352)
    gpu_id = get_free_gpu()
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"✅ 使用 GPU: {device}")

    with torch.no_grad():
        model = sam2hiera(config_file, ckpt_path, device=device)
        x = x.to(device)
        out = model(x)

        features = {
            'top_level': out[3],
            'high_level': out[2],
            'mid_level': out[1],
            'low_level': out[0]
        }

        print(f"顶级特征形状 (全局尺度): {features['top_level'].shape}")
        print(f"高级特征形状 (高等尺度): {features['high_level'].shape}")
        print(f"中级特征形状 (中等尺度): {features['mid_level'].shape}")
        print(f"低级特征形状 (局部尺度): {features['low_level'].shape}")

        visualize_feature_maps_mean(features, backbone_name='SAM2')
        visualize_feature_maps_pca(features, backbone_name='SAM2')
        visualize_feature_maps_tsne(features, backbone_name='SAM2')

        print("Hiera多尺度特征提取完成!")
