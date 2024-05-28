import torch
from torchvision.models import vit_b_16
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from models.cswin_boat import CSWin_BOAT_64_24322_small_224
# 创建resnet50网络
model = CSWin_BOAT_64_24322_small_224(num_classes=13).cuda()
print(model)
# 创建输入网络的tensor
tensor = (torch.rand(16, 3, 224, 224),)

# 分析FLOPs
flops = FlopCountAnalysis(model, tensor)
print("FLOPs: ", flops.total())

# 分析parameters
print(parameter_count_table(model))
