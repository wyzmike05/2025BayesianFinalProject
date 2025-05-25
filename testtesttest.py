from torchsummary import summary
from utils_train import make_bfn, make_config
from train import setup
from omegaconf import OmegaConf

cfg_file = "configs/mnist_discrete_.yaml"
cfg = make_config(cfg_file)
model, dataloaders, optimizer = setup(cfg)

for data in dataloaders["train"]:
    x, y = data
    print(x.shape)
    print(y.shape)
    break

# print(model)
# summary(model, (28, 28, 1), device="cpu")
