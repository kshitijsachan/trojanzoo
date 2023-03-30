# %%
# %reload_ext autoreload
# %autoreload 2
from trojanvision.scripts.train import train

# %%
dataset_models = [
    ("cifar10", "resnet18"),
    ("cifar10", "densenet121"),
    ("cifar10", "vgg13"),
    ("cifar100", "resnet18"),
    ("gtsrb", "resnet18"),
    ("imagenet", "resnet18"),
    ("vggface2", "resnet18"),
]
dataset, model = dataset_models[0]
# train(f"--config ./examples/train_model_cifar10_resnet18.yml --verbose 1 --color")
train(f"--config ./examples/train_model_cifar10_resnet18")

# %%
