# %%
%reload_ext autoreload
%autoreload 2
from trojanvision.scripts.train import train

# %%
train("--download --tensorboard --log_dir data/tensorboard --validate_interval 1")

# %%
