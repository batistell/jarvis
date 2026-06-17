import pytest
from tqdm import tqdm

# Desativa a thread de monitoramento do tqdm para evitar problemas com threads concorrentes do PyTorch/OpenMP no Windows
tqdm.monitor_interval = 0
