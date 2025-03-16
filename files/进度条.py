import time
from tqdm import tqdm

for i in tqdm(range(100),desc = "Training",
              unit = "epoch"):
    time.sleep(0.1)

