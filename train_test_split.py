import glob
import os
import random
import shutil
from collections import defaultdict
import tqdm

random.seed(2019)
# %%
all_images = glob.glob(os.path.join("data", "raw", "**", "*.jpg"))

per_class = defaultdict(list)

for f in all_images:
    cls = f.split("/")[-2].lower()
    per_class[cls].append(f)

# %%
train_dir = "./data/train/"
test_dir = "./data/test/"

shutil.rmtree(train_dir, ignore_errors=True)
shutil.rmtree(test_dir, ignore_errors=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for cls in tqdm.tqdm(per_class):
    list_images = per_class[cls]
    random.shuffle(list_images)

    split = int(0.75 * len(list_images))

    train_images = list_images[:split]
    test_images = list_images[split:]
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

    for i, f in enumerate(train_images):
        shutil.copyfile(f, os.path.join(train_dir, cls, "{}_{}.jpg".format(cls, i)))
    for i, f in enumerate(test_images):
        shutil.copyfile(f, os.path.join(test_dir, cls, "{}_{}.jpg".format(cls, i)))

