# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Training a basic CNN with Ignite on Google AI Platform Notebook
# Toulouse Data Science 18/06/2019

# %% [markdown]
# ## Installation of custom dependencies

# %%
# Houston we have GPU
# !nvcc -V
# !nvidia-smi

# %%
# List existing dependencies
# !pip3 list

# %%
# installation of custom dependencies
# %pip install tqdm pytorch-ignite

# %% [markdown]
# ## Download data from GCS

# %%
# get data
# !gsutil -m cp -r gs://fchouteau-storage/eurosat.tar.gz .

# %%
# !tar -zxf eurosat.tar.gz

# %%
# !ls eurosat/
# !ls eurosat/train/

# %%
# various imports
import glob
import os
import random
from itertools import cycle

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import ignite.engine
import ignite.metrics
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, RandomVerticalFlip
from tqdm import tqdm

# %% [markdown]
# ## Configuration

# %%
random.seed(2019)
BATCH_SIZE = 64
LEARNING_RATE = 0.01
MOMENTUM = 0.9
MAX_EPOCHS = 2

# %% [markdown]
# ## Datasets and dataloaders definitions


# %%
# Define datasets
class EurosatDataset(Dataset):
    def __init__(self, images_files, shuffle=True):
        self.images = images_files
        self.classes = list(set([f.split("/")[-2] for f in self.images]))
        self.classes = sorted(self.classes)

        if shuffle:
            random.shuffle(self.images)

    @classmethod
    def from_fold(cls, data_dir, fold, shuffle=True):
        images = glob.glob(os.path.join(data_dir, fold, "**", "*.jpg"))
        return cls(images, shuffle=shuffle)

    def split(self, n=0.75):
        n = int(n * len(self))
        return EurosatDataset(self.images[:n], shuffle=True), EurosatDataset(self.images[n:], shuffle=True)

    def __getitem__(self, item):
        img = self.images[item]
        img = Image.open(img)
        cls = self.classes.index(self.images[item].split("/")[-2])

        return img, cls

    def __len__(self):
        return len(self.images)


class TransformedDataset(Dataset):
    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x, y = self.dataset[item]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y


# %%
# Get data loaders functions


def get_datasets(root_dir):

    train_data_transform = Compose([RandomHorizontalFlip(), RandomVerticalFlip(), ToTensor()])
    val_data_transform = Compose([ToTensor()])

    train_dataset = EurosatDataset.from_fold(data_dir=root_dir, fold="train")
    train_dataset, val_dataset = train_dataset.split(n=0.7)
    test_dataset = EurosatDataset.from_fold(data_dir=root_dir, fold="test")

    train_dataset = TransformedDataset(dataset=train_dataset, transform=train_data_transform)
    val_dataset = TransformedDataset(dataset=val_dataset, transform=val_data_transform)
    test_dataset = TransformedDataset(dataset=test_dataset, transform=val_data_transform)

    return train_dataset, val_dataset, test_dataset


def get_data_loaders(train_dataset, val_dataset):

    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=BATCH_SIZE)

    val_loader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=BATCH_SIZE)

    return train_loader, val_loader


# %% [markdown]
# ## Model definition


# %%
# Define model to be trained
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(64, 96, kernel_size=3, padding=1), nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(4 * 4 * 96, 32), nn.ReLU())
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # 64x64x3
        x = F.max_pool2d(self.conv1(x), 2)  # 32x32x32
        x = F.max_pool2d(self.conv2(x), 2)  # 16x16x64
        x = F.max_pool2d(self.conv3(x), 2)  # 8x8x64
        x = F.max_pool2d(self.conv4(x), 2)  # 4x4x96
        x = F.dropout2d(x, training=self.training)
        x = x.view(-1, 4 * 4 * 96)
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


# %% [markdown]
# ## Training initialization

# %%
# Initialize datasets and models
train_dataset, val_dataset, test_dataset = get_datasets(root_dir="./eurosat")
train_loader, val_loader = get_data_loaders(train_dataset, val_dataset)
model = SimpleCNN()

device = 'cpu'

if torch.cuda.is_available():
    device = 'cuda'

# %%
# initialize training resources
optimizer = SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, nesterov=True)
trainer = ignite.engine.create_supervised_trainer(model, optimizer, F.nll_loss, device=device)
evaluator = ignite.engine.create_supervised_evaluator(
    model, metrics={
        'accuracy': ignite.metrics.Accuracy(),
        'nll': ignite.metrics.Loss(F.nll_loss)
    }, device=device)

desc = "ITERATION - loss: {:.2f}"
pbar = tqdm(initial=0, leave=False, total=len(train_loader), desc=desc.format(0))


@trainer.on(ignite.engine.Events.ITERATION_COMPLETED)
def log_training_loss(engine):
    iter = (engine.state.iteration - 1) % len(train_loader) + 1

    if iter % 100 == 0:
        pbar.desc = desc.format(engine.state.output)
        pbar.update(100)


@trainer.on(ignite.engine.Events.EPOCH_COMPLETED)
def log_training_results(engine):
    pbar.refresh()
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_nll = metrics['nll']
    tqdm.write("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
        engine.state.epoch, avg_accuracy, avg_nll))


@trainer.on(ignite.engine.Events.EPOCH_COMPLETED)
def log_validation_results(engine):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_nll = metrics['nll']
    tqdm.write("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
        engine.state.epoch, avg_accuracy, avg_nll))

    pbar.n = pbar.last_print_n = 0


# %% [markdown]
# ## Run training

# %%
# go !
trainer.run(train_loader, max_epochs=MAX_EPOCHS)
pbar.close()

# %% [markdown]
# ## Evaluate on test dataset (PR curve)

# %%
test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=BATCH_SIZE)

# %%
test_iterator = iter(test_loader)

# %%
all_y_preds = []
all_y_trues = []
for x, y_true in test_iterator:
    if torch.cuda.is_available():
        y_pred = F.softmax(model(x.cuda()), dim=-1).detach().cpu().numpy()
        y_true = y_true.cpu().numpy()
    else:
        y_pred = F.softmax(model(x), dim=-1).detach().numpy()
        y_true = y_true.numpy()
    y_true = np.eye(10)[y_true]
    all_y_trues.extend(y_true)
    all_y_preds.extend(y_pred)
y_pred = np.asarray(all_y_preds)
y_true = np.asarray(all_y_trues)

# %%
# Compute precision and recall for each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(10):
    precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
    average_precision[i] = average_precision_score(y_true[:, i], y_pred[:, i])

# %%
# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(y_true.ravel(), y_pred.ravel())
average_precision["micro"] = average_precision_score(y_true, y_pred, average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))

# %%
# Plot pr curve (code taken from sklearn)
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'red', 'yellow', 'purple', 'green'])
plt.figure(figsize=(7, 8))
f_scores = np.linspace(0.2, 0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.2f})' ''.format(average_precision["micro"]))
for i, color in zip(range(10), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                  ''.format(test_dataset.dataset.classes[i], average_precision[i]))
fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(lines, labels, loc=(0, -1.0), prop=dict(size=14))
plt.show()

# %%
# save model on google storage
from google.cloud import storage
torch.save(model.state_dict(), "./model.pt")

"""Uploads a file to the bucket."""
storage_client = storage.Client()
bucket = storage_client.get_bucket("fchouteau-storage")
blob = bucket.blob("model.pt")

blob.upload_from_filename("./model.pt")
