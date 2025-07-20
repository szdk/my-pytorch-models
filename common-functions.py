import pandas as pd
import torch
import torch.nn as nn
import zipfile
import os
import requests
from tqdm.notebook import tqdm
import math
import random

def extractZip(zip_path, extract_to):
  if not os.path.isdir(extract_to):
    print("Extracting zip")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete.")
  else:
    print("Directory already exists.")

def loadFiles(directory):
  file_list = [
      os.path.join(directory, f)
      for f in os.listdir(directory)
      if os.path.isfile(os.path.join(directory, f))
  ]
  return file_list

def downloadRemote(remote_url, download_to):
  response = requests.get(remote_url)
  with open(download_to, "w") as f:
      f.write(response.text)
  return download_to


def trainModel(
    files,
    model,
    criterion,
    optimizer,
    batch_size,
    window_size,
    output_size,
    train_split_perc,
    device
):
    model.train()
    for file in tqdm(files):
        df = pd.read_csv(file).iloc[:, 1:]
        fullTensor = torch.tensor(df.values).float().to(device)
        trainTensor = fullTensor[:math.floor(len(fullTensor)*train_split_perc)]
        print(f"\nLoaded {df.shape} from {file}")
        print(f"fullTensor: {fullTensor.shape}, trainTensor: {trainTensor.shape}")
        cur_batch_x = []
        cur_batch_y = []
        batches_trained = 0

        end = len(trainTensor)
        pbar = tqdm(total=(end-window_size))
        for i in range(window_size, end):
          cur_sample = trainTensor[i-window_size:i, :6]
          cur_batch_x.append(cur_sample.unsqueeze(0))
          cur_batch_y.append(trainTensor[i, -output_size:])
          if len(cur_batch_x) == batch_size or i == end-1:
            batches_trained += 1
            X = torch.stack(cur_batch_x).to(device)
            Y = torch.stack(cur_batch_y).to(device)
            outputs = model(X)
            loss = criterion(outputs, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cur_batch_x = []
            cur_batch_y = []
            pbar.update(batch_size)
            print(f"\rLoss: {loss.item():.4f}", end="")
        pbar.close()


def normalize_batchwise(X, feature_range=(0.0, 1.0)):
    # X shape: [batch_size, 1, 1024, 6]
    
    # Keep dims for broadcasting: reduce dims 1,2,3 (channel, time, feature)
    min_vals = X.amin(dim=(1, 2, 3), keepdim=True)  # shape: [batch_size, 1, 1, 1]
    max_vals = X.amax(dim=(1, 2, 3), keepdim=True)  # shape: [batch_size, 1, 1, 1]
    
    scale = feature_range[1] - feature_range[0]
    X_norm = (X - min_vals) / (max_vals - min_vals + 1e-8)
    X_norm = X_norm * scale + feature_range[0]
    
    return X_norm



def getones(window_size, trainTensor):
    ones = []
    for i in range(window_size, len(trainTensor)):
        if (trainTensor[i, -1] > 0.9):
            ones.append(trainTensor[i-window_size:i, :6].unsqueeze(0))
    return ones

def getonesAll(window_size, files, train_split_perc, device):
   print(f"Getting 1.0s from {len(files)} files")
   result = []
   for file in files:
        df = pd.read_csv(file).iloc[:, 1:]
        fullTensor = torch.tensor(df.values).float().to(device)
        trainTensor = fullTensor[:math.floor(len(fullTensor)*train_split_perc)]
        result.extend(getones(window_size, trainTensor))
   print(f"Got {len(result)} 1.0s from {len(files)} files")
   return result
        


#equalizes by signal (single predection)
def trainModelBalancedSingle(
    files,
    model,
    criterion,
    optimizer,
    batch_size,
    window_size,
    train_split_perc,
    device
):
    model.train()
    ones = getonesAll(window_size, files, train_split_perc, device)
    cursorOne = 0
    for file in tqdm(files):
        df = pd.read_csv(file).iloc[:, 1:]
        fullTensor = torch.tensor(df.values).float().to(device)
        trainTensor = fullTensor[:math.floor(len(fullTensor)*train_split_perc)]
        print(f"\nLoaded {df.shape} from {file}")
        print(f"fullTensor: {fullTensor.shape}, trainTensor: {trainTensor.shape}")
        cur_batch_x = []
        cur_batch_y = []
        batches_trained = 0

        end = len(trainTensor) - 1
        pbar = tqdm(total=(end-window_size))
        i = window_size - 1
        while i < end:
          i += 1
          if (trainTensor[i, -1].item() > 0.9):
             continue
          if (random.random() < 0.5):
            cur_sample = trainTensor[i-window_size:i, :6].unsqueeze(0)
            cur_batch_y.append(torch.tensor([0.0], device=device, dtype=torch.float32))
          else:
            cur_sample = ones[cursorOne % len(ones)].clone()
            cursorOne += 1
            cur_batch_y.append(torch.tensor([1.0], device=device, dtype=torch.float32))
            i -= 1
          cur_batch_x.append(cur_sample)
          if len(cur_batch_x) == batch_size or i == end-1:
            batches_trained += 1
            X = torch.stack(cur_batch_x).to(device)
            Y = torch.stack(cur_batch_y)
            outputs = model(normalize_batchwise(X))
            loss = criterion(outputs, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(len(cur_batch_x))
            cur_batch_x = []
            cur_batch_y = []
            print(f"\rLoss: {loss.item():.4f}", end="")
        pbar.close()
