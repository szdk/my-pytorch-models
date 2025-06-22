import pandas as pd
import torch
import torch.nn as nn
import zipfile
import os
import requests
from tqdm.notebook import tqdm
import math

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
    train_split_perc,
    device
):
    model.train()
    for file in files:
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
          cur_sample = trainTensor[i-window_size:i]
          cur_batch_x.append(cur_sample.unsqueeze(0))
          cur_batch_y.append(trainTensor[i, -1].unsqueeze(0))
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
