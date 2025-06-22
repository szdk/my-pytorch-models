import os
import requests

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
