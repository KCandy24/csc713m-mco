import os
import cv2
import numpy as np
from tqdm import tqdm

def load_lung_disease_dataset(base_path, img_size=(224, 224), clahe_on=True, flatten=True):
    data = []
    labels = []
    
    if not os.path.exists(base_path):
        print(f"Error: Path {base_path} not found.")
        return None, None
        
    class_names = sorted([f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))])
    
    if clahe_on:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    # Iterate over every folder
    for idx, label in enumerate(class_names):
        folder_path = os.path.join(base_path, label)
        files = os.listdir(folder_path)
        
        # Load each image then process via imread
        for img_name in tqdm(files, desc=f"Loading {label}", leave=False):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
            
            # Pre process through resizing and CLAHE
            if clahe_on:
                img = clahe.apply(img)
                
            img = cv2.resize(img, img_size)
            
            if flatten:
                data.append(img.flatten() / 255.0)
            else:
                data.append(img / 255.0)
            
            labels.append(idx)
            
    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.int32)