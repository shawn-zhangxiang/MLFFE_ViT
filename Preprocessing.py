import torch
import cv2 
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm.notebook import tqdm
from pathlib import Path

data_path = Path("Path")

image_path_list = list(data_path.glob("*.mp4"))
mtcnn = MTCNN(margin=40, keep_all=True, post_process=False, device='device')

for k in range(0,10):
    
    dir_path=f'{image_path_list[k]}'
    v_cap = cv2.VideoCapture(dir_path)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    total_faces=[]
    
    for i in tqdm(range(v_len)):
    
        success = v_cap.grab()
        if i % 5 == 0:
            success, frame = v_cap.retrieve()    
        else:
            continue
        if not success:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))

    for i in range(len(frames)):
        save_paths = [f'save_dir']
        mtcnn(frames[i], save_path=save_paths)
  
