import pandas as pd
import numpy as np
import cv2
import librosa
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
from tqdm import tqdm
import subprocess
import shutil
import soundfile as sf





def extract_features_dynamic(subject_id, vid_id, vid_filename, target_duration, 
                             video_folder, target_hz, buffer_sec, device, resnet, preprocess):
    video_path = os.path.join(video_folder, vid_filename)
    temp_wav = f"temp_{subject_id}_{vid_id}.wav"
    
    
    STEPS = int(target_duration * target_hz)  
    
    
    command = f"ffmpeg -i \"{video_path}\" -ab 160k -ac 1 -ar 44100 -vn \"{temp_wav}\" -y -hide_banner -loglevel error"
    subprocess.call(command, shell=True)
    
    audio_feats = None
    try:
       
        y, sr = librosa.load(temp_wav, sr=None, offset=buffer_sec, duration=target_duration)
        
        
        
        if os.path.exists(temp_wav): os.remove(temp_wav)
        hop_length = int(sr / target_hz)
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length).T 
        rms = librosa.feature.rms(y=y, hop_length=hop_length).T
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length).T
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length).T
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length).T
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length).T
        
        
        audio_feats = np.hstack([mfcc, rms, cent, bw, zcr, contrast])
      
        
        
        
        if audio_feats.shape[0] < STEPS:  
            pad_width = STEPS - audio_feats.shape[0]
            audio_feats = np.pad(audio_feats, ((0, pad_width), (0, 0)), mode='constant')
        else:
            audio_feats = audio_feats[:STEPS, :]
        
        
            
    except Exception as e:
        print(f"Audio Error {vid_filename}: {e}")
        audio_feats = np.zeros((STEPS, 31))
        if os.path.exists(temp_wav): os.remove(temp_wav)

    
    cap = cv2.VideoCapture(video_path) 
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_MSEC, buffer_sec * 1000) 
    
    batch_images = []
    frame_indices = [int(i / target_hz * fps) for i in range(STEPS)] 
    current_frame = 0
    collected_frames = 0
    
    while cap.isOpened():
        ret, frame = cap.read() 

        if not ret: break
        
        if current_frame in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            img = Image.fromarray(frame)
            batch_images.append(preprocess(img))
            collected_frames += 1
            if collected_frames >= STEPS: break
        
        current_frame += 1
    
    cap.release() 
    while len(batch_images) < STEPS:
        batch_images.append(torch.zeros(3, 224, 224))
        
    batch_tensor = torch.stack(batch_images).to(device)
    
    video_feats_list = []
    gpu_batch_size = 64
    
    with torch.no_grad():
        for i in range(0, len(batch_tensor), gpu_batch_size):
            batch = batch_tensor[i : i+gpu_batch_size]
            feats = resnet(batch).cpu().numpy()
            
            
           
            video_feats_list.append(feats.reshape(batch.shape[0], -1))
           
    video_feats = np.vstack(video_feats_list)
    
    return np.hstack([video_feats, audio_feats])





def run_extraction(csv_path, video_folder, output_folder, target_hz=10, buffer_sec=5):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
        print(f"Deleted old {output_folder}. Ready to re-run extraction.")
        os.makedirs(output_folder, exist_ok=True)
    else:
        print(f"{output_folder} does not exist yet.")
        os.makedirs(output_folder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing ResNet-18 model on {device}...")
    resnet = models.resnet18(pretrained=True)
    resnet = nn.Sequential(*list(resnet.children())[:-1]).to(device).eval()
    #print(resnet)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    df = pd.read_csv(csv_path)
    groups = df.groupby(['Subject', 'VideoID'])

   
    video_feature_cache = {} 

    print(f"Starting extraction for {len(groups)} sessions...") 

    for (sub, vid), group_data in tqdm(groups): 
        try:
            vid_filename = group_data['VideoFile'].iloc[0]
            
            max_time = group_data['Time_Sec'].max()
            target_duration = max_time + 0.1 
            
        except:
            print("Bad!!")
            continue
        
        
        save_name_x = os.path.join(output_folder, f"X_{sub}_{vid}.npy")
        save_name_y = os.path.join(output_folder, f"y_{sub}_{vid}.npy")
        
        if os.path.exists(save_name_x) and os.path.exists(save_name_y): 
            continue

        
        cache_key = (vid_filename, round(target_duration, 1))  
        
        if cache_key in video_feature_cache:
            features = video_feature_cache[cache_key].copy()
        else:
            features = extract_features_dynamic(sub, vid, vid_filename, target_duration, 
                video_folder, target_hz, buffer_sec, device, resnet, preprocess) 
            if features is not None:
                video_feature_cache[cache_key] = features.copy()

        if features is not None:
            
            group_data = group_data.sort_values('Time_Sec')
            
            labels = group_data['Intensity'].values
            
            
            if len(features) != len(labels):
                min_len = min(len(features), len(labels))
                features = features[:min_len]
                labels = labels[:min_len]
                
                
            np.save(save_name_x, features)   
            np.save(save_name_y, labels)
        else:
            print("Wrong!!")

    print("Feature Extraction Complete!")
if __name__ == "__main__":
    run_extraction(
        csv_path="./data/asmr_exp2_dataset.csv",
        video_folder="./data/stim",
        output_folder="./data/features"
    )