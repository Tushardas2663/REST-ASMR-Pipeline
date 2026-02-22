import pandas as pd
import numpy as np
import os
import glob
from scipy.signal import resample
from tqdm import tqdm



def load_ppg_file(filepath):
    
   
    encodings = ['shift_jis', 'utf-8', 'latin-1', 'cp1252']
    
    df = None
    for enc in encodings:
        try:
           
            df = pd.read_csv(filepath, sep='\t', skiprows=8, encoding=enc, on_bad_lines='skip')
            
            if df.shape[1] >= 3:
                
                df = df.iloc[:, [0, 2]] 
                df.columns = ['Time_Min', 'PPG_Val']   
                break
        except UnicodeDecodeError:
            continue
        except Exception as e:
           
            continue
            
    if df is None:
        print(f"CRITICAL ERROR: Could not decode {os.path.basename(filepath)}")
        return None

    try:
        
        df = df[pd.to_numeric(df['Time_Min'], errors='coerce').notnull()]
        
        
        df['Time_Sec'] = df['Time_Min'].astype(float) * 60.0  
        df['PPG_Val'] = df['PPG_Val'].astype(float) 
        
        
        return df
        
    except Exception as e:
        print(f"Error processing data in {os.path.basename(filepath)}: {e}")
        return None
def extract_ppg_features(output_folder,csv_path,ppg_folder,log_folder,target_hz=10,ppg_raw_hz=2000,buffer_sec=5):
    if not os.path.exists(output_folder):  
        os.makedirs(output_folder)
    if not os.path.exists(csv_path):
        print("Error: CSV not found. Run the log parser first.")
        return

    
    
    print("Building Timeline Lookup from Logs...")
    
    timeline_map = {} 
    
    log_files = glob.glob(os.path.join(log_folder, "*.log"))
    for log_f in log_files:
        try:
            sub_id = int(os.path.basename(log_f).split('-')[0])
           
            with open(log_f, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            header_idx = 0
            for i, line in enumerate(lines):
                if line.startswith("Subject"): header_idx = i; break
            
            ldf = pd.read_csv(log_f, sep='\t', skiprows=header_idx)
            vids = ldf[ldf['Event Type'] == 'Video'].reset_index(drop=True)
            
            for i, row in vids.iterrows():
                code = row['Code']
                start = row['Time']
                
                
                if i == 0: duration = 55.7
                else: duration = 60.0
                
                end = start + (duration * 10000)
                timeline_map[(sub_id, code)] = (start/10000.0, end/10000.0) 
                
        except: continue

    print(f"Timeline built for {len(timeline_map)} clips.")
   
    
    
    df_csv = pd.read_csv(csv_path)
    subjects = df_csv['Subject'].unique()
    print(subjects)
    c=0
    for sub in tqdm(subjects, desc="Processing Subjects"):
        
        print("Subject",sub,f"{str(sub).zfill(3)}_csv.txt")
        
        ppg_file = os.path.join(ppg_folder, f"{str(sub).zfill(3)}_csv.txt")
        
        if not os.path.exists(ppg_file):
            
            ppg_file = os.path.join(ppg_folder, f"{sub}_csv.txt")
            if not os.path.exists(ppg_file):
                print(f"Missing PPG for Subject {sub}")
                continue
                
        
        ppg_data = load_ppg_file(ppg_file)
        if ppg_data is None: continue
        
        
        sub_videos = df_csv[df_csv['Subject'] == sub]['VideoID'].unique()
       
        for vid in sub_videos:
            if (sub, vid) not in timeline_map:
                print("Wrong!!")
                continue
            
            abs_start, abs_end = timeline_map[(sub, vid)]
            
           
            valid_start = abs_start + buffer_sec
            valid_end = abs_end
           
            mask = (ppg_data['Time_Sec'] >= valid_start) & (ppg_data['Time_Sec'] < valid_end)
            ppg_clip = ppg_data.loc[mask, 'PPG_Val'].values
           
            if len(ppg_clip) == 0:
                print(f"Empty PPG for {sub} {vid}")
                continue
                
           
            target_duration = valid_end - valid_start
            target_len = int(np.round(target_duration * target_hz))
            
            
            
            if len(ppg_clip) > target_len:  
                resampled_ppg = resample(ppg_clip, target_len)
            else:
               
                resampled_ppg = np.pad(ppg_clip, (0, target_len - len(ppg_clip)))
           
           
            if np.std(resampled_ppg) > 0:
                resampled_ppg = (resampled_ppg - np.mean(resampled_ppg)) / np.std(resampled_ppg)
            else:
                resampled_ppg = resampled_ppg - np.mean(resampled_ppg)

            
            save_path = os.path.join(output_folder, f"P_{sub}_{vid}.npy")
            np.save(save_path, resampled_ppg.astype(np.float32))
        
if __name__=='__main__':
    extract_ppg_features(csv_path="./data/asmr_exp2_dataset.csv",
        ppg_folder="./data/ppg",
        output_folder="./data/features", log_folder="./data/log")