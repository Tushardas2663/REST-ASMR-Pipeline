import pandas as pd
import numpy as np
import glob
import os
import warnings


pd.options.mode.chained_assignment = None 




def parse_exp2_log(filepath,target_hz=10, buffer_sec=5):
    filename = os.path.basename(filepath)
    
    
    try:
        subject_str = filename.split('-')[0]
        subject_id = int(subject_str)
    except:
        return None

 
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    header_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("Subject"):
            
            header_idx = i
            break
            
    try:
        df = pd.read_csv(filepath, sep='\t', skiprows=header_idx)
    except:
        return None
    
    
    video_events = df[df['Event Type'] == 'Video'].copy().reset_index(drop=True)
    
    responses = df[df['Event Type'] == 'Response'].copy()
    
    responses['Code'] = pd.to_numeric(responses['Code'], errors='coerce')
    
    
    all_sequences = []
    
    
    for i, vid_row in video_events.iterrows():
        start_time_units = vid_row['Time']
        vid_code = vid_row['Code']
        
        
        
        
        
        if i == 0: 
            current_clip_duration = 55.7
        else:
            current_clip_duration = 60.0
            
        
        end_time_units = start_time_units + (current_clip_duration * 10000)
        
        
        
        try:
            vid_num = int(''.join(filter(str.isdigit, str(vid_code))))
            
            if 1 <= vid_num <= 4:
                content_type = 'ASMR'
                vid_filename = f"ASMR{vid_num}.avi"
            elif 5 <= vid_num <= 8:
                content_type = 'Nature'
                nature_num = vid_num - 4
                vid_filename = f"nature{nature_num}.avi"
            else:
                continue 
        except ValueError:
            continue

        
        effective_start_units = start_time_units + (buffer_sec * 10000)
        
        if effective_start_units >= end_time_units:
            continue 

        
        buffer_resps = responses[
            (responses['Time'] >= start_time_units) &   
            (responses['Time'] < effective_start_units)
        ].sort_values('Time')
        
       
        if not buffer_resps.empty:
            last_press = buffer_resps.iloc[-1]['Code']  
            current_state = int(max(0, min(last_press - 1, 3))) 
        else:
            current_state = 0

 
        clip_resps = responses[
            (responses['Time'] >= effective_start_units) & 
            (responses['Time'] < end_time_units)
        ].copy()
        
        
        
        clip_resps['Rel_Time_Sec'] = (clip_resps['Time'] - effective_start_units) / 10000.0
        clip_resps['Intensity'] = (clip_resps['Code'] - 1).clip(0, 3).fillna(0).astype(int)
        
        valid_duration = (end_time_units - effective_start_units) / 10000.0
        time_index = np.arange(0, valid_duration, 1/target_hz)  
        
        dense_labels = []
        resp_idx = 0
        sorted_resps = clip_resps.sort_values('Rel_Time_Sec').reset_index(drop=True)
        
        
        for t in time_index:  
            
            while resp_idx < len(sorted_resps) and sorted_resps.loc[resp_idx, 'Rel_Time_Sec'] <= t: 
                current_state = int(sorted_resps.loc[resp_idx, 'Intensity'])
                resp_idx += 1
            
            dense_labels.append({
                'Subject': subject_id,
                'Condition': content_type,
                'VideoID': vid_code,
                'VideoFile': vid_filename,
                'Time_Sec': round(t, 2),  
                'Intensity': current_state 
            })
            
            
        all_sequences.append(pd.DataFrame(dense_labels))
        
        
            
    if all_sequences:
       
        
        return pd.concat(all_sequences, ignore_index=True)
    else:
        return None

def process_all_logs(input_log_folder, output_csv_path, target_hz=10, buffer_sec=5):
    all_data = []
    log_files = glob.glob(os.path.join(input_log_folder, "*.log"))
    print(f"Found {len(log_files)} log files. Parsing with Index-Based Clamping...")
    
    for log_file in log_files:
        df = parse_exp2_log(log_file, target_hz=target_hz, buffer_sec=buffer_sec) 
        if df is not None:
            all_data.append(df)
            
    if all_data:
        master_df = pd.concat(all_data, ignore_index=True)
        master_df.to_csv(output_csv_path, index=False)
        print(f"SUCCESS! Saved dataset to: {output_csv_path}")
        print(f"Total Rows: {len(master_df)}") 
        #print(master_df)
    else:
        print("No data extracted.")

if __name__ == "__main__":
    process_all_logs(
        input_log_folder="./data/log", 
        output_csv_path="./data/asmr_exp2_dataset.csv"
    )
