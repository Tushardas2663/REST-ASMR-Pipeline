import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os
import random
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings


warnings.filterwarnings("ignore")




class VideoOnlyDataset(Dataset):
    def __init__(self, x_paths,input_dim=512, target_len=550):
        self.samples = []
        self.labels = []
        self.nature_flags = []
        
        self.target_len =target_len
        
        for x_path in x_paths:
           
            x_full = np.load(x_path).astype(np.float32)
            
            x_data = x_full[:, :512]                              
            
            
            
       
            y_path = x_path.replace("X_", "y_")
            filename = os.path.basename(x_path)
            parts = filename.replace("X_", "").replace(".npy", "").split("_")
            vid_id = parts[1]
            is_nature = 1 if vid_id in ["vid5", "vid6", "vid7", "vid8"] else 0
            
            if os.path.exists(y_path):
                y_raw = np.load(y_path)
                label_seq = np.zeros_like(y_raw, dtype=np.int64) if is_nature else (y_raw > 0).astype(np.int64)
            else:
                
                print("Wrong!!")
                label_seq = np.zeros(x_data.shape[0], dtype=np.int64)
                
            
            
            
            curr_x = x_data.shape[0]
            if curr_x > target_len:
                x_data = x_data[:target_len, :]
            elif curr_x < target_len:
                pad_amt = target_len - curr_x
                
                x_data = np.vstack([x_data, np.zeros((pad_amt, input_dim), dtype=np.float32)])
                
           
            curr_y = len(label_seq)
            if curr_y > target_len:
                label_seq = label_seq[:target_len]
            elif curr_y < target_len:
                pad_amt = target_len - curr_y
                label_seq = np.pad(label_seq, (0, pad_amt), mode='constant')
                
            self.samples.append(x_data)
            self.labels.append(label_seq)
            self.nature_flags.append(is_nature)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.samples[idx]), 
                torch.tensor(self.labels[idx], dtype=torch.long), 
                self.nature_flags[idx])


class VideoPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(VideoPredictor, self).__init__()
    
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
     
        self.fc_out = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        logits = self.fc_out(lstm_out) 
        return logits


def run_video_only_experiment(data_folder, input_dim=512, hidden_dim=128, num_classes=2, 
                               batch_size=16, epochs=35, lr=0.0005, target_len=550):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_x_files = sorted(glob.glob(os.path.join(data_folder, "X_*.npy")))
    
    if len(all_x_files) == 0:
        print("Error: No feature files found.")
        return

  
    X_files = np.array(all_x_files)
    
    
    asmr_test_map = {0: "vid1", 1: "vid2", 2: "vid3", 3: "vid4"}
    nature_test_map = {0: "vid5", 1: "vid6", 2: "vid7", 3: "vid8"}
    
    subjects = sorted(list(set([os.path.basename(f).split('_')[1] for f in X_files])))
    
    print(f"Starting VIDEO-ONLY Ablation Experiment on {len(subjects)} subjects...")
    print(f"Input Dimension: {input_dim} (ResNet 512 Only)")
    
    metrics_history = {
        'accuracy': [],
        'macro_f1': [],
        'class1_f1': [],
        'class0_f1': [],
        'nature_spec': [],
        'precision':[],
        'recall':[]
    }

    for fold in range(4):
       
        num_test_subs = len(subjects) // 4
        
        start_idx = fold * num_test_subs
        end_idx = (fold + 1) * num_test_subs if fold < 3 else len(subjects) 
        test_sub_pool = subjects[start_idx : end_idx]
        
        print()
        print(f"        FOLD {fold+1} / 4        ")
        print(f" TEST VIDEOS    : {asmr_test_map[fold]}, {nature_test_map[fold]} ")
        print(f" TEST SUBJECTS : {len(test_sub_pool)} subjects") 
        print()
        
        train_pool_files = []
        test_files = []

        
        for f in X_files:
            fname = os.path.basename(f)
            parts = fname.replace("X_", "").replace(".npy", "").split("_")
            subj = parts[0]
            vid = parts[1]
            
            is_test_video = (vid == asmr_test_map[fold] or vid == nature_test_map[fold])
            is_test_subject = (subj in test_sub_pool)
            
            if is_test_subject and is_test_video:
                test_files.append(f)
            elif not is_test_subject and not is_test_video:
                train_pool_files.append(f)

        
        train_pool_subjects = sorted(list(set([os.path.basename(f).split('_')[1] for f in train_pool_files])))
        random.seed(42) 
        num_inner_val = max(1, int(len(train_pool_subjects) * 0.1)) 
        inner_val_subs = random.sample(train_pool_subjects, num_inner_val)
        
        inner_train_files = []
        inner_val_files = []
        
        for f in train_pool_files:
            subj = os.path.basename(f).split('_')[1]
            if subj in inner_val_subs:
                inner_val_files.append(f)
            else:
                inner_train_files.append(f)

        print(f"  > Inner Train : {len(inner_train_files)}")
        print(f"  > Inner Valid : {len(inner_val_files)}")
        print(f"  > Final Test  : {len(test_files)}")
        
        
        train_ds = VideoOnlyDataset(inner_train_files,input_dim=input_dim,target_len=target_len)
        val_ds = VideoOnlyDataset(inner_val_files,input_dim=input_dim,target_len=target_len)
        test_ds = VideoOnlyDataset(test_files,input_dim=input_dim,target_len=target_len)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        
        model = VideoPredictor(input_dim, hidden_dim, num_classes).to(device)

      
        all_train_labels = np.concatenate(train_ds.labels)
        class_counts = np.bincount(all_train_labels)
        total_samples = len(all_train_labels)
        weights = total_samples / (2.0 * class_counts)
        weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
        
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        best_f1 = 0.0
        best_state = None
        
        
        for epoch in range(epochs):
            model.train()
            for X_b, y_b, _ in train_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                out = model(X_b)
                loss = criterion(out.view(-1, 2), y_b.view(-1))
                loss.backward()
                optimizer.step()
                
            # Validation
            model.eval()
            preds, truths = [], []
            with torch.no_grad():
                for X_b, y_b, _ in val_loader:
                    X_b, y_b = X_b.to(device), y_b.to(device)
                    out = model(X_b)
                    preds.extend(torch.argmax(out, dim=2).view(-1).cpu().numpy())
                    truths.extend(y_b.view(-1).cpu().numpy())
            
            curr_f1 = f1_score(truths, preds, average='macro', zero_division=0)
            if curr_f1 > best_f1:
                best_f1 = curr_f1
                best_state = model.state_dict()
                
        # Test
        if best_state is not None:
            model.load_state_dict(best_state)
            
        final_preds, final_truths = [], []
        nature_only_preds, nature_only_truths = [], []
        
        model.eval()
        with torch.no_grad():
            for X_b, y_b, is_nat_batch in test_loader:
                X_b = X_b.to(device)
                out = model(X_b)
                batch_preds = torch.argmax(out, dim=2).cpu().numpy()
                batch_truths = y_b.numpy()
                
                final_preds.extend(batch_preds.flatten())
                final_truths.extend(batch_truths.flatten())
                
                for i in range(len(is_nat_batch)):
                    if is_nat_batch[i] == 1:
                        nature_only_preds.extend(batch_preds[i].flatten())
                        nature_only_truths.extend(batch_truths[i].flatten())

        # Reporting
        report_dict = classification_report(final_truths, final_preds, output_dict=True, zero_division=0)
        print(f"  > Fold Macro F1: {report_dict['macro avg']['f1-score']:.4f}")

        print(report_dict)

        metrics_history['accuracy'].append(report_dict['accuracy'])
        metrics_history['macro_f1'].append(report_dict['macro avg']['f1-score'])
        metrics_history['class1_f1'].append(report_dict['1']['f1-score'] if '1' in report_dict else 0)
        metrics_history['class0_f1'].append(report_dict['0']['f1-score'] if '0' in report_dict else 0)
        metrics_history['precision'].append(report_dict['macro avg']['precision'])
        metrics_history['recall'].append(report_dict['macro avg']['recall'])
        fold_nat_spec = accuracy_score(nature_only_truths, nature_only_preds) if nature_only_truths else 1.0
        metrics_history['nature_spec'].append(fold_nat_spec)

    print()
    print(f"      FINAL VIDEO-ONLY SUMMARY (4-FOLD)      ")
    print()
    
    def p_stat(name, key):
        mean_val = np.mean(metrics_history[key])
        std_val = np.std(metrics_history[key])
        print(f"{name:<25} : {mean_val:.4f} (+/- {std_val:.4f})")

    p_stat("Global Accuracy", 'accuracy')
    p_stat("Global Macro F1", 'macro_f1')
    p_stat("Global Macro Precision", 'precision')
    p_stat("Global Macro Recall", 'recall')
   
    p_stat("Nature Specificity", 'nature_spec')
    print()
    print()

if __name__ == "__main__":
    run_video_only_experiment(data_folder="./data/features")