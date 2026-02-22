import numpy as np
import glob
import os
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_recall_fscore_support
import warnings

warnings.filterwarnings("ignore")



def load_and_flatten_data(file_paths, is_training=True,target_len=550,input_dim=544):
    
    X_list, y_list, nature_list, subj_list = [], [], [], []
    
    for x_path in file_paths:
        
        x_data = np.load(x_path).astype(np.float32)
        x_data[:, 533] = np.log1p(x_data[:, 533])
        x_data[:, 534] = np.log1p(x_data[:, 534])
        
       
        p_path = x_path.replace("X_", "P_")
        p_data = np.load(p_path).astype(np.float32).reshape(-1, 1) if os.path.exists(p_path) else np.zeros((x_data.shape[0], 1), dtype=np.float32)
        
        
        min_len = min(len(x_data), len(p_data))
        fused_data = np.hstack([x_data[:min_len], p_data[:min_len]])
        
        
        filename = os.path.basename(x_path)
        parts = filename.replace("X_", "").replace(".npy", "").split("_")
        subj_id = int(parts[0])
        vid_id = parts[1]
        is_nature = 1 if vid_id in ["vid5", "vid6", "vid7", "vid8"] else 0
        
       
        y_path = x_path.replace("X_", "y_")
        if os.path.exists(y_path):
            y_raw = np.load(y_path)
            label_seq = np.zeros_like(y_raw, dtype=np.int64) if is_nature else (y_raw > 0).astype(np.int64)
        else:
            label_seq = np.zeros(min_len, dtype=np.int64)
            
      
        curr_x = fused_data.shape[0]
        if curr_x > target_len:
            fused_data = fused_data[:target_len, :]
        elif curr_x < target_len:
            fused_data = np.vstack([fused_data, np.zeros((target_len - curr_x, input_dim), dtype=np.float32)])
            
        curr_y = len(label_seq)
        if curr_y > target_len:
            label_seq = label_seq[:target_len]
        elif curr_y < target_len:
            label_seq = np.pad(label_seq, (0, target_len - curr_y), mode='constant')
            
        X_list.append(fused_data)
        y_list.append(label_seq)
        nature_list.extend([is_nature] * target_len)
        subj_list.extend([subj_id] * target_len)
        
    
    if not X_list:
        return np.array([]), np.array([]), np.array([]), np.array([])
        
    X_flat = np.vstack(X_list)
    y_flat = np.concatenate(y_list)
    return X_flat, y_flat, np.array(nature_list), np.array(subj_list)

def run_xgboost_baseline(data_folder,input_dim=544,target_len=550):
    all_x_files = sorted(glob.glob(os.path.join(data_folder, "X_*.npy")))
    X_files = np.array(all_x_files)
    
    asmr_test_map = {0: "vid1", 1: "vid2", 2: "vid3", 3: "vid4"}
    nature_test_map = {0: "vid5", 1: "vid6", 2: "vid7", 3: "vid8"}
    subjects = sorted(list(set([os.path.basename(f).split('_')[1] for f in X_files])))
    
    print(f"Starting XGBoost Baseline on {len(subjects)} subjects...")
    
    metrics_history = {'accuracy': [], 'macro_f1': [], 'precision': [], 'recall': [], 'nature_spec': []}
    
    for fold in range(4):
        num_test_subs = len(subjects) // 4
        start_idx = fold * num_test_subs
        end_idx = (fold + 1) * num_test_subs if fold < 3 else len(subjects) 
        test_sub_pool = subjects[start_idx : end_idx]
        
        print()
        print(f"      FOLD {fold+1} / 4 (XGBoost)      ")
        print()
        
        train_files, test_files = [], []
        
        for f in X_files:
            parts = os.path.basename(f).replace("X_", "").replace(".npy", "").split("_")
            subj, vid = parts[0], parts[1]
            if (subj in test_sub_pool) and (vid == asmr_test_map[fold] or vid == nature_test_map[fold]):
                test_files.append(f)
            elif not (subj in test_sub_pool) and not (vid == asmr_test_map[fold] or vid == nature_test_map[fold]):
                train_files.append(f)

        
        X_train, y_train, _, _ = load_and_flatten_data(train_files, is_training=True, input_dim=input_dim,target_len=target_len)
        X_test, y_test, nat_test, _ = load_and_flatten_data(test_files, is_training=False)
        
        
        num_neg = np.sum(y_train == 0)
        num_pos = np.sum(y_train == 1)
        scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
        
        
        clf = xgb.XGBClassifier(
            n_estimators=100, 
            max_depth=6, 
            learning_rate=0.1, 
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        
        
        report_dict = classification_report(y_test, preds, output_dict=True, zero_division=0)
        print(f"  > Fold Macro F1: {report_dict['macro avg']['f1-score']:.4f}")
        
        metrics_history['accuracy'].append(report_dict['accuracy'])
        metrics_history['macro_f1'].append(report_dict['macro avg']['f1-score'])
        metrics_history['precision'].append(report_dict['macro avg']['precision'])
        metrics_history['recall'].append(report_dict['macro avg']['recall'])
        
        
        nature_mask = (nat_test == 1)
        if np.sum(nature_mask) > 0:
            nat_spec = accuracy_score(y_test[nature_mask], preds[nature_mask])
        else:
            nat_spec = 1.0
        metrics_history['nature_spec'].append(nat_spec)

    print()
    print("      FINAL XGBOOST SUMMARY (4-FOLD)      ")
    print()
    for key in ['accuracy', 'macro_f1', 'precision', 'recall', 'nature_spec']:
        mean_val = np.mean(metrics_history[key])
        std_val = np.std(metrics_history[key])
        print(f"{key.replace('_', ' ').title():<22} : {mean_val:.4f} (+/- {std_val:.4f})")

if __name__ == "__main__":
    run_xgboost_baseline(data_folder="./data/features")