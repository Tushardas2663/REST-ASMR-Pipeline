import os
import warnings


from pipeline.log_parser import process_all_logs
from pipeline.feature_extractor import run_extraction
from pipeline.ppg_processor import extract_ppg_features
from pipeline.fusion_bilstm import run_full_fusion_experiment
from pipeline.video_only_ablation import run_video_only_experiment
from pipeline.audio_only_ablation import run_audio_only_experiment
from pipeline.xgboost_baseline import run_xgboost_baseline

warnings.filterwarnings("ignore")


RAW_LOG_DIR   = "./data/log"
RAW_STIM_DIR  = "./data/stim"
RAW_PPG_DIR   = "./data/ppg"
CSV_PATH      = "./data/asmr_exp2_dataset.csv"
FEATURES_DIR  = "./data/features"


RUN_PARSER       = True   # Step 1: Parse behavioral logs
RUN_EXTRACTION   = True   # Step 2: Extract ResNet & Librosa features
RUN_PPG          = True   # Step 3: Process Physiological data
RUN_FUSION       = True   # Step 4a: Train Full Multimodal BiLSTM
RUN_VIDEO_ONLY   = True   # Step 4b: Train Video-Only Ablation
RUN_AUDIO_ONLY   = True   # Step 4c: Train Audio-Only Ablation
RUN_XGBOOST      = True   # Step 4d: Train Tabular XGBoost Baseline

if __name__ == "__main__":
    print()
    print("      REST-ASMR MULTIMODAL PIPELINE ORCHESTRATOR      ")
    print()
    
    
    os.makedirs(FEATURES_DIR, exist_ok=True)

    
    if RUN_PARSER:
        print("\n[STEP 1] Running Behavioral Log Parser...")
        process_all_logs(
            input_log_folder=RAW_LOG_DIR, 
            output_csv_path=CSV_PATH
        )
    else:
        print("\n[STEP 1] Skipping Log Parser...")

    
    if RUN_EXTRACTION:
        print("\n[STEP 2] Running Audiovisual Feature Extraction...")
        run_extraction(
            csv_path=CSV_PATH, 
            video_folder=RAW_STIM_DIR, 
            output_folder=FEATURES_DIR
        )
    else:
        print("\n[STEP 2] Skipping Audiovisual Extraction...")


    if RUN_PPG:
        print("\n[STEP 3] Running PPG Signal Processing...")
        extract_ppg_features(
            output_folder=FEATURES_DIR, 
            csv_path=CSV_PATH, 
            ppg_folder=RAW_PPG_DIR, 
            log_folder=RAW_LOG_DIR
        )
    else:
        print("\n[STEP 3] Skipping PPG Processing...")

    
    print("\n" + "="*60)
    print("             STARTING MODEL EVALUATIONS               ")
    print("="*60)

    if RUN_FUSION:
        print("\n>>> EVALUATING FULL FUSION BILSTM <<<")
        run_full_fusion_experiment(data_folder=FEATURES_DIR)

    if RUN_VIDEO_ONLY:
        print("\n>>> EVALUATING VIDEO-ONLY ABLATION <<<")
        run_video_only_experiment(data_folder=FEATURES_DIR)

    if RUN_AUDIO_ONLY:
        print("\n>>> EVALUATING AUDIO-ONLY ABLATION <<<")
        run_audio_only_experiment(data_folder=FEATURES_DIR)

    if RUN_XGBOOST:
        print("\n>>> EVALUATING TABULAR XGBOOST BASELINE <<<")
        run_xgboost_baseline(data_folder=FEATURES_DIR)

    print("\n" + "="*60)
    print("               PIPELINE EXECUTION COMPLETE            ")
    print("="*60)