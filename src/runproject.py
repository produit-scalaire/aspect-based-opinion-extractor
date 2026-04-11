import json
import time
from statistics import mean
import pandas as pd
from accelerate import Accelerator

from tqdm import tqdm
from config import Config
import pyrallis
# from noft_extractor import OpinionExtractor
from ftlora_extractor import OpinionExtractor

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# PLEASE DO NOT EDIT/MODIFY THIS FILE



acc = Accelerator()

ASPECTS = ["Price", "Food", "Service"]

def load_data():
    df_train = pd.read_csv("../data/ftdataset_train.tsv", sep=' *\t *', encoding='utf-8', engine='python').to_dict(orient='records')
    df_val = pd.read_csv("../data/ftdataset_val.tsv", sep=' *\t *', encoding='utf-8', engine='python').to_dict(orient='records')
    try:
        df_test = pd.read_csv("../data/ftdataset_test.tsv", sep=' *\t *', encoding='utf-8', engine='python').to_dict(orient='records')
    except:
        df_test = None
    return (
        df_train, 
        df_val, 
        df_test if df_test is not None else None
    )

def eval(preds: list[dict], eval_data: list[dict]) -> dict[str,float]:
    n = len(eval_data)
    correct_counts = {aspect: 0.0 for aspect in ASPECTS}
    for pred, ref in zip(preds, eval_data):
        if pred is None:
            continue
        for aspect in ASPECTS:
            if aspect in pred and pred[aspect] == ref[aspect]:
                correct_counts[aspect] += 1
    for aspect in correct_counts:
        correct_counts[aspect] = round(100*correct_counts[aspect]/n, 2)
    macro_acc = round(sum(acc for acc in correct_counts.values())/len(ASPECTS), 2)
    correct_counts['macro_acc'] = macro_acc
    return correct_counts


def run_project(cfg: Config):
    train_data, val_data, test_data = load_data()
    eval_data = test_data if test_data else val_data
    if cfg.n_train > 0:
        train_data = train_data[:cfg.n_train]
    if cfg.n_eval > 0:
        eval_data = eval_data[:cfg.n_eval]
    eval_texts = [element['Review'] for element in eval_data]

    if OpinionExtractor.method == "FT":
        n = cfg.n_runs
    else:
        n = 1

    all_runs_acc = []
    for run_id in range(1, n+1):
        print(f"RUN {run_id}/{cfg.n_runs}")
        extractor = OpinionExtractor(cfg)
        if extractor.method == "FT":
            print("Training...")
            extractor.train(train_data, val_data)
        if acc.is_main_process:
            # Evaluate only in the main process:
            print("Evaluation...")
            preds = []
            for start_idx in tqdm(range(0, len(eval_texts), cfg.eval_batch_size)):
                batch_preds = extractor.predict(eval_texts[start_idx:start_idx+cfg.eval_batch_size])
                preds.extend(batch_preds)
            accuracies = eval(preds, eval_data)
            all_runs_acc.append(accuracies['macro_acc'])
            print(f"\nRUN{run_id}:", accuracies)
    if acc.is_main_process:
        print("\nALL RUNS ACC:", all_runs_acc)
        avg_acc = round(mean(all_runs_acc), 2)
        print("AVG MACRO ACC:", avg_acc)



if __name__ == "__main__":
    cfg = pyrallis.parse(config_class=Config)
    cfg_str = json.dumps(vars(cfg), sort_keys=False, indent=3)
    acc.print(cfg_str)
    start_time = time.perf_counter()
    run_project(cfg)
    total_exec_time = round(time.perf_counter() - start_time, 1)
    acc.print("TOTAL EXEC TIME:", total_exec_time)
    


