import os
import subprocess
import numpy as np
import pandas as pd
import re

def main():
    # Only test a few thresholds to save time since Explainer optimization is slow per node
    # You can test the same range if you have time.
    datasets = ["mutag", "nlm_chem", "pqaref"] # Added all datasets
    taus = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    results_data = []
    
    print("🚀 启动路线B - 解释器引导门控 (Explainer-Guided Gating) 评估")
    print("-" * 60)
    
    for dataset in datasets:
        ckpt_path = f"ckpt/{dataset}_base_h20_o20.pth.tar"
        
        if not os.path.exists(ckpt_path):
            print(f"⚠️ 找不到检查点 {ckpt_path}，将跳过 {dataset}。")
            continue
            
        print(f"\n========== 🎯 数据集: {dataset.upper()} ==========")
        baseline_recorded = False
        baseline_acc = 0.0
        
        for tau in taus:
            cmd = [
                "python", "evaluate_explainer_gating.py",
                "--dataset", dataset,
                "--gpu",
                "--cuda", "0",
                "--ckpt-path", ckpt_path,
                "--text-tau", str(tau),
                "--explainer-epochs", "10" 
            ]
            print(f"  [Explainer 推断中] \u03C4 = {tau:.1f} ... ", end="", flush=True)
            
            try:
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                output = res.stdout
                
                # Fetch output metrics
                ours_match = re.search(r"Adaptive Explainer Gating Accuracy:\s*([0-9.]+)", output, re.IGNORECASE)
                if not ours_match:
                    ours_match = re.search(r"OURS[^:]*:\s*([0-9.]+)", output, re.IGNORECASE)
                    
                baseline_match = re.search(r"Baseline[^:]*:\s*([0-9.]+)", output, re.IGNORECASE)
                
                ours_acc = float(ours_match.group(1)) if ours_match else None
                if not baseline_recorded and baseline_match:
                    baseline_acc = float(baseline_match.group(1))
                    baseline_recorded = True
                
                if ours_acc is not None:
                    print(f"Route B Acc: {ours_acc:.4f}", end="")
                else:
                    print("Route B Acc: N/A (可能报错了)", end="")
                    
                if baseline_recorded:
                    print(f" | Baseline Acc: {baseline_acc:.4f}")
                else:
                    print()
                    
                results_data.append({
                    "Dataset": dataset,
                    "Tau": tau,
                    "Route_B_Accuracy": ours_acc,
                    "Baseline_Accuracy": baseline_acc
                })
                
            except Exception as e:
                print(f"❌ 运行失败: {e}")

    df = pd.DataFrame(results_data)
    os.makedirs("analysis_results", exist_ok=True)
    csv_path = "analysis_results/route_b_sensitivity.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ 路线B数据已导出至: {csv_path}")

if __name__ == "__main__":
    main()

