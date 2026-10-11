import pandas as pd
import numpy as np
from pathlib import Path

def analyze_audit_results(csv_path):
    print(f"Analyzing {csv_path}...")
    df = pd.read_csv(csv_path)

    # 基础统计：计算各模态独立贡献
    # 如果去掉文本后概率下降，说明文本是正向贡献
    # 如果去掉文本后概率上升，说明文本是负向贡献（干扰）
    df['text_contribution'] = df['prob_full'] - df['prob_no_text']
    df['image_contribution'] = df['prob_full'] - df['prob_no_image']
    
    # 计算协同项 (Synergy)
    # 协同项 = 联合效果 - 各自独立效果之和（与均没有任何模态的基线比）
    # P_full - P_no_both: 两个模态加上去相比都不加带来的提升
    # (P_full - P_no_text): 图像单独的贡献
    # (P_full - P_no_image): 文本单独的贡献
    df['synergy'] = (df['prob_full'] - df['prob_no_both']) - \
                    (df['text_contribution'] + df['image_contribution'])
                    
    # 或者用更直观的公式：
    # synergy = P_full - (P_no_image + P_no_text - P_no_both)

    # 定义分类阈值
    TAU = 0.01  # 贡献 > 1% 认为有显著意义，可调

    def classify_sample(row):
        t_cont = row['text_contribution']
        i_cont = row['image_contribution']
        
        # 1. Distractive (干扰): 任意一个模态去掉后，结果显著更好 (贡献为负)
        if t_cont < -TAU or i_cont < -TAU:
            if t_cont < -TAU and i_cont < -TAU:
                return "Distractive (Both)"
            elif t_cont < -TAU:
                return "Distractive (Text)"
            else:
                return "Distractive (Image)"
                
        # 2. Synergistic (协同): 两者都有正向贡献，或者协同项显著大于0
        if (t_cont > TAU and i_cont > TAU) or row['synergy'] > TAU:
            return "Synergistic"
            
        # 3. Dominant (主导): 一个模态贡献显著，另一个几乎没啥用
        if t_cont > TAU and abs(i_cont) <= TAU:
            return "Dominant (Text)"
        if i_cont > TAU and abs(t_cont) <= TAU:
            return "Dominant (Image)"
            
        return "Neutral / Unclear"

    df['modality_type'] = df.apply(classify_sample, axis=1)

    # 统计饼图数据
    type_counts = df['modality_type'].value_counts()
    
    print("\n" + "="*50)
    print("🌟 样本模态行为分型统计 🌟")
    print("="*50)
    total_samples = len(df)
    for category, count in type_counts.items():
        percentage = (count / total_samples) * 100
        print(f"{category:<20}: {count:>4} samples ({percentage:.1f}%)")
        
    print("\n" + "="*50)
    print("📊 门控潜力分析 (Gating Potential) 📊")
    print("="*50)
    distractive_text_mask = df['modality_type'] == 'Distractive (Text)'
    if distractive_text_mask.sum() > 0:
        avg_improve = (-df[distractive_text_mask]['text_contribution']).mean()
        print(f"如果我们在推断时精准切断这 {distractive_text_mask.sum()} 个样本的文本模态，")
        print(f"它们预测真实类别的概率平均可以提升: {avg_improve:.4f}")
    
    distractive_image_mask = df['modality_type'] == 'Distractive (Image)'
    if distractive_image_mask.sum() > 0:
        avg_improve = (-df[distractive_image_mask]['image_contribution']).mean()
        print(f"如果我们在推断时精准切断这 {distractive_image_mask.sum()} 个样本的图像模态，")
        print(f"它们预测真实类别的概率平均可以提升: {avg_improve:.4f}")

    # 将带有分类结果的表格保存
    out_path = Path("analysis_results") / "mutag_audit_classified.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✅ 详细分类结果已保存至: {out_path}")

if __name__ == "__main__":
    csv_file = Path("analysis_results") / "mutag_audit_results.csv"
    if csv_file.exists():
        analyze_audit_results(csv_file)
    else:
        print(f"Error: {csv_file} does not exist. Run extract_sample_logits.py first.")
