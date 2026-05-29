import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import defaultdict

os.makedirs("results", exist_ok=True)

# ===================== 加载实验 1 原始数据 =====================
raw_path = "results/exp1_qwen_moe_20260529_015316_raw.pt"
raw = torch.load(raw_path, map_location="cpu")
distributions = raw["distributions"]  # {"Math": [60], "Law": [60], "Neutral": [60]}

domains = list(distributions.keys())
data = np.array([distributions[d] for d in domains]).T  # [60, 3]

report_lines = []
report_lines.append("=" * 60)
report_lines.append("实验 1 后处理三件套：聚类 + 纪念 + 脑波")
report_lines.append("=" * 60)

# ===================== 1. 专家群聚类（5 功能柱） =====================
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10).fit(data)
labels = kmeans.labels_

clusters = defaultdict(list)
for eid, cid in enumerate(labels):
    clusters[cid].append(eid)

report_lines.append("\n" + "=" * 60)
report_lines.append("一、专家群聚类（5 功能柱 = 大脑皮层功能柱映射）")
report_lines.append("=" * 60)

for cid in sorted(clusters.keys()):
    experts = clusters[cid]
    report_lines.append(f"\n功能柱 {cid}: 专家 {experts}")
    for domain in domains:
        mass = sum(distributions[domain][e] for e in experts)
        report_lines.append(f"  {domain} 激活质量: {mass:.2f}%")

# ===================== 2. 忘却的纪念（残余能量统计） =====================
report_lines.append("\n" + "=" * 60)
report_lines.append("二、忘却的纪念：58 个被遗忘专家的残余能量")
report_lines.append("=" * 60)

for domain in domains:
    dist = np.array(distributions[domain])
    sorted_idx = np.argsort(dist)[::-1]
    top2_mass = dist[sorted_idx[:2]].sum()
    forgotten_mass = dist[sorted_idx[2:]].sum()
    memorial_ratio = forgotten_mass / top2_mass if top2_mass > 0 else 0
    
    report_lines.append(f"\n{domain}:")
    report_lines.append(f"  Top-2 专家占据: {top2_mass:.2f}%")
    report_lines.append(f"  被遗忘 58 专家占据: {forgotten_mass:.2f}%")
    report_lines.append(f"  纪念比 (遗忘/选中): {memorial_ratio:.3f}")

# ===================== 3. 脑波能量场（Math 火山 vs Law 平原） =====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

math_grid = np.array(distributions["Math"]).reshape(6, 10)
im0 = axes[0].imshow(math_grid, cmap='hot', aspect='auto', interpolation='nearest')
axes[0].set_title('Math Expert Activation Field (Volcanic)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Expert Index (mod 10)')
axes[0].set_ylabel('Expert Index (div 10)')
plt.colorbar(im0, ax=axes[0], label='Activation %')

law_grid = np.array(distributions["Law"]).reshape(6, 10)
im1 = axes[1].imshow(law_grid, cmap='cool', aspect='auto', interpolation='nearest')
axes[1].set_title('Law Expert Activation Field (Plain)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Expert Index (mod 10)')
axes[1].set_ylabel('Expert Index (div 10)')
plt.colorbar(im1, ax=axes[1], label='Activation %')

plt.tight_layout()
plt.savefig("results/brainwave_math_vs_law.png", dpi=150, bbox_inches='tight')

report_lines.append("\n" + "=" * 60)
report_lines.append("三、脑波能量场热力图已保存")
report_lines.append("=" * 60)

# 落盘
with open("results/exp1_post_analysis_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("\n".join(report_lines))
print("\n[Saved] results/brainwave_math_vs_law.png")
print("[Saved] results/exp1_post_analysis_report.txt")