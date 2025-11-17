import numpy as np

# --- 1. 定义数据 ---
# 样本1: (密度=0.697, 含糖率=0.460)
# 样本2: (密度=0.774, 含糖率=0.376)
# 样本3: (密度=0.634, 含糖率=0.264)
s1 = np.array([0.697, 0.460])
s2 = np.array([0.774, 0.376])
s3 = np.array([0.634, 0.264])

# 闵可夫斯基距离 (Minkowski distance) L_p，在 numpy 中
# 可以使用 np.linalg.norm(x - y, ord=p) 来计算。
# ord=p 参数即为闵可夫斯基距离中的参数 p。

print("--- 第2题计算开始 ---")
print("\n样本1 和 样本2 之间的闵可夫斯基距离:")
# (1) p = 1 (曼哈顿距离)
p = 1
dist_1_2_p1 = np.linalg.norm(s1 - s2, ord=p)
print(f"  p = {p}: {dist_1_2_p1:.4f}")  # :.4f 表示保留4位小数

# (2) p = 2 (欧氏距离)
p = 2
dist_1_2_p2 = np.linalg.norm(s1 - s2, ord=p)
print(f"  p = {p}: {dist_1_2_p2:.4f}")

# (3) p = 3
p = 3
dist_1_2_p3 = np.linalg.norm(s1 - s2, ord=p)
print(f"  p = {p}: {dist_1_2_p3:.4f}")


print("\n样本1 和 样本3 之间的闵可夫斯基距离:")
# (1) p = 1 (曼哈顿距离)
p = 1
dist_1_3_p1 = np.linalg.norm(s1 - s3, ord=p)
print(f"  p = {p}: {dist_1_3_p1:.4f}")

# (2) p = 2 (欧氏距离)
p = 2
dist_1_3_p2 = np.linalg.norm(s1 - s3, ord=p)
print(f"  p = {p}: {dist_1_3_p2:.4f}")

# (3) p = 3
p = 3
dist_1_3_p3 = np.linalg.norm(s1 - s3, ord=p)
print(f"  p = {p}: {dist_1_3_p3:.4f}")

print("\n--- 结论分析 ---")
# 比较 Dist(1,2) 和 Dist(1,3)
print(f"当 p=1: 距离(1,2)={dist_1_2_p1:.4f}, 距离(1,3)={dist_1_3_p1:.4f}")
print(f"当 p=2: 距离(1,2)={dist_1_2_p2:.4f}, 距离(1,3)={dist_1_3_p2:.4f}")
print(f"当 p=3: 距离(1,2)={dist_1_2_p3:.4f}, 距离(1,3)={dist_1_3_p3:.4f}")

# 检查相似性关系
rel_p1 = "1与2更近" if dist_1_2_p1 < dist_1_3_p1 else "1与3更近"
rel_p2 = "1与2更近" if dist_1_2_p2 < dist_1_3_p2 else "1与3更近"
rel_p3 = "1与2更近" if dist_1_2_p3 < dist_1_3_p3 else "1与3更近"

print(f"\np=1 时, {rel_p1}")
print(f"p=2 时, {rel_p2}")
print(f"p=3 时, {rel_p3}")

if rel_p1 == rel_p2 == rel_p3:
    print("\n结论: 在不同的 p 取值下, 相似性关系(谁更近)保持一致。")
else:
    print("\n结论: 在不同的 p 取值下, 相似性关系(谁更近)发生了变化。")