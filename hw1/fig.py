import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 1. 设置 ---

# 创建网格
w1 = np.linspace(-3, 3, 400)
w2 = np.linspace(-3, 3, 400)
W1, W2 = np.meshgrid(w1, w2)

# 定义 OLS (无正则化) 的最优解 (故意设置在非原点)
# 我们假设 OLS 解在 (2, 0.5)
w_ols = np.array([2, 0.5])

# 定义损失函数 (均方误差), 简化为二次型
# (W1 - w_ols[0])**2 + (W2 - w_ols[1])**2
# 我们使用一个简单的圆形损失等高线
Loss = (W1 - w_ols[0])**2 + (W2 - w_ols[1])**2
# 定义等高线级别
levels = [0.5, 1.25, 3, 5, 8, 12]

# 设置约束大小
C = 1.0

# --- 2. 创建绘图 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# --- 图 1: L1 正则化 (Lasso) ---

# 绘制损失函数 L(w) 的等高线
ax1.contour(W1, W2, Loss, levels=levels, colors='gray', linestyles='--')

# 绘制 OLS 解
ax1.plot(w_ols[0], w_ols[1], 'kx', markersize=10, mew=2, label='OLS Solution (Overfitted)')

# 绘制 L1 约束 (菱形)
l1_diamond = patches.Polygon([
    [C, 0], [0, C], [-C, 0], [0, -C]
], color='red', fill=False, lw=2, label=f'L1 Constraint: $|w_1| + |w_2| \leq {C}$')
ax1.add_patch(l1_diamond)

# 绘制 L1 最优解 (等高线与菱形的切点)
# OLS(2, 0.5) 到 (1,0) 的距离: (2-1)^2 + (0.5-0)^2 = 1.25
# OLS(2, 0.5) 到 (0,1) 的距离: (2-0)^2 + (0.5-1)^2 = 4.25
# 显然, 等高线会先碰到 (1, 0)
w_l1 = np.array([1, 0])
ax1.plot(w_l1[0], w_l1[1], 'ro', markersize=10, label='L1 Solution (Sparse)')

# 设置图 1 属性
ax1.set_title('L1 Regularization (Lasso)', fontsize=16)
ax1.set_xlabel('$w_1$', fontsize=14)
ax1.set_ylabel('$w_2$', fontsize=14)
ax1.set_aspect('equal')
ax1.axhline(0, color='black', lw=0.5, linestyle=':')
ax1.axvline(0, color='black', lw=0.5, linestyle=':')
ax1.legend()
ax1.set_xlim(-2, 3)
ax1.set_ylim(-1.5, 2.5)


# --- 图 2: L2 正则化 (Ridge) ---

# 绘制损失函数 L(w) 的等高线
ax2.contour(W1, W2, Loss, levels=levels, colors='gray', linestyles='--')

# 绘制 OLS 解
ax2.plot(w_ols[0], w_ols[1], 'kx', markersize=10, mew=2, label='OLS Solution (Overfitted)')

# 绘制 L2 约束 (圆形)
l2_circle = patches.Circle((0, 0), radius=C, color='blue', fill=False, lw=2, label=f'L2 Constraint: $w_1^2 + w_2^2 \leq {C}$')
ax2.add_patch(l2_circle)

# 绘制 L2 最优解 (等高线与圆形的切点)
# 解在 OLS 点和原点的连线上
w_l2 = w_ols / np.linalg.norm(w_ols) * C
ax2.plot(w_l2[0], w_l2[1], 'bo', markersize=10, label='L2 Solution (Shrunk)')

# 绘制从原点到 OLS 和 L2 解的辅助线
ax2.plot([0, w_ols[0]], [0, w_ols[1]], 'k:', lw=1)

# 设置图 2 属性
ax2.set_title('L2 Regularization (Ridge)', fontsize=16)
ax2.set_xlabel('$w_1$', fontsize=14)
ax2.set_ylabel('$w_2$', fontsize=14)
ax2.set_aspect('equal')
ax2.axhline(0, color='black', lw=0.5, linestyle=':')
ax2.axvline(0, color='black', lw=0.5, linestyle=':')
ax2.legend()
ax2.set_xlim(-2, 3)
ax2.set_ylim(-1.5, 2.5)

# --- 显示图像 ---
fig.suptitle('L1 (Lasso) vs L2 (Ridge) Regularization', fontsize=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()