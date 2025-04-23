import numpy as np
import torch
import matplotlib.pyplot as plt

# 加载数据
e, u, x, y = torch.load('data/{}.pt'.format('cora'))
# cov_matrix = torch.cov(u.T, correction=1)
matrix_abs = abs(u.T @ u)

# 创建画布并调整右侧空间
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
plt.subplots_adjust(right=0.85)  # 为colorbar留出空间

# 绘制第一个子图
im1 = ax1.imshow(matrix_abs, cmap='viridis')
ax1.set_title("(a) Linear Independence", pad=10)
ax1.set_xticks([])
ax1.set_yticks([])

# 绘制第二个子图
im2 = ax2.imshow(matrix_abs, cmap='viridis')
ax2.set_title("(b) Linear Independence", pad=10)
ax2.set_xticks([])
ax2.set_yticks([])

# 添加共享colorbar
cax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
fig.colorbar(im1, cax=cax)

# 保存和显示
# plt.savefig('stat/basis_indepen.pdf', dpi=500)
plt.savefig('stat/basis_indepen.pdf', dpi=500, bbox_inches='tight')  # 确保包含完整元素

plt.show()