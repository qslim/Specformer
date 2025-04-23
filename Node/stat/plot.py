import numpy as np
import torch
import matplotlib.pyplot as plt

e, u, x, y = torch.load('data/{}.pt'.format('cora'))
# cov_matrix = torch.cov(u.T, correction=1)
matrix = abs(u.T @ u)


# # 生成示例方阵（5x5）
# matrix = np.array([
#     [3, 7, 2, 5, 1],
#     [8, 4, 6, 9, 0],
#     [2, 5, 8, 3, 7],
#     [1, 6, 4, 2, 9],
#     [5, 0, 3, 7, 4]
# ])

# 绘制热力图
plt.figure(figsize=(8, 6))
heatmap = plt.imshow(matrix, cmap='viridis')  # cmap指定颜色映射
plt.colorbar(heatmap)  # 添加颜色条
plt.title("Linear Independence")
plt.xticks([])  # 可选：隐藏坐标轴刻度
plt.yticks([])
# plt.savefig("heatmap.png", dpi=300)
plt.savefig('stat/basis_indepen.pdf', dpi=500)
plt.show()