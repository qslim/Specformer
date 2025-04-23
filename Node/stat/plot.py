import numpy as np
import torch
import matplotlib.pyplot as plt


# # 创建画布并调整右侧空间
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
# plt.subplots_adjust(right=0.85)  # 为colorbar留出空间
# 创建2x4子图画布
fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(12, 6))
plt.subplots_adjust(right=0.88, wspace=0.1, hspace=0.1)  # 调整间距


# 加载数据
e, u, x, y = torch.load('data/{}.pt'.format('cora'))
matrix_abs = abs(u.T @ u)
im1 = ax1.imshow(matrix_abs, cmap='viridis')
ax1.set_title("Cora", pad=10)
ax1.set_xticks([])
ax1.set_yticks([])

# 加载数据
e, u, x, y = torch.load('data/{}.pt'.format('citeseer'))
matrix_abs = abs(u.T @ u)
im2 = ax2.imshow(matrix_abs, cmap='viridis')
ax2.set_title("Citeseer", pad=10)
ax2.set_xticks([])
ax2.set_yticks([])

# 加载数据
e, u, x, y = torch.load('data/{}.pt'.format('squirrel'))
matrix_abs = abs(u.T @ u)
im3 = ax3.imshow(matrix_abs, cmap='viridis')
ax3.set_title("Squirrel", pad=10)
ax3.set_xticks([])
ax3.set_yticks([])

# 加载数据
e, u, x, y = torch.load('data/{}.pt'.format('chameleon'))
matrix_abs = abs(u.T @ u)
im4 = ax4.imshow(matrix_abs, cmap='viridis')
ax4.set_title("Chameleon", pad=10)
ax4.set_xticks([])
ax4.set_yticks([])


# 加载数据
e, u, x, y = torch.load('data/{}_desc.pt'.format('cora'))
matrix_abs = abs(u.T @ u)
im5 = ax5.imshow(matrix_abs, cmap='viridis')
# ax5.set_title("Cora", pad=10)
ax5.set_xticks([])
ax5.set_yticks([])

# 加载数据
e, u, x, y = torch.load('data/{}_desc.pt'.format('citeseer'))
matrix_abs = abs(u.T @ u)
im6 = ax6.imshow(matrix_abs, cmap='viridis')
# ax6.set_title("Citeseer", pad=10)
ax6.set_xticks([])
ax6.set_yticks([])

# 加载数据
e, u, x, y = torch.load('data/{}_desc.pt'.format('squirrel'))
matrix_abs = abs(u.T @ u)
im7 = ax7.imshow(matrix_abs, cmap='viridis')
# ax7.set_title("Squirrel", pad=10)
ax7.set_xticks([])
ax7.set_yticks([])

# 加载数据
e, u, x, y = torch.load('data/{}_desc.pt'.format('chameleon'))
matrix_abs = abs(u.T @ u)
im8 = ax8.imshow(matrix_abs, cmap='viridis')
# ax8.set_title("Chameleon", pad=10)
ax8.set_xticks([])
ax8.set_yticks([])

# 添加共享colorbar
# cax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
# fig.colorbar(im1, cax=cax)
# 添加共享colorbar
cax = fig.add_axes([0.9, 0.15, 0.02, 0.7])  # 调整colorbar位置
fig.colorbar(im8, cax=cax)

# 保存和显示
# plt.savefig('stat/basis_indepen.pdf', dpi=500)
plt.savefig('stat/basis_indepen.pdf', dpi=500, bbox_inches='tight')  # 确保包含完整元素

plt.show()