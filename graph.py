import matplotlib.pyplot as plt
import numpy as np

gpus = ['RTX 4060 Ti 16GB', 'RTX 4070 Ti', 'RTX 3090', 'RTX 4080',
        'Colab 무료 (K80)', 'Colab Pro (T4)', 'Colab Pro+ (A100)']
vram = [16, 12, 24, 16, 12, 16, 40]  # GB
cuda_cores = [4352, 7680, 10496, 9728, 4992, 2560, 6912]
fp32_tflops = [20, 40, 35.6, 49, 8.7, 8.1, 19.5]

x = np.arange(len(gpus))
width = 0.3

fig, axs = plt.subplots(3, 1, figsize=(12, 14))

# VRAM
axs[0].bar(x, vram, width, color='skyblue')
axs[0].set_title('VRAM (GB)')
axs[0].set_xticks(x)
axs[0].set_xticklabels(gpus, rotation=45, ha='right')

# CUDA Cores
axs[1].bar(x, cuda_cores, width, color='orange')
axs[1].set_title('CUDA Cores')
axs[1].set_xticks(x)
axs[1].set_xticklabels(gpus, rotation=45, ha='right')

# FP32 TFLOPS
axs[2].bar(x, fp32_tflops, width, color='green')
axs[2].set_title('FP32 TFLOPS')
axs[2].set_xticks(x)
axs[2].set_xticklabels(gpus, rotation=45, ha='right')

plt.tight_layout()
plt.show()
