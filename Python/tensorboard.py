import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

def tensorboard_smoothing(x,smooth=0.99):
    x = x.copy()
    weight = smooth
    for i in range(1,len(x)):
        x[i] = (x[i-1] * weight + x[i]) / (weight + 1)
        weight = (weight + 1) * smooth
    return x

fig, ax1 = plt.subplots(1, 1)    # a figure with a 2x1 grid of Axes
len_mean = pd.read_csv("./run-ReformerNet-v25-GoPro-tag-losses_l_pix.csv")
ax1.plot(len_mean['Step'], tensorboard_smoothing(len_mean['Value'], smooth=0.6), color="#3399FF", label="PSNRLoss")
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:,.0f}'))
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:,.0f}'))
ax1.legend()
ax1.grid(True)
#ax1.set_xticks(np.arange(0, 24, step=2))
ax1.set_xlabel("timesteps")
ax1.set_ylabel("Loss Value")
# ax1.set_title("Average Episode Length")

fig, ax2 = plt.subplots(1, 1)    # a figure with a 2x1 grid of Axes
len_mean = pd.read_csv("./run-ReformerNet-v25-GoPro-tag-metrics_gopro-test_psnr.csv")
ax2.plot(len_mean['Step'], tensorboard_smoothing(len_mean['Value'], smooth=0.6), color="g", label="PSNR")
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:,.0f}'))
ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:,.0f}'))
ax2.legend()
ax2.grid(True)
#ax1.set_xticks(np.arange(0, 24, step=2))
ax2.set_xlabel("timesteps")
ax2.set_ylabel("PSNR Value")

fig, ax3 = plt.subplots(1, 1)    # a figure with a 2x1 grid of Axes
len_mean = pd.read_csv("./run-ReformerNet-v25-GoPro-tag-metrics_gopro-test_ssim.csv")
ax3.plot(len_mean['Step'], tensorboard_smoothing(len_mean['Value'], smooth=0.6), color="r", label="SSIM")
ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:,.2f}'))
ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:,.0f}'))
ax3.legend()
ax3.grid(True)
#ax1.set_xticks(np.arange(0, 24, step=2))
ax3.set_xlabel("timesteps")
ax3.set_ylabel("SSIM Value")
plt.show()
# fig.savefig(fname='./figures/ep_len_mean'+'.pdf', format='pdf')
