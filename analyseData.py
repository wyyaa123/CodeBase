import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

if __name__ == "__main__":
    blurredPose = pd.read_csv("./bluredPose.csv")
    # deblurPose = pd.read_csv("./deblurPose.csv")
    groundtruthPose = pd.read_csv("./data.csv")
    groundtruthPose['#timestamp'] *= 1e-9
    groundtruthPose[' p_RS_R_x [m]'] = -groundtruthPose[' p_RS_R_x [m]'] - 1.053197
    groundtruthPose[' p_RS_R_y [m]'] = -groundtruthPose[' p_RS_R_y [m]'] + 0.428061
    groundtruthPose[' p_RS_R_z [m]'] = groundtruthPose[' p_RS_R_z [m]'] - 1.332837

    print (groundtruthPose[' p_RS_R_x [m]'])
    # print (deblurPose['P_x'])

    # 创建一个包含3个子图的图形
    fig, axes = plt.subplots(1, 3)

    # 在不同的子图中绘制图形
    # deblurmse = ((deblurPose['P_x'] - groundtruthPose[' p_RS_R_x [m]']) ** 2).mean()
    blurredmse = ((blurredPose['P_x'] - groundtruthPose[' p_RS_R_x [m]']) ** 2).mean()
    axes[0].plot(blurredPose['timestamp'], blurredPose['P_x'], linestyle='-.', linewidth=2, color='red', label='blurredPose')
    # axes[0].plot(deblurPose['timestamp'], deblurPose['P_x'], linestyle='--', linewidth=2, color='blue', label='deblurredPose')
    axes[0].plot(groundtruthPose['#timestamp'], groundtruthPose[' p_RS_R_x [m]'], linestyle='-', linewidth=2, color='green', label='groundtruth')
    # axes[0].set_title(f'deblurred mse is {deblurmse}, blurred mse is {blurredmse}')
    axes[0].set_xlabel('timestamp')
    axes[0].set_ylabel('P_x')
    axes[0].legend()

    # deblurmse = ((deblurPose['P_y'] - groundtruthPose[' p_RS_R_y [m]']) ** 2).mean()
    blurredmse = ((blurredPose['P_y'] - groundtruthPose[' p_RS_R_y [m]']) ** 2).mean()
    axes[1].plot(blurredPose['timestamp'], blurredPose['P_y'], linestyle='-.', linewidth=2, color='red', label='blurredPose')
    # axes[1].plot(deblurPose['timestamp'], deblurPose['P_y'], linestyle='--', linewidth=2, color='blue', label='debluredPose')
    axes[1].plot(groundtruthPose['#timestamp'], groundtruthPose[' p_RS_R_y [m]'], linestyle='-', linewidth=2, color='green', label='groundtruth')
    # axes[1].set_title(f'deblurred mse is {deblurmse}, blurred mse is {blurredmse}')
    axes[1].set_xlabel('timestamp')
    axes[1].set_ylabel('P_y')
    axes[1].legend()

    # deblurmse = ((deblurPose['P_z'] - groundtruthPose[' p_RS_R_z [m]']) ** 2).mean()
    blurredmse = ((blurredPose['P_z'] - groundtruthPose[' p_RS_R_z [m]']) ** 2).mean()
    axes[2].plot(blurredPose['timestamp'], blurredPose['P_z'], linestyle='-.', linewidth=2, color='red', label='blurredPose')
    # axes[2].plot(deblurPose['timestamp'], deblurPose['P_z'], linestyle='--', linewidth=2, color='blue', label='debluredPose')
    axes[2].plot(groundtruthPose['#timestamp'], groundtruthPose[' p_RS_R_z [m]'], linestyle='-', linewidth=2, color='green', label='groundtruth')
    # axes[2].set_title(f'deblurred mse is {deblurmse}, blurred mse is {blurredmse}')
    axes[2].set_xlabel('timestamp')
    axes[2].set_ylabel('P_z')
    axes[2].legend()

    # 调整子图之间的距离
    plt.tight_layout()

    # 显示图形
    plt.show()



