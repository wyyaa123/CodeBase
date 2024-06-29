import math
import pandas as pd
import numpy as np
from typing import List, overload
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.ticker import ScalarFormatter
from scipy.spatial.transform import Rotation as R

class Pose:
    def __init__(self, file_path, gt_relat=None, sync_tolerate=0.001):
        self.timestamp: List[float] = []
        self.position: List[float] = []
        self.orientation: List[float] = []
        self.total_length: int = 0

        self.delta_position: List[float] = []
        self.delta_orientation: List[float] = []

        self.error_position: List[float] = []
        self.error_timestamp: List[float] = []

        self.errorx_max: float = 0
        self.errory_max: float = 0
        self.errorz_max: float = 0

        self.related_velocity: List[float] = [0]

        self.iter: int = 0

        self.toler = sync_tolerate

        file = open(file_path, "r")

        pose = file.readlines()

        for i in range(0, len(pose)):
            pose_line = pose[i].strip().split(" ")
            self.timestamp.append(float(pose_line[0]))
            self.position.append([float(pose_line[1]), float(pose_line[2]), float(pose_line[3])])
            self.orientation.append([float(pose_line[4]), float(pose_line[5]), float(pose_line[6]), float(pose_line[7])])
            if len(self.position) > 1:
                # self.total_length += math.sqrt(float(self.position[-1][0] - self.position[-2][0]) ** 2 + float(self.position[-1][1] - self.position[-2][1]) ** 2 + float(self.position[-1][2] - self.position[-2][2]) ** 2)
                delta_length = math.sqrt(float(self.position[-1][0] - self.position[-2][0]) ** 2 + float(self.position[-1][1] - self.position[-2][1]) ** 2)
                self.total_length += delta_length
                self.related_velocity.append(delta_length / (self.timestamp[-1] - self.timestamp[-2]))
        
        self.timestamp = np.array(self.timestamp)
        self.position = np.array(self.position).transpose()
        self.orientation = np.array(self.orientation).transpose()

        file.close()

    @overload
    def ailgned(self, gt, tolerate, ): # the data type of gt is class Pose
        for i in range(self.iter, len(gt.timestamp)):
            if np.abs(self.timestamp[0] - gt.timestamp[i]) < tolerate:
                self.iter = i
                break
            elif gt.timestamp[i] - self.timestamp[0] > 0.1: 
                print("No matched timestamp!")
                exit(-1)

        self.delta_orientation = self.quat2roation(gt.orientation[:, self.iter]) @ np.linalg.inv(self.quat2roation(self.orientation[:, 0]))
        self.delta_position = gt.position[:, self.iter] - self.delta_orientation @ self.position[:, 0]
        self.position = self.delta_orientation @ self.position + self.delta_position.reshape(3, 1)

        self.iter = 0


    def ailgned(self, comp_pose: np.ndarray):
        self.position = comp_pose[:3, :3] @ self.position + comp_pose[:, 3].reshape(3, 1)
 
    # Time synchronization
    def sync(self, gt): # the data type of gt is class Pose

        for i in range(0, len(self.timestamp)):
            for j in range(self.iter, len(gt.timestamp)):
                if np.abs(self.timestamp[i] - gt.timestamp[j]) < self.toler:

                    self.errorx_max = max(self.errorx_max, np.abs(gt.position[0, j] - self.position[0, i]))
                    self.errory_max = max(self.errory_max, np.abs(gt.position[1, j] - self.position[1, i]))
                    self.errorz_max = max(self.errorz_max, np.abs(gt.position[2, j] - self.position[2, i]))

                    self.error_position.append(gt.position[:, j] - self.position[:, i])
                    self.error_timestamp.append(self.timestamp[i])
                    self.iter = j + 1
                    break


        self.iter = 0

        self.error_position = np.array(self.error_position).transpose()
        self.error_timestamp = np.array(self.error_timestamp)

    @staticmethod
    def vel2color(data, cmap):
        """数值映射为颜色"""
    
        dmin, dmax = np.nanmin(data), np.nanmax(data)
        cmo = plt.cm.get_cmap(cmap)
        cs, k = list(), 256 / cmo.N
        
        for i in range(cmo.N):
            c = cmo(i)
            for _ in range(int(i * k), int((i + 1) * k)):
                cs.append(c)
        cs = np.array(cs)
        data = np.uint8(255 * (data - dmin) / (dmax - dmin))
        
        return cs[data]

    @staticmethod
    def quat2roation(quat: np.ndarray):
        w, x, y, z = quat
        return np.array([[1-2*y*y-2*z*z,   2*x*y+2*w*z,   2*x*z-2*w*y],
                         [  2*x*y-2*w*z, 1-2*x*x-2*z*z,   2*y*z+2*w*x],
                         [  2*x*z+2*w*y,   2*y*z-2*w*x, 1-2*x*x-2*y*y]])

if __name__ == "__main__":

    plot_relative = True

    ## Reality
    # gt = Pose("gt/Reality/truth.tum")

    # vins = Pose("vins/Reality/vio4.csv")

    # nafnet = Pose("nafnet/Reality/vio.csv")

    # mprnet = Pose("mprnet/Reality/vio.csv")

    # uformer = Pose("uformer/Reality/vio.csv")

    # fftformer = Pose("fftformer/Reality/vio.csv")

    # mimounet = Pose("mimounet/Reality/vio.csv")

    # restormer = Pose("restormer/Reality/vio.csv")

    # mimounetplus = Pose("mimounetplus/Reality/vio.csv")

    # hinet = Pose("hinet/Reality/vio.csv")

    # proposed = Pose("proposed/Reality/proposed.csv")

    # MH_04_difficult
    gt = Pose("gt/MH_04_difficult/truth.tum")

    vins = Pose("vins/MH_04_difficult/vins.csv")

    nafnet = Pose("nafnet/MH_04_difficult/vio.csv")

    mprnet = Pose("mprnet/MH_04_difficult/vio.csv")

    uformer = Pose("uformer/MH_04_difficult/vio.csv")

    fftformer = Pose("fftformer/MH_04_difficult/vio.csv")

    mimounet = Pose("mimounet/MH_04_difficult/vio.csv")

    restormer = Pose("restormer/MH_04_difficult/vio.csv")

    mimounetplus = Pose("mimounetplus/MH_04_difficult/vio.csv")

    hinet = Pose("hinet/MH_04_difficult/vio.csv")

    proposed = Pose("proposed/MH_04_difficult/proposed.csv")

    # V2_03_difficult
    # gt = Pose("gt/V2_03_difficult/truth.tum")

    # vins = Pose("vins/V2_03_difficult/vins.csv")

    # nafnet = Pose("nafnet/V2_03_difficult/vio.csv")

    # mprnet = Pose("mprnet/V2_03_difficult/vio.csv")

    # uformer = Pose("uformer/V2_03_difficult/vio.csv")

    # fftformer = Pose("fftformer/V2_03_difficult/vio.csv")

    # mimounet = Pose("mimounet/V2_03_difficult/vio.csv")

    # restormer = Pose("restormer/V2_03_difficult/vio.csv")

    # mimounetplus = Pose("mimounetplus/V2_03_difficult/vio.csv")

    # hinet = Pose("hinet/V2_03_difficult/vio.csv")

    # proposed = Pose("proposed/V2_03_difficult/proposed.csv")

    ## caculated by evo
    ## calibrate the result of vins and proposed
    ## MH_04_difficult
    comp_vins = np.array([[6.65973124e-01, -7.45975699e-01, -2.34255495e-04, 4.82176653], 
                          [7.45967967e-01,  6.65967588e-01, -4.35471112e-03, -2.18052017], 
                          [3.40451524e-03,  2.72537347e-03,  9.99990491e-01, 0.56964317]])
    
    comp_nafnet = np.array([[ 0.68327535, -0.73015833, -0.00189978, 4.54431451],
                            [ 0.73014835,  0.68327658, -0.00406244, -2.00086429],
                            [ 0.0042643,   0.00138865,  0.99998994, 0.56724829]])
    
    comp_mprnet = np.array([[ 6.88246023e-01, -7.25463232e-01, -4.52885453e-03, 4.56324108],
                            [ 7.25439329e-01,  6.88260452e-01, -5.94391387e-03, -2.04138647],
                            [ 7.42912244e-03,  8.05465899e-04,  9.99972079e-01, 0.56123537]])
    
    comp_uformer = np.array([[6.70121196e-01, -7.42251493e-01, -5.51227849e-04, 4.53562738],
                             [7.42247785e-01,  6.70119891e-01, -2.74919278e-03, -1.97722762],
                             [2.40998119e-03,  1.43314470e-03,  9.99996069e-01, 0.57718897]])
    
    comp_fftformer = np.array([[ 0.69760073, -0.71647839, -0.00345664, 4.69150584],
                               [ 0.71646154,  0.69760854, -0.00501914, -2.14214821],
                               [ 0.00600748,  0.00102481,  0.99998143, 0.57252072]])
    
    comp_mimounet = np.array([[ 6.74112023e-01, -7.38601209e-01, -6.42139109e-03, 4.66401748],
                              [ 7.38566615e-01,  6.74142602e-01, -7.14889570e-03, -2.00326432],
                              [ 9.60911631e-03,  7.65314550e-05,  9.99953828e-01, 0.62192803]])
    
    comp_restormer = np.array([[ 8.27650298e-01, -5.61230900e-01, -3.85490500e-03, 4.74108574],
                               [ 5.61221348e-01,  8.27659065e-01, -3.32727503e-03, -2.14307858],
                               [ 5.05791663e-03,  5.90365197e-04,  9.99987034e-01, 0.8340337]])
    
    comp_mimounetplus = np.array([[ 0.66721587, -0.7448635,   0.00116118, 4.64080714],
                                  [ 0.74486272,  0.66721025, -0.00316278, -1.9398375],
                                  [ 0.00158109,  0.00297518,  0.99999432, 0.58786839]])
    
    comp_hinet = np.array([[ 0.68268207, -0.73042907, -0.00168207, 4.54431451],
                            [ 0.73042907,  0.68268207, -0.00168207, -2.00086429],
                            [ 0.00168207,  0.00168207,  0.99999691, 0.56724829]])
    
    comp_proposed = np.array([[6.75551706e-01, -7.37312613e-01,  5.01269145e-05, 4.53945646], 
                           [7.37311532e-01,  6.75550597e-01, -1.75939819e-03, -1.9888367], 
                           [1.26336321e-03,  1.22552360e-03,  9.99998451e-01, 0.56824543]])

    ## V2_03_difficult
    # comp_vins = np.array([[-0.99358398,  0.11309006, -0.00122711, -0.91113537], 
    #                       [-0.11306874, -0.99303844,  0.03301688,  0.02790666], 
    #                       [ 0.00251531,  0.03294379,  0.99945404,  1.21711546]])

    # comp_nafnet = np.array([[ -0.96700505,  0.25464979, -0.0073974, -0.97114374],
    #                         [ -0.2547551,  -0.96647091,  0.03215299, 0.34245263],
    #                         [ 0.00103838,  0.03297662,  0.99945558,  1.32560948]])
    
    # comp_mprnet = np.array([[ -0.98365193,  0.17999803,  0.00544021, -1.03390138],
    #                         [ -0.17992511, -0.9836097,   0.0117864, 0.4570809],
    #                         [ 0.00747257,  0.01061489,  0.99991574, 1.40823212]])
    
    # comp_uformer = np.array([[-0.94195472,  0.33568706,  0.00595871, -1.09289129],
    #                          [-0.33565567, -0.94196758,  0.00568774, 0.20565098],
    #                          [ 0.00752222,  0.00335752,  0.99996607, 1.75762732]])
    
    # comp_fftformer = np.array([[-0.97941745,  0.20172736,  0.00689461, -0.89523814],
    #                            [-0.20167284,  -0.9794214,  0.00786031, 0.41917632],
    #                            [ 0.00833837,  0.00630807,  0.99994534, 1.45750218]])
    
    # comp_mimounet = np.array([[ -0.93180326,  0.36295971, -0.001714, -0.92965839],
    #                           [ -0.36283123, -0.93132506,  0.03141874, 0.23699347],
    #                           [ 0.00980744,  0.02989798,  0.99950484, 1.27042246]])
    
    # comp_restormer = np.array([[-9.84227036e-01,  1.76907888e-01, -8.60162658e-04, -0.92459707],
    #                            [-1.76904562e-01, -9.84147218e-01,  1.26106401e-02, 0.32760372],
    #                            [ 1.38439502e-03,  1.25638996e-02,  9.99920113e-01, 1.38920495]])
    
    # comp_mimounetplus = np.array([[-0.99031425, -0.13879807, -0.00357409, -0.67827686],
    #                               [ 0.13870611, -0.99014964,  0.01908692, 0.17176478],
    #                               [-0.00618811,  0.0184063,   0.99981144, 1.53273816]])
    
    # comp_hinet = np.array([[ -0.95817789, -0.28616505,  0.0021652, -1.20444811],
    #                         [ 0.28616951, -0.95809931,  0.01235818, -0.11968799],
    #                         [-0.001462,    0.01246095,  0.99992129, 1.48469399]])
    
    # comp_proposed = np.array([[-0.98557818,  0.16917497,  0.00393337, -1.01333988], 
    #                         [-0.16908849, -0.98546076,  0.01661861,  0.43565408], 
    #                         [ 0.00668764,  0.01571385,  0.99985416,  1.51702763]])

    ## Reality
    # comp_vins = np.array([[-0.02409954,  0.99966784, -0.00913336, -1.01440124], 
    #                       [-0.99965294, -0.02399993,  0.01086294,  0.64366377], 
    #                       [ 0.01064013,  0.00939198,  0.99989928, -0.96868531]])

    # comp_nafnet = np.array([[ -1.74041550e-02,  9.99848204e-01, -8.15177738e-04, -1.16527564],
    #                         [ -9.99707987e-01, -1.73880329e-02,  1.67808400e-02, -2.67833438],
    #                         [  1.67641184e-02,  1.10699604e-03,  9.99858859e-01, -0.48302632]])
    
    # comp_mprnet = np.array([[ -0.01761535,  0.99983946, -0.00328049, -1.1383894],
    #                         [ -0.99974466, -0.01756705,  0.01421305, -2.02311163],
    #                         [  0.01415314,  0.00353002,  0.99989361, -0.47737607]])
    
    # comp_uformer = np.array([[-2.88429526e-02,  9.99583660e-01,  7.67956791e-04, -1.4095847],
    #                          [-9.99462787e-01, -2.88514098e-02,  1.55477656e-02, -1.06694345],
    #                          [ 1.55634491e-02, -3.19100769e-04,  9.99878831e-01, -0.68245193]])
    
    # comp_fftformer = np.array([[-3.05836963e-02,  9.99532152e-01, -3.38729140e-04,  1.76045362],
    #                            [-9.99383727e-01, -3.05733102e-02,  1.72464179e-02, -3.35239406],
    #                            [ 1.72279931e-02,  8.65979598e-04,  9.99851212e-01, -0.61196235]])
    
    # comp_mimounet = np.array([[ -2.51277930e-02,  9.99683961e-01, -7.56488004e-04, -0.31115359],
    #                           [ -9.99576049e-01, -2.51139329e-02,  1.47313391e-02, -0.84635974],
    #                           [  1.47076850e-02,  1.12633333e-03,  9.99891202e-01, -0.80494142]])
    
    # comp_restormer = np.array([[-4.18555309e-02,  9.99123302e-01,  8.61281885e-04, -0.76163117],
    #                            [-9.99026207e-01, -4.18634729e-02,  1.39314960e-02, -0.84056098],
    #                            [ 1.39553385e-02, -2.77333014e-04,  9.99902581e-01, -0.53924941]])
    
    # comp_mimounetplus = np.array([[-0.09185661,  0.99562671, -0.01702411, 1.07388723],
    #                               [-0.99571618, -0.0916566,   0.01218003, 2.35288783],
    #                               [ 0.01056639,  0.01807,     0.99978089, -1.38917772]])
    
    # comp_hinet = np.array([[-0.11867145,  0.99274507, -0.01934727, 1.80068841],
    #                        [-0.99290364, -0.11849403,  0.01007648, 10.66126535],
    #                        [ 0.00771084,  0.02040577,  0.99976205, -1.64731224]])

    # comp_proposed = np.array([[-0.0495103,   0.99874241, -0.00789513, -0.19402639], 
    #                           [-0.99864441, -0.0493752,   0.01647545, 1.21238598], 
    #                           [ 0.01606491,  0.00870014,  0.9998331, -1.12891343]])


    vins.ailgned(comp_vins)
    nafnet.ailgned(comp_nafnet)
    mprnet.ailgned(comp_mprnet)
    uformer.ailgned(comp_uformer)
    fftformer.ailgned(comp_fftformer)
    mimounet.ailgned(comp_mimounet)
    restormer.ailgned(comp_restormer)
    mimounetplus.ailgned(comp_mimounetplus)
    hinet.ailgned(comp_hinet)
    proposed.ailgned(comp_proposed)
    # hinet.ailgned(comp_hinet)

    if plot_relative:
        vins.timestamp -= gt.timestamp[0]
        nafnet.timestamp -= gt.timestamp[0]
        mprnet.timestamp -= gt.timestamp[0]
        uformer.timestamp -= gt.timestamp[0]
        fftformer.timestamp -= gt.timestamp[0]
        mimounet.timestamp -= gt.timestamp[0]
        restormer.timestamp -= gt.timestamp[0]
        mimounetplus.timestamp -= gt.timestamp[0]
        hinet.timestamp -= gt.timestamp[0]
        proposed.timestamp -= gt.timestamp[0]
        # hinet.timestamp -= gt.timestamp[0]
        gt.timestamp -= gt.timestamp[0]

    gt.sync(gt)
    vins.sync(gt)
    nafnet.sync(gt)
    mprnet.sync(gt)
    uformer.sync(gt)
    fftformer.sync(gt)
    mimounet.sync(gt)
    restormer.sync(gt)
    mimounetplus.sync(gt)
    hinet.sync(gt)
    proposed.sync(gt)

    # 创建一个包含3个子图的图形
    fig, axes_traj = plt.subplots(3, 1, figsize=(12, 12))

    axes_traj[0].plot(gt.timestamp, gt.position[0, :], linestyle='-', linewidth=2, color='green', label='groundtruth')
    axes_traj[0].plot(vins.timestamp, vins.position[0, :], linestyle='--', linewidth=2, color='blue', label='vins')
    axes_traj[0].plot(nafnet.timestamp, nafnet.position[0, :], linestyle='-.', linewidth=2, color='purple', label='nafnet')
    axes_traj[0].plot(mprnet.timestamp, mprnet.position[0, :], linestyle=':', linewidth=2, color='orange', label='mprnet')
    axes_traj[0].plot(uformer.timestamp, uformer.position[0, :], linestyle='-', linewidth=2, color='black', label='uformer')
    axes_traj[0].plot(fftformer.timestamp, fftformer.position[0, :], linestyle='--', linewidth=2, color='yellow', label='fftformer')
    axes_traj[0].plot(mimounet.timestamp, mimounet.position[0, :], linestyle='-.', linewidth=2, color='brown', label='mimounet')
    axes_traj[0].plot(restormer.timestamp, restormer.position[0, :], linestyle=':', linewidth=2, color='pink', label='restormer')
    axes_traj[0].plot(mimounetplus.timestamp, mimounetplus.position[0, :], linestyle='-', linewidth=2, color='gray', label='mimounetplus')
    axes_traj[0].plot(hinet.timestamp, hinet.position[0, :], linestyle='--', linewidth=2, color='#476156', label='hinet')
    axes_traj[0].plot(proposed.timestamp, proposed.position[0, :], linestyle='-.', linewidth=2, color='red', label='proposed')
    axes_traj[0].set_xlabel('timestamp [s]', fontsize=14)
    axes_traj[0].set_ylabel('x [m]', fontsize=14)
    axes_traj[0].legend(fontsize=12, ncol=2)
    axes_traj[0].grid(True)

    axes_traj[1].plot(gt.timestamp, gt.position[1, :], linestyle='-', linewidth=2, color='green', label='groundtruth')
    axes_traj[1].plot(vins.timestamp, vins.position[1, :], linestyle='--', linewidth=2, color='blue', label='vins')
    axes_traj[1].plot(nafnet.timestamp, nafnet.position[1, :], linestyle='-.', linewidth=2, color='purple', label='nafnet')
    axes_traj[1].plot(mprnet.timestamp, mprnet.position[1, :], linestyle=':', linewidth=2, color='orange', label='mprnet')
    axes_traj[1].plot(uformer.timestamp, uformer.position[1, :], linestyle='-', linewidth=2, color='black', label='uformer')
    axes_traj[1].plot(fftformer.timestamp, fftformer.position[1, :], linestyle='--', linewidth=2, color='yellow', label='fftformer')
    axes_traj[1].plot(mimounet.timestamp, mimounet.position[1, :], linestyle='-.', linewidth=2, color='brown', label='mimounet')
    axes_traj[1].plot(restormer.timestamp, restormer.position[1, :], linestyle=':', linewidth=2, color='pink', label='restormer')
    axes_traj[1].plot(mimounetplus.timestamp, mimounetplus.position[1, :], linestyle='-', linewidth=2, color='gray', label='mimounetplus')
    axes_traj[1].plot(hinet.timestamp, hinet.position[1, :], linestyle='--', linewidth=2, color='#476156', label='hinet')
    axes_traj[1].plot(proposed.timestamp, proposed.position[1, :], linestyle='-.', linewidth=2, color='red', label='proposed')
    axes_traj[1].set_xlabel('timestamp [s]', fontsize=14)
    axes_traj[1].set_ylabel('y [m]', fontsize=14)
    axes_traj[1].legend(fontsize=12, ncol=2)
    axes_traj[1].grid(True)

    axes_traj[2].plot(gt.timestamp, gt.position[2, :], linestyle='-', linewidth=2, color='green', label='groundtruth')
    axes_traj[2].plot(vins.timestamp, vins.position[2, :], linestyle='--', linewidth=2, color='blue', label='vins')
    axes_traj[2].plot(nafnet.timestamp, nafnet.position[2, :], linestyle='-.', linewidth=2, color='purple', label='nafnet')
    axes_traj[2].plot(mprnet.timestamp, mprnet.position[2, :], linestyle=':', linewidth=2, color='orange', label='mprnet')
    axes_traj[2].plot(uformer.timestamp, uformer.position[2, :], linestyle='-', linewidth=2, color='black', label='uformer')
    axes_traj[2].plot(fftformer.timestamp, fftformer.position[2, :], linestyle='--', linewidth=2, color='yellow', label='fftformer')
    axes_traj[2].plot(mimounet.timestamp, mimounet.position[2, :], linestyle='-.', linewidth=2, color='brown', label='mimounet')
    axes_traj[2].plot(restormer.timestamp, restormer.position[2, :], linestyle=':', linewidth=2, color='pink', label='restormer')
    axes_traj[2].plot(mimounetplus.timestamp, mimounetplus.position[2, :], linestyle='-', linewidth=2, color='gray', label='mimounetplus')
    axes_traj[2].plot(hinet.timestamp, hinet.position[2, :], linestyle='--', linewidth=2, color='#476156', label='hinet')
    axes_traj[2].plot(proposed.timestamp, proposed.position[2, :], linestyle='-.', linewidth=2, color='red', label='proposed')
    axes_traj[2].set_xlabel('timestamp [s]', fontsize=14)
    axes_traj[2].set_ylabel('z [m]', fontsize=14)
    axes_traj[2].legend(fontsize=12, ncol=2)
    axes_traj[2].grid(True)

    fig, axes_err = plt.subplots(3, 1, figsize=(12, 12))
    axes_err[0].plot(gt.timestamp, gt.error_position[0, :], linestyle='-', linewidth=2, color='green')
    axes_err[0].plot(vins.error_timestamp, vins.error_position[0, :], linestyle='--', linewidth=2, color='blue', label='vins')
    axes_err[0].plot(nafnet.error_timestamp, nafnet.error_position[0, :], linestyle='-.', linewidth=2, color='purple', label='nafnet')
    axes_err[0].plot(mprnet.error_timestamp, mprnet.error_position[0, :], linestyle=':', linewidth=2, color='orange', label='mprnet')
    axes_err[0].plot(uformer.error_timestamp, uformer.error_position[0, :], linestyle='-', linewidth=2, color='black', label='uformer')
    axes_err[0].plot(fftformer.error_timestamp, fftformer.error_position[0, :], linestyle='--', linewidth=2, color='yellow', label='fftformer')
    axes_err[0].plot(mimounet.error_timestamp, mimounet.error_position[0, :], linestyle='-.', linewidth=2, color='brown', label='mimounet')
    axes_err[0].plot(restormer.error_timestamp, restormer.error_position[0, :], linestyle=':', linewidth=2, color='pink', label='restormer')
    axes_err[0].plot(mimounetplus.error_timestamp, mimounetplus.error_position[0, :], linestyle='-', linewidth=2, color='gray', label='mimounetplus')
    axes_err[0].plot(hinet.error_timestamp, hinet.error_position[0, :], linestyle='--', linewidth=2, color='#476156', label='hinet')
    axes_err[0].plot(proposed.error_timestamp, proposed.error_position[0, :], linestyle='-.', linewidth=2, color='red', label='proposed')
    axes_err[0].set_xlabel('timestamp [s]', fontsize=14)
    axes_err[0].set_ylabel('error x [m]', fontsize=14)
    axes_err[0].legend(fontsize=12, ncol=2)
    axes_err[0].grid(True)

    axes_err[1].plot(gt.timestamp, gt.error_position[1, :], linestyle='-', linewidth=2, color='green')
    axes_err[1].plot(vins.error_timestamp, vins.error_position[1, :], linestyle='--', linewidth=2, color='blue', label='vins')
    axes_err[1].plot(nafnet.error_timestamp, nafnet.error_position[1, :], linestyle='-.', linewidth=2, color='purple', label='nafnet')
    axes_err[1].plot(mprnet.error_timestamp, mprnet.error_position[1, :], linestyle=':', linewidth=2, color='orange', label='mprnet')
    axes_err[1].plot(uformer.error_timestamp, uformer.error_position[1, :], linestyle='-', linewidth=2, color='black', label='uformer')
    axes_err[1].plot(fftformer.error_timestamp, fftformer.error_position[1, :], linestyle='--', linewidth=2, color='yellow', label='fftformer')
    axes_err[1].plot(mimounet.error_timestamp, mimounet.error_position[1, :], linestyle='-.', linewidth=2, color='brown', label='mimounet')
    axes_err[1].plot(restormer.error_timestamp, restormer.error_position[1, :], linestyle=':', linewidth=2, color='pink', label='restormer')
    axes_err[1].plot(mimounetplus.error_timestamp, mimounetplus.error_position[1, :], linestyle='-', linewidth=2, color='gray', label='mimounetplus')
    axes_err[1].plot(hinet.error_timestamp, hinet.error_position[1, :], linestyle='--', linewidth=2, color='#476156', label='hinet')
    axes_err[1].plot(proposed.error_timestamp, proposed.error_position[1, :], linestyle='-.', linewidth=2, color='red', label='proposed')
    axes_err[1].set_xlabel('timestamp [s]', fontsize=14)
    axes_err[1].set_ylabel('error y [m]', fontsize=14)
    axes_err[1].legend(fontsize=12, ncol=2)
    axes_err[1].grid(True)


    axes_err[2].plot(gt.timestamp, gt.error_position[2, :], linestyle='-', linewidth=2, color='green')
    axes_err[2].plot(vins.error_timestamp, vins.error_position[2, :], linestyle='--', linewidth=2, color='blue', label='vins')
    axes_err[2].plot(nafnet.error_timestamp, nafnet.error_position[2, :], linestyle='-.', linewidth=2, color='purple', label='nafnet')
    axes_err[2].plot(mprnet.error_timestamp, mprnet.error_position[2, :], linestyle=':', linewidth=2, color='orange', label='mprnet')
    axes_err[2].plot(uformer.error_timestamp, uformer.error_position[2, :], linestyle='-', linewidth=2, color='black', label='uformer')
    axes_err[2].plot(fftformer.error_timestamp, fftformer.error_position[2, :], linestyle='--', linewidth=2, color='yellow', label='fftformer')
    axes_err[2].plot(mimounet.error_timestamp, mimounet.error_position[2, :], linestyle='-.', linewidth=2, color='brown', label='mimounet')
    axes_err[2].plot(restormer.error_timestamp, restormer.error_position[2, :], linestyle=':', linewidth=2, color='pink', label='restormer')
    axes_err[2].plot(mimounetplus.error_timestamp, mimounetplus.error_position[2, :], linestyle='-', linewidth=2, color='gray', label='mimounetplus')
    axes_err[2].plot(hinet.error_timestamp, hinet.error_position[2, :], linestyle='--', linewidth=2, color='#476156', label='hinet')
    axes_err[2].plot(proposed.error_timestamp, proposed.error_position[2, :], linestyle='-.', linewidth=2, color='red', label='proposed')
    axes_err[2].set_xlabel('timestamp [s]', fontsize=14)
    axes_err[2].set_ylabel('error z [m]', fontsize=14)
    axes_err[2].legend(fontsize=12, ncol=2)
    axes_err[2].grid(True)

    # x, y = gt.position[0, :], gt.position[1, :]
    # ps = np.stack([x, y], axis=1)
    # segments = np.stack([ps[:-1], ps[1:]], axis=1)
    # colors = Pose.vel2color(gt.related_velocity, 'jet')

    # # norm = mcolors.Normalize(vmin=np.min(gt.related_velocity), vmax=np.max(gt.related_velocity))
    # line_segments = LineCollection(segments, colors=colors, linewidth=3)

    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot()
    # ax.set_xlim(np.min(x) - 0.1, np.max(x) + 0.1)
    # ax.set_ylim(np.min(y) - 0.1, np.max(y) + 0.1)
    # ax.add_collection(line_segments)
    # cb = fig.colorbar(line_segments, ax=ax, cmap='jet')
    # # ax.set_xlabel('timestamp [s]', fontsize=14)
    # # ax.set_ylabel('velocity [m/s]', fontsize=14)
    # # ax.legend()
    # ax.grid(True)

    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot()

    # ax.plot(gt.position[0, :], gt.position[1, :], linestyle='-', linewidth=2, color='#27B7B0', label='groundtruth', zorder=2)
    # ax.plot(vins.position[0, :], vins.position[1, :], linestyle='--', linewidth=2, color='blue', label='vins')
    # ax.plot(hinet.position[0, :], hinet.position[1, :], hinet.position[2, :], linestyle='-.', linewidth=2, color='purple', label='hinet')
    # ax.plot(proposed.position[0, :], proposed.position[1, :], linestyle='-.', linewidth=2, color='red', label='proposed')
    # for i in range(1, len(gt.position[0, :]), 200):
    #     plt.annotate(
    #         '', xy=(gt.position[0, i], gt.position[1, i]), xytext=(gt.position[0, i-1], gt.position[1, i-1]),
    #         arrowprops=dict(arrowstyle="->", color='r', lw=3)
    #     )
    # ax.scatter(gt.position[0, 0], gt.position[1, 0], s=200, color='#A1B727', label='start', marker='*', zorder=3)
    # ax.scatter(gt.position[0, -1], gt.position[1, -1], s=200, color='#277BB7', label='end', marker='*', zorder=3)
    # ax.scatter(hinet.position[0, 0], hinet.position[1, 0], s=200, color='purple', label='start', marker='*', zorder=3)    
    # ax.scatter(hinet.position[0, -1], hinet.position[1, -1], s=200, color='purple', label='end', marker='*', zorder=3)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.tick_params(axis='both', which='major', labelsize=16)
    # ax.set_xlabel('x [m]', fontsize=16)
    # ax.set_ylabel('y [m]', fontsize=16)
    # ax.set_title('2D Trajectory', fontsize=20)
    # ax.legend(prop={'size': 16}, loc='upper right')
    # ax.grid(True, linestyle=':', linewidth=1)

    # 调整子图之间的距离
    plt.tight_layout()
    # 显示图形
    plt.show()

    print("total length: ", gt.total_length)
    print("total length: ", proposed.total_length)
    print("error x max: ", proposed.errorx_max)
    print("error y max: ", proposed.errory_max)
    print("error z max: ", proposed.errorz_max)



