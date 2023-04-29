import h5py
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import mvsecLoadRectificationMaps, mvsecRectifyEvents
from tools import load_mvsec, get_event_stacks, denoise, FrameGenerator, write_to_file

H = 260
W = 346
GAMMA = 0.8 # largest decay ratio

def main():
    framesL = []
    framesR = []
    depth = []

    for i in range(1, 5):
        # load data and ground truth
        datapath = '/home/tlab4/zjg/data/mvsec/indoor_flying' + str(i) + '_data.hdf5'
        gtpath = '/home/tlab4/zjg/data/mvsec/indoor_flying' + str(i) + '_gt.hdf5'
        rect_Levents, rect_Revents, depth_rect, depth_ts = load_mvsec(datapath, gtpath)
        # get the left and right event stacks
        Levent_stacks = get_event_stacks(depth_ts, rect_Levents)
        Revent_stacks = get_event_stacks(depth_ts, rect_Revents)
        # get the frames
        Lframes = FrameGenerator(rect_Levents, depth_ts, Levent_stacks)
        Rframes = FrameGenerator(rect_Revents, depth_ts, Revent_stacks)

        framesL.append(Lframes)
        framesR.append(Rframes)
        depth.append(depth_rect)

    for i in range(4):
        if i == 0:
            start = 90
            end = 1301
        elif i == 1:
            start = 180
            end = 1571
        elif i == 2:
            start = 130
            end = 1801
        elif i == 3:
            start = 120
            end = 351
        framesL[i] = framesL[i][start:end]
        framesR[i] = framesR[i][start:end]
        depth[i] = depth[i][start:end]

    # LeftFrames = list(framesL[0]) + list(framesL[1]) + list(framesL[2]) + list(framesL[3])
    # RightFrames = list(framesR[0]) + list(framesR[1]) + list(framesR[2]) + list(framesR[3])
    # LeftDepth = list(depth[0]) + list(depth[1]) + list(depth[2]) + list(depth[3])

    LeftFrames = np.concatenate((framesL[0], framesL[1], framesL[2], framesL[3]), axis=0)
    RightFrames = np.concatenate((framesR[0] , framesR[1] , framesR[2] , framesR[3]), axis=0)
    LeftDepth = np.concatenate((depth[0] , depth[1] , depth[2] , depth[3]), axis=0)

    trainL = LeftFrames[0:3601]
    trainR = RightFrames[0:3601]
    trainD = LeftDepth[0:3601]
    testL = LeftFrames[3601:]
    testR = RightFrames[3601:]
    testD = LeftDepth[3601:]

    # write the data into png files
    print('Start writing to files......')
    root1 = '/home/tlab4/cj/data/data_train/TRAIN'
    for i in range(len(trainL)):
        cv2.imwrite(root1+'/left/{}.jpg'.format(str(i)), trainL[i])
        cv2.imwrite(root1+'/right/{}.jpg'.format(str(i)), trainR[i])
        cv2.imwrite(root1+'/depth/{}.pfm'.format(str(i)), trainD[i])
    
    root2 = '/home/tlab4/cj/data/data_train/VAL'
    for i in range(len(testL)):
        cv2.imwrite(root2+'/left/{}.jpg'.format(str(i)), testL[i])
        cv2.imwrite(root2+'/right/{}.jpg'.format(str(i)), testR[i])
        cv2.imwrite(root2+'/depth/{}.pfm'.format(str(i)), testD[i])



        # if i == 1:
        #     root = '/home/tlab4/cj/data/data_train/VAL'
        #     write_to_file(Lframes, save_root=root+'/left')
        #     write_to_file(Rframes, save_root=root+'/right')
        # elif i == 2:
        #     root = '/home/tlab4/cj/data/data_train/TRAIN'
        #     write_to_file(Lframes, save_root=root+'/left')
        #     write_to_file(Rframes, save_root=root+'/right')
        # elif i == 3:
        #     root = '/home/tlab4/cj/data/data_train/TRAIN'
        #     write_to_file(Lframes, save_root=root+'/left', sequence_num=1691)
        #     write_to_file(Rframes, save_root=root+'/right', sequence_num=1691)

    print('Files have been written!')


if __name__ == '__main__':
    main()

