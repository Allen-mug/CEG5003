import h5py
import numpy as np
import cv2
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from filters import filter,filter_by_pix_num,cal_pix_threshold
from utils import mvsecLoadRectificationMaps, mvsecRectifyEvents

HEIGHT = 260
WIDTH = 346


def load_mvsec(datapath, gtpath):
    data = h5py.File(datapath, 'r')
    gt = h5py.File(gtpath, 'r')
    # load the left & right events, rectified depth and timestamp in array
    Levents = np.array(data['davis']['left']['events'])
    Revents = np.array(data['davis']['right']['events'])
    depth_ts = np.array(gt['davis']['left']['depth_image_rect_ts'])
    depth_rect = np.array(gt['davis']['left']['depth_image_rect'])
    # rectify the input left and right events
    root = '/home/tlab4/zjg/data/mvsec'
    Lx_path = root+'/indoor_flying_left_x_map.txt'
    Ly_path = root+'/indoor_flying_left_y_map.txt'
    Rx_path = root+'/indoor_flying_right_x_map.txt'
    Ry_path = root+'/indoor_flying_right_y_map.txt'
    Lx_map, Ly_map, Rx_map, Ry_map = mvsecLoadRectificationMaps(Lx_path, Ly_path, Rx_path, Ry_path)
    rect_Levents = np.array(mvsecRectifyEvents(Levents, Lx_map, Ly_map)) 
    rect_Revents = np.array(mvsecRectifyEvents(Revents, Rx_map, Ry_map))

    return rect_Levents, rect_Revents, depth_rect, depth_ts


def get_event_stacks(depth_ts, rect_events):
    # get the event stacks by searching based on every depth timestamp
    # every stack stores an event window that accumulated before current depth_ts
    event_stacks = []  # initialize events_stacks
    ts_event = rect_events[0][2] # first ts of event
    # ts_sum = rect_events[0][2] # 
    sum_event = 0  # record the total number of the events
    print('Generating event stacks......')
    for ts in tqdm(depth_ts):
        event = rect_events[sum_event]
        ts_event = event[2]

        event_window = [] # record the accumulated events before current depth_ts
        count = 0  # record the number of events in current loop
        while ts_event < ts:  # loop till the depth_ts
            # idx = sum_event + count
            event_window.append(event)
            count += 1
            event = rect_events[sum_event+count]
            ts_event = event[2]
        sum_event += count

        event_stacks.append(np.array(event_window))

    event_stacks = np.array(event_stacks) # shape: depth_ts.shape, # of events in every stack, 4
    print('Done with the event stacks!')
    return event_stacks
    # print(event_stacks.shape, event_stacks[0].shape)


def denoise(event_window_iterator):
    # predefined two parameters
    length = 10  
    stride = 5

    real_event_stream = [] 
    noise_event_num = [] 
    noise_event_pix_num = []
    noise_event_pix_pol = []

    index_of_event_window = -1   
    for event_window in event_window_iterator:
        index_of_event_window += 1
        # print('==================The index of event_window:',index_of_event_window + 1,'======================')
        real_event_stream.append(event_window) 
        # print('event_window.shape:',event_window.shape)

        flag_arr = np.zeros((HEIGHT,WIDTH),dtype = int)
        pix_arr = {} 

        i = 0
        for event in event_window:
            x = int(event[1])
            y = int(event[0])
            flag_arr[x][y] = 1

            if (x,y) not in pix_arr:
                pix_arr[(x,y)] = []
            pix_arr[(x,y)].append(i)
            i += 1
                    
        index_X = range(0, WIDTH + 1 - length, stride)
        index_Y = range(0, HEIGHT + 1 - length, stride) 

        Threshold_flag_min = 3
        Threshold_flag_max = 0
        num_list = []   
        flag_sum = [] 

        for index_y in index_Y:
            for index_x in index_X:
                num = 0    
                for i in range(index_y,index_y + length): 
                    for j in range(index_x,index_x + length):
                        if flag_arr[i][j] == 1:
                            num += 1
                num_list.append(num)
                flag_sum.append([(index_x,index_y),num])

        num_list.sort(reverse = True)
        while 0 in num_list:
            num_list.remove(0)
        index_min = int(len(num_list) * 0.9) - 1
        if Threshold_flag_min < num_list[index_min]:  
            Threshold_flag_min = num_list[index_min]    
        Threshold_flag_max = math.ceil(length * length * 0.9) 
        # print('The candidate threshold of sparse/dense noise :',Threshold_flag_min,'-',Threshold_flag_max)
        Pix_Threshold = cal_pix_threshold(pix_arr)
        
        # print('start filetering.')
        filter(Threshold_flag_min,Threshold_flag_max,flag_sum, pix_arr, Pix_Threshold, index_of_event_window, length, real_event_stream, noise_event_pix_num, noise_event_pix_pol, noise_event_num)
        # print('filtering done.')

    print('The number of filtered events:',sum(noise_event_num))
    return np.array(real_event_stream)


def FrameGenerator(rect_events, depth_ts, event_stacks, step=1, GAMMA=0.8, denoise_flag=False):
    # define a threshold to accumulate some number of events to one frame (channel)
    # input events # / depth_ts #
    th = int(len(rect_events) / len(depth_ts)) 
    # frames = np.ones((len(depth_ts), H, W, 3)) * 128
    frames = np.zeros((len(depth_ts), HEIGHT, WIDTH, 3))
    frames_mapped = np.zeros((len(depth_ts), HEIGHT, WIDTH, 3))

    if denoise_flag:
        print('Start the denoising process......')
        event_window_iterator = np.copy(event_stacks)
        event_stacks = denoise(event_window_iterator)
        print('Denoising done!')

    print('Accumulating events to frames......')
    for i, stack in enumerate(event_stacks):
        # print('Running on {} stack'.format(i))

        num_channel = int(len(stack) / th) + 1 # define the number of channels
        avg_event_num = int(len(stack) / num_channel) + 1  # average the events for every channel
        # record the intensity values in different channels with polarity
        intensity = np.zeros((num_channel, 2, HEIGHT, WIDTH))  
        
        ts = np.zeros((num_channel, 1))  # record the last ts of current channel 
        ts_start = stack[0][2] 
        ts_end = stack[-1][2] 
        ts_range = ts_end - ts_start
        
        channel_sum = np.zeros((2, HEIGHT, WIDTH)) # initialize the sum of different channels
        
        idx = 0 # the number of events in current stack
        for j in range(num_channel):
            freq_count = np.zeros((2, HEIGHT, WIDTH)) # record the events frequency in every pixel
            while idx < ((j+1)*avg_event_num):  # a frame built by avg number of events
                x = int(stack[idx][0])
                y = int(stack[idx][1])
                pol = int(stack[idx][3])
                if pol == 1:   # stack the events intensity based on the frequency
                    freq = freq_count[0][y][x]
                    # intensity[j][0][y][x] += (step + freq/10)
                    intensity[j][0][y][x] += step
                    freq_count[0][y][x] += 1
                else:
                    freq = freq_count[1][y][x]
                    # intensity[j][1][y][x] += (step + freq/10)
                    intensity[j][1][y][x] += step
                    freq_count[1][y][x] += 1

                idx += 1
                if stack[idx-1][2] == ts_end:  # check the terminal
                    break

            ts[j] = stack[idx-1][2]  
            ratio_ts = (ts[j]-ts_start) / ts_range
            channel_sum += (intensity[j] * (GAMMA + (1-GAMMA)*ratio_ts))

        frames[i,:,:,0] = channel_sum[0]
        frames[i,:,:,2] = channel_sum[1]
    
    for i in range(len(frames)):
        frame = frames[i]
        frame_mapped = np.interp(frame, (frame.min(), GAMMA*frame.max()), (0, 255), right=255)
        frames_mapped[i] = frame_mapped
        
    print('Done with generating the frames!')
    return frames_mapped


def write_to_file(frames, save_root, sequence_num=0):
    for i in range(len(frames)):
        frame = frames[i]
        value_range = frame.max() - frame.min()
        frame_map = (frame / value_range) * 255 
        # frame_map = np.interp(frame, (frame.min(), frame.max()), (0, 255))
        cv2.imwrite(save_root+'/{}.png'.format(i+sequence_num), frame_map)
    print('Finishing Writing!')
