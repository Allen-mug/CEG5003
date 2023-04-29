from __future__ import print_function, division
import os
import argparse
import torch.nn as nn
from skimage import io
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from datasets import __datasets__
from models import __models__
from utils import *
from utils.KittiColormap import *

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='MobileStereoNet')
parser.add_argument('--model', default='MSNet2D', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--dataset', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--testlist', required=True, help='testing list')
parser.add_argument('--loadckpt', required=True, help='load the weights from a specific checkpoint')
parser.add_argument('--colored', default=1, help='save colored or save for benchmark submission')

# parse arguments
args = parser.parse_args()

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath, args.testlist, False)
# TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4, drop_last=False)
TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4, drop_last=True)

# model, optimizer
model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
model.cuda()

# load parameters
print("Loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])


def test(args):
    print("Generating the disparity maps...")

    os.makedirs('./predictions', exist_ok=True)

    for batch_idx, sample in enumerate(TestImgLoader):

        disp_est_tn = test_sample(sample)
        disp_est_np = tensor2numpy(disp_est_tn)
        top_pad_np = tensor2numpy(sample["top_pad"])
        right_pad_np = tensor2numpy(sample["right_pad"])
        left_filenames = sample["left_filename"]

        for disp_est, top_pad, right_pad, fn in zip(disp_est_np, top_pad_np, right_pad_np, left_filenames):

            assert len(disp_est.shape) == 2

            disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)

            disp_est = disparity_to_depth(disp_est)

            name = fn.split('/')
            # fn = os.path.join("predictions", '_'.join(name[2:]))
            # fn = os.path.join("predictions", '_'.join(name[-1]))
            fn = os.path.join('predictions', name[-1])
            # store as pfm
            fn = fn.replace('jpg', 'pfm')

            if float(args.colored) == 1:
                disp_est = kitti_colormap(disp_est)
                cv2.imwrite(fn, disp_est)
            else:
                # disp_est = np.round(disp_est * 256).astype(np.uint16)
                # disp_est = np.round(disp_est * 256).astype(np.uint8)
                # io.imsave(fn, disp_est)
                cv2.imwrite(fn, disp_est)

    print("Done!")


@make_nograd_func
def test_sample(sample):
    model.eval()
    disp_ests = model(sample['left'].cuda(), sample['right'].cuda())
    return disp_ests[-1]


DISPARITY_MULTIPLIER = 7.0
# DISPARITY_MULTIPLIER = 3.0
FOCAL_LENGTH_X_BASELINE = {
    'indoor_flying': 19.941772,
}

def depth_to_disparity(depth_maps):
    """
    Conversion from depth to disparity used in the paper "Learning an event sequence embedding for dense event-based
    deep stereo" (ICCV 2019)

    Original code available at https://github.com/tlkvstepan/event_stereo_ICCV2019
    """
    disparity_maps = DISPARITY_MULTIPLIER * FOCAL_LENGTH_X_BASELINE['indoor_flying'] / (depth_maps + 1e-15)
    return disparity_maps


def disparity_to_depth(disparity_map):
    depth_map = DISPARITY_MULTIPLIER * FOCAL_LENGTH_X_BASELINE['indoor_flying'] / (disparity_map + 1e-7)
    return depth_map


if __name__ == '__main__':
    test(args)
