import sys

from PyQt5.QtWidgets import *
import argparse

from lib.dataset.h36m import H36MDataset3D
from lib.utils.ui.window import Window


def parse_args():
    parser = argparse.ArgumentParser(description='valid score model')

    parser.add_argument('--num-human', type=int, default=1, dest='vis_num_human', help='max number of visualized humans')
    parser.add_argument('--num-hypo', type=int, default=1, dest='vis_num_hypo', help='number of visualized hypotheses')
    parser.add_argument('--num-perline', type=int, default=5, dest='vis_num_perline', help='number of visualized humans per line')
    parser.add_argument('--vis-gt', action='store_true', default=False)

    args = parser.parse_args()

    return args


def preprocess_data(data_dict):
    """
    Preprocess 3d data
    """

    # change the key of pred data, legacy.
    if 'sample3d' in data_dict and 'pred3d' not in data_dict:
        data_dict['pred3d'] = data_dict.pop('sample3d')

    # align the coordinate system.
    for k, trajs in data_dict.items():
        if trajs.shape[-1] == 2:
            # fill the depth dim with 0
            trajs_3d = np.zeros((trajs.shape[0], trajs.shape[1], trajs.shape[2], 3), dtype=np.float32)  # [t, b, j, 3]
            trajs_3d[..., 0] = trajs[..., 0]
            trajs_3d[..., 2] = trajs[..., 1]
            trajs_3d[..., 2] *= -1
        else:
            # adjust axes for plot
            trajs_3d = trajs[..., [0, 2, 1]]
            trajs_3d[..., -1] *= -1
        data_dict[k] = trajs_3d
        print(f'{k}: {trajs_3d.shape}')

    return data_dict


if __name__ == '__main__':

    args = parse_args()

    meta_info = {
        'skeleton': H36MDataset3D.get_skeleton(),
        'preprocess_func': preprocess_data,
        # RGB
        'camera_colors': [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 0, 255], [0, 255, 255], [255, 255, 0],],
        'human_colors': [[250, 249, 25], [50, 166, 250]],  # yellow and blue used in demo
        'symbols': ['t', 'o', '+', 's', 'p', 'h', 'star', 'd', 't1', 't2', 't3'],
    }

    app = QApplication(sys.argv)
    win = Window(meta_info, args)
    win.show()
    sys.exit(app.exec())  # block
