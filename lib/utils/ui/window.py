import warnings

import numpy as np
import pims
import pyqtgraph as pg
import pyqtgraph.opengl as gl
# import zarr
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from pyqtgraph import GraphicsLayoutWidget, Vector

from .custom import CustomPlotItem, Plot3dToolBar
from .ui import UiForm


warnings.simplefilter('ignore', RuntimeWarning)


class Window(QMainWindow):
    def __init__(self, meta_info, args, parent=None):
        super().__init__(parent)
        self.meta_info = meta_info
        self.args = args
        self.setupUi()
        self.connectSlots()

    def setupUi(self):
        pg.setConfigOptions(useCupy=False)  # True if you install cupy
        pg.setConfigOptions(imageAxisOrder='row-major')
        pg.setConfigOptions(antialias=True)
        self.setWindowTitle('PoseViewer')

        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.resize(1920, 1000)

        # ui code from Qt Designer
        self.m_ui = UiForm()
        self.m_ui.setupUi(self.centralWidget)

        # Specialize
        self.item_dict = {'2d': {}, '3d': {}, 'plot':{}}  # used for update func

        # self.m_ui.openGLWidget.setBackgroundColor(231,228,231)

        # plot 3d axes
        origin = [-1200, -1200, 100]
        # axes = [gl.GLLinePlotItem(), gl.GLLinePlotItem(), gl.GLLinePlotItem()]
        x_line_1 = np.linspace(origin[0], origin[0] + 100, 10)
        x_line_2 = np.linspace(origin[1], origin[1], 10)
        x_line_3 = np.linspace(origin[2], origin[2], 10)
        x_line = np.vstack([x_line_1, x_line_2, x_line_3]).transpose()
        axis_x = gl.GLLinePlotItem(pos=x_line, color=pg.glColor('r'), width=5, antialias=True)
        self.m_ui.openGLWidget.addItem(axis_x)
        self.item_dict['3d']['axis_x'] = axis_x

        y_line_1 = np.linspace(origin[0], origin[0], 10)
        y_line_2 = np.linspace(origin[1], origin[1] + 100, 10)
        y_line_3 = np.linspace(origin[2], origin[2], 10)
        y_line = np.vstack([y_line_1, y_line_2, y_line_3]).transpose()
        axis_y = gl.GLLinePlotItem(pos=y_line, color=pg.glColor('g'), width=5, antialias=True)
        self.m_ui.openGLWidget.addItem(axis_y)
        self.item_dict['3d']['axis_y'] = axis_y

        z_line_1 = np.linspace(origin[0], origin[0], 10)
        z_line_2 = np.linspace(origin[1], origin[1], 10)
        z_line_3 = np.linspace(origin[2], origin[2] + 100, 10)
        z_line = np.vstack([z_line_1, z_line_2, z_line_3]).transpose()
        axis_z = gl.GLLinePlotItem(pos=z_line, color=pg.glColor('b'), width=5, antialias=True)
        self.m_ui.openGLWidget.addItem(axis_z)
        self.item_dict['3d']['axis_z'] = axis_z

        # init 3d viewpoint
        self.m_ui.openGLWidget.opts['distance'] = 450
        self.m_ui.openGLWidget.opts['elevation'] = 8
        self.m_ui.openGLWidget.opts['azimuth'] = 792

        # floor
        gz = gl.GLGridItem()
        # gz.setColor('#808080')
        gz.scale(150, 140, 40)
        # gz.translate(0, 0, -100)
        gz.translate(0, 0, 0)
        self.m_ui.openGLWidget.addItem(gz)
        self.item_dict['3d']['floor'] = gz

        # 2d area
        self.item_dict['2d']['img_list'] = []
        for cam_area_name in ['graphicsView_cam%d' % v for v in range(1, 10)]:
            layoutWidget = getattr(self.m_ui, cam_area_name, None)
            if layoutWidget is not None:
                view = layoutWidget.addViewBox()
                view.setAspectLocked(True)
                img = pg.ImageItem()
                view.addItem(img)
                self.item_dict['2d']['img_list'].append(img)
            else:
                break

        # Plot3dToolBar
        self.openGLToolBar = Plot3dToolBar()
        self.openGLToolBar.setupUi(self.m_ui.openGLWidget)

        # some status var
        self.video_list = []
        self.current_frame = 0
        self.m_ui.num_views = 4
        self.visibility_dict = {}
        self.graphicsView_plot_idx_list = [1, 2]
        self.current_batch = 0

    def connectSlots(self):
        self.m_ui.pushButton_open.clicked.connect(self.openfile)
        self.m_ui.pushButton_reset.clicked.connect(self.clear)
        self.m_ui.spinBox_numFrame.valueChanged.connect(self.update)
        self.m_ui.pushButton_back.clicked.connect(self.backward_batch)
        self.m_ui.pushButton_forward.clicked.connect(self.forward_batch)

    def clear(self):
        # clear 2d camera view
        for video, cam_item in zip(self.video_list, self.item_dict['2d']['img_list']):
            cam_item.setImage(np.zeros((20, 20, 3), dtype=np.uint8))

        # clear plot
        for idx in self.graphicsView_plot_idx_list:
            getattr(self.m_ui, f'graphicsView_plot{idx}').clear()
            self.item_dict['plot'].clear()

        # clear 3d view
        if 'gt_human_line_list' in self.item_dict['3d']:
            for human_idx in range(self.gt_num_humans):
                for gt_line in self.item_dict['3d']['gt_human_line_list'][human_idx]:
                    # gt_line.setData(pos=np.zeros((2, 3)))
                    self.m_ui.openGLWidget.removeItem(gt_line)
            self.item_dict['3d'].pop('gt_human_line_list')

        if 'pred_human_line_list' in self.item_dict['3d']:
            for human_idx in range(self.pred_num_humans):
                for hypo_idx in range(self.num_hypotheses):
                    for pred_line in self.item_dict['3d']['pred_human_line_list'][human_idx][hypo_idx]:
                        # pred_line.setData(pos=np.zeros((2, 3)))
                        self.m_ui.openGLWidget.removeItem(pred_line)
            self.item_dict['3d'].pop('pred_human_line_list')

        if 'camera_axes_list' in self.item_dict['3d']:
            for camera_axes in self.item_dict['3d']['camera_axes_list']:
                for camera_ax in camera_axes:
                    # camera_ax.setData(pos=np.zeros((2, 3)))
                    self.m_ui.openGLWidget.removeItem(camera_ax)
            self.item_dict['3d'].pop('camera_axes_list')


        self.video_list = []
        # self.current_frame = 0
        self.current_batch = 0
        if hasattr(self, 'video_len'):
            delattr(self, 'video_len')
        if hasattr(self, 'zarr_len'):
            delattr(self, 'zarr_len')
        if hasattr(self, 'data3d_group'):
            delattr(self, 'data3d_group')

    def add_view(self):
        self.m_ui.num_views += 1

        side = 'l' if self.m_ui.num_views % 2 == 1 else 'r'
        side_num = (self.m_ui.num_views + 1) // 2

        # groupbox
        groupBox_name = f'groupBox_{side:s}{side_num:d}'
        setattr(
            self.m_ui,
            groupBox_name,
            QGroupBox(getattr(self.m_ui, f'scrollAreaWidgetContents_{side:s}')),
        )
        getattr(self.m_ui, groupBox_name).setObjectName(groupBox_name)
        getattr(self.m_ui, groupBox_name).setFont(self.m_ui.font)

        # self.groupBox_r2 = QGroupBox(self.scrollAreaWidgetContents_r)
        # self.groupBox_r2.setObjectName(u"groupBox_r2")
        # self.groupBox_r2.setFont(font)

        # groupbox_layout
        groupbox_layout_name = f'verticalLayout_{side:s}{side_num:d}'
        setattr(
            self.m_ui,
            groupbox_layout_name,
            QVBoxLayout(getattr(self.m_ui, groupBox_name)),
        )
        getattr(self.m_ui, groupbox_layout_name).setObjectName(groupbox_layout_name)
        # self.verticalLayout_r2 = QVBoxLayout(self.groupBox_r2)
        # self.verticalLayout_r2.setObjectName(u"verticalLayout_r2")

        # graphicsView
        graphicsView_name = f'graphicsView_cam{self.m_ui.num_views:d}'
        setattr(
            self.m_ui,
            graphicsView_name,
            GraphicsLayoutWidget(getattr(self.m_ui, groupBox_name)),
        )
        getattr(self.m_ui, graphicsView_name).setObjectName(graphicsView_name)
        getattr(self.m_ui, graphicsView_name).setMinimumSize(QSize(256, 256))

        # self.graphicsView_cam4 = GraphicsLayoutWidget(self.groupBox_r2)
        # self.graphicsView_cam4.setObjectName(u"graphicsView_cam4")
        # self.graphicsView_cam4.setMinimumSize(QSize(256, 256))

        # self.verticalLayout_r2.addWidget(self.graphicsView_cam4)
        # self.verticalLayout_r.addWidget(self.groupBox_r2)

        getattr(self.m_ui, groupbox_layout_name).addWidget(getattr(self.m_ui, graphicsView_name))
        getattr(self.m_ui, f'verticalLayout_{side:s}').addWidget(getattr(self.m_ui, groupBox_name))

        getattr(self.m_ui, groupBox_name).setTitle(
            QCoreApplication.translate('Form', f'Camera {self.m_ui.num_views:d}', None)
        )

        # set 2d view tile color
        cam_color = self.meta_info['camera_colors'][self.m_ui.num_views - 1]
        getattr(self.m_ui, groupBox_name).setStyleSheet(f'QGroupBox:title {{color: rgb({cam_color[0]}, {cam_color[1]}, {cam_color[2]});}}')

        # add PlotItem for graphicsView
        layoutWidget = getattr(self.m_ui, graphicsView_name)
        view = layoutWidget.addViewBox()
        view.setAspectLocked(True)
        img = pg.ImageItem()
        view.addItem(img)
        self.item_dict['2d']['img_list'].append(img)

    def init_3d_widgets(self):
        # if 'gt3d' in self.data3d_group:
        #     self.gt_num_humans = self.data3d_group['gt3d'].shape[1]
        # else:
        #     self.gt_num_humans = 0

        self.max_num_humans = self.args.vis_num_human
        self.gt_num_humans = self.pred_num_humans = self.max_num_humans
        self.num_hypotheses = self.args.vis_num_hypo

        # self.pred_num_humans = self.data3d_group['pred3d'].shape[1]
        # self.max_num_humans = max(self.gt_num_humans, self.pred_num_humans)

        self.human_colors = self.meta_info['human_colors']

        self.item_dict['3d']['gt_human_line_list'] = [[] for _ in range(self.max_num_humans)]
        self.item_dict['3d']['pred_human_line_list'] = [[[] for _ in range(self.num_hypotheses)] for _ in range(self.max_num_humans)]

        for human_idx in range(self.max_num_humans):
            for _ in range(len(self.meta_info['skeleton'])):
                gt_human_line_item = gl.GLLinePlotItem(
                    pos=np.zeros((2, 3)),
                    color=[1.0, 1.0, 1.0] + [1.0],
                    width=3,
                    antialias=True,
                )
                self.item_dict['3d']['gt_human_line_list'][human_idx].append(gt_human_line_item)
                self.m_ui.openGLWidget.addItem(gt_human_line_item)

                for hypo_idx in range(self.num_hypotheses):
                    pred_human_line_item = gl.GLLinePlotItem(
                        pos=np.zeros((2, 3)),
                        color=[c / 255 for c in self.human_colors[0]] + [1.0],
                        width=3,
                        antialias=True,
                    )
                    self.item_dict['3d']['pred_human_line_list'][human_idx][hypo_idx].append(pred_human_line_item)
                    self.m_ui.openGLWidget.addItem(pred_human_line_item)

        # initialize camera iterm
        if 'camera' in self.data3d_group:
            self.max_num_cameras = self.data3d_group['camera'].shape[1]

            self.item_dict['3d']['camera_axes_list'] = [
                [] for _ in range(self.max_num_cameras)
            ]

            for camera_axes, color in zip(
                self.item_dict['3d']['camera_axes_list'], self.meta_info['camera_colors']
            ):
                for _ in range(4):
                    # 4 axes
                    line_item = gl.GLLinePlotItem(
                        pos=np.zeros((2, 3)),
                        color=pg.glColor(color),
                        width=3,
                        antialias=True,
                    )
                    camera_axes.append(line_item)
                    self.m_ui.openGLWidget.addItem(line_item)

                for _ in range(4):
                    # exra 4 axes, rectangle
                    line_item = gl.GLLinePlotItem(
                        pos=np.zeros((2, 3)),
                        color=pg.glColor(color),
                        width=3,
                        antialias=True,
                    )
                    camera_axes.append(line_item)
                    self.m_ui.openGLWidget.addItem(line_item)

        # # cylinder
        # md = gl.MeshData.cylinder(rows=10, cols=20, radius=[0.1, 20.], length=20.)
        # # md2 = gl.MeshData.cylinder(rows=10, cols=20, radius=[2., 0.5], length=10.)

        # colors = np.ones((md.faceCount(), 4), dtype=float)
        # colors[::2,0] = 0
        # colors[:,1] = np.linspace(0, 1, colors.shape[0])
        # md.setFaceColors(colors)
        # m5 = gl.GLMeshItem(meshdata=md, smooth=True, drawEdges=False, shader='balloon')
        # self.m_ui.openGLWidget.addItem(m5)
        # self.item_dict['3d']['marker'] = m5
        # self.item_dict['3d']['marker'].translate(*self.offset)

    def translate_map(self, offset):
        self.item_dict['3d']['floor'].translate(*offset)
        self.item_dict['3d']['axis_x'].translate(*offset)
        self.item_dict['3d']['axis_y'].translate(*offset)
        self.item_dict['3d']['axis_z'].translate(*offset)

        # adjust 3d viewpoint
        self.m_ui.openGLWidget.opts['center'] = Vector(*offset)
        # self.m_ui.openGLWidget.update()

    def get_interval_for_plot(self, data_dict):
        if 'gt3d' in data_dict:
            pose3d = data_dict['gt3d'][0, :, 0, ...]  # [b, j, 3]
        else:
            pose3d = data_dict['pred3d'][-1, :, 0, ...]  # [b, j, 3]
        width = np.mean(pose3d[..., 0].max(axis=-1) - pose3d[..., 0].min(axis=-1))
        return width * 5

    def openfile(self):
        filepath_list, filters = QFileDialog.getOpenFileNames(
            self, 'Choose File', '', 'Numpy Data(*.npz);;Videos (*.mp4 *.avi *.mov)'
        )

        if len(filepath_list) > 0:
            if 'video' in filters.lower():
                filepath_list.sort()
                for video_path in filepath_list:
                    self.video_list.append(self.read_video(video_path))
                # can relax this constraints for better tolerance
                assert all(len(self.video_list[0]) == len(v) for v in self.video_list)

                # add more views to ui if necessary
                while len(self.video_list) - self.m_ui.num_views > 0:
                    self.add_view()  # self.m_ui.num_views += 1 internally

                self.video_len = len(self.video_list[0])
                self._check_len()
                if self.current_frame != 0:
                    self.reset()  # implicitly call update()
                else:
                    self.update_2d_frames()

            if 'numpy' in filters.lower():
                assert len(filepath_list) == 1, 'Open just one npz file'
                self.data3d_group = self.read_numpy(filepath_list[0])
                self.data3d_group = self.meta_info['preprocess_func'](self.data3d_group)

                if 'pred3d' in self.data3d_group:
                    self.zarr_len = len(self.data3d_group['pred3d'])
                elif 'gt3d' in self.data3d_group:
                    self.zarr_len = len(self.data3d_group['gt3d'])
                else:
                    assert 0, 'Npz file does not contain 3d human data!'

                # human interval for plot
                self.interval_for_plot = self.get_interval_for_plot(self.data3d_group)

                # adjust map
                if 'map_center' in self.data3d_group:
                    offset = self.data3d_group['map_center']
                    self.translate_map(offset)

                self.init_3d_widgets()
                # self.init_plot()

                self._check_len()
                if self.current_frame != 0:
                    self.reset()
                else:
                    self.update_3d()
                    # self.update_plot()

    def _check_len(self):
        """
        Calculate an appropriate (min) len from loaded video and zarr for slider len.
        """
        if not hasattr(self, 'video_len') and not hasattr(self, 'zarr_len'):
            return

        if hasattr(self, 'video_len') and hasattr(self, 'zarr_len'):
            if self.video_len == self.zarr_len:
                self.data_len = self.zarr_len
            else:
                self.data_len = min(self.video_len, self.zarr_len)
                print(
                    'Warning:',
                    'video_len: %d' % self.video_len,
                    'zarr_len: %d' % self.zarr_len,
                    'data_len: %d' % self.data_len,
                )
                msgBox = QMessageBox()
                msgBox.setIcon(QMessageBox.Warning)
                msgBox.setText('Video len != Zarr len')
                msgBox.setInformativeText('Min len is set')
                msgBox.setDetailedText(
                    'video_len: %d\nzarr_len: %d\ndata_len: %d'
                    % (self.video_len, self.zarr_len, self.data_len)
                )
                msgBox.exec_()
        else:
            if hasattr(self, 'video_len'):
                self.data_len = self.video_len

            if hasattr(self, 'zarr_len'):
                self.data_len = self.zarr_len

        self.m_ui.label_numFrame.setText('/%d' % (self.data_len - 1))
        self.m_ui.horizontalSlider.setMaximum(self.data_len - 1)
        self.m_ui.spinBox_numFrame.setMaximum(self.data_len - 1)

    def read_video(self, video_path):
        """
        Some bugs of pims: connot open video with frame number == 2.
        """
        video = pims.PyAVVideoReader(video_path)  # indexing method is really slow
        return video

    # def read_zarr(self, path):
    #     return zarr.open_consolidated(path, mode='r')

    def read_numpy(self, path):
        data_dict = {}
        with np.load(path) as data:
            for k in data.files:
                data_dict[k] = data[k]

        return data_dict

    def reset(self):
        """
        This func calls update()
        """
        self.m_ui.spinBox_numFrame.setValue(0)
        self.m_ui.horizontalSlider.setValue(0)
        # self.current_frame = 0

    def update(self, value):
        if self.current_frame != value:
            self.current_frame = value

            # self.update_plot()

            self.update_3d()
            # t = threading.Thread(target=self.show_frame)
            # t.start()
            # self.update_2d_frames()
            # t.join()

    def update_2d_frames(self):
        if len(self.video_list) > 0:
            for video, cam_item in zip(self.video_list, self.item_dict['2d']['img_list']):
                cam_item.setImage(np.flip(video[self.current_frame], axis=0))

    def update_3d(self):
        if not hasattr(self, 'data3d_group'):
            return

        # plot pred 3d
        if 'pred3d' in self.data3d_group:
            pred3d_arr_plot = self.data3d_group['pred3d'][self.current_frame].copy()  # [N, j, 3]
            for human_idx in range(self.current_batch * self.max_num_humans,
                (self.current_batch + 1) * self.max_num_humans):
                if human_idx == pred3d_arr_plot.shape[0]:
                    break
                human_idx_this_frame = human_idx % self.max_num_humans
                for hypo_idx in range(min(pred3d_arr_plot.shape[1] ,self.num_hypotheses)):
                    for pair, pred_line in zip(self.meta_info['skeleton'],
                                             self.item_dict['3d']['pred_human_line_list'][human_idx_this_frame][hypo_idx]):
                        u, v = pair
                        s = pred3d_arr_plot[human_idx, hypo_idx, u]
                        e = pred3d_arr_plot[human_idx, hypo_idx, v]
                        pred_line_ends = np.vstack([s, e])
                        pred_line_ends[:, 0] += self.interval_for_plot * (human_idx_this_frame % self.args.vis_num_perline)  # offset on x
                        pred_line_ends[:, 1] += self.interval_for_plot * (human_idx_this_frame // self.args.vis_num_perline)
                        pred_line.setData(pos=pred_line_ends, antialias=True)

        # plot gt 3d
        if 'gt3d' in self.data3d_group and self.args.vis_gt:
            if len(self.data3d_group['gt3d']) == len(self.data3d_group['pred3d']):
                gt3d_arr_plot = self.data3d_group['gt3d'][self.current_frame].squeeze(1)  # [b, j, 3]
            else:
                # always frame -1
                gt3d_arr_plot = self.data3d_group['gt3d'][-1].squeeze(1)  # [b, j, 3]

            # # zero center
            # gt3d_arr_plot = gt3d_arr_plot - gt3d_arr_plot[:, [0], :]

            # # scale data
            # if np.amax(gt3d_arr_plot) > 100:
            #     gt3d_arr_plot /= 2344
            for human_idx in range(self.current_batch * self.max_num_humans,
                (self.current_batch + 1) * self.max_num_humans):
                if human_idx == gt3d_arr_plot.shape[0]:
                    break
                human_idx_this_frame = human_idx % self.max_num_humans
                for pair, gt_line in zip(self.meta_info['skeleton'],
                                         self.item_dict['3d']['gt_human_line_list'][human_idx_this_frame]):
                    u, v = pair
                    s = gt3d_arr_plot[human_idx, u]
                    e = gt3d_arr_plot[human_idx, v]
                    gt_line_ends = np.vstack([s, e])
                    gt_line_ends[:, 0] += self.interval_for_plot * (human_idx_this_frame % self.args.vis_num_perline)  # offset on x
                    gt_line_ends[:, 1] += self.interval_for_plot * (human_idx_this_frame // self.args.vis_num_perline)
                    gt_line.setData(pos=gt_line_ends, antialias=True)

        if 'camera' in self.data3d_group:
            # plot cameras
            for corners_world, camera_axes in zip(
                self.data3d_group['camera'][self.current_frame],
                self.item_dict['3d']['camera_axes_list'],
            ):
                corners_world_plot = corners_world.copy()
                # corners_world_plot[:, 1] *= -1  # y_qt = -y_unreal

                # scale the camera size
                scale = 1.5
                scaled_edge = (corners_world_plot[:-1] - corners_world_plot[-1]) * scale
                corners_world_plot[:-1] = scaled_edge + corners_world_plot[-1]

                cam_loc_plot = corners_world_plot[-1, ...]
                # cam_loc_plot = camera_param[:3].copy()
                # cam_loc_plot[1] *= -1

                for idx, corner in enumerate(corners_world_plot[:-1]):
                    cam_line_ends = np.vstack([cam_loc_plot, corner])
                    camera_axes[idx].setData(pos=cam_line_ends, antialias=True)

                # plot rectangle
                for idx, pair in enumerate([(0, 1), (0, 2), (1, 3), (2, 3)]):
                    cam_line_ends = np.vstack(
                        [corners_world_plot[pair[0]], corners_world_plot[pair[1]]]
                    )
                    camera_axes[-(idx + 1)].setData(pos=cam_line_ends, antialias=True)

        # # update marker
        # gt_target_position = gt3d_arr_plot[0, :, :2].mean(axis=0)
        # top_offset = self.offset + [gt_target_position[0], gt_target_position[1], 160]
        # self.item_dict['3d']['marker'].translate(*top_offset)

    def init_plot(self):
        """
        Init curve plot, called when openning files
        """
        if not hasattr(self, 'data3d_group'):
            return

        # compute statistics
        gt3d = self.data3d_group['gt3d'][:]  # [F, N, J, 3]
        pred3d = self.data3d_group['pred3d'][:]  # [F, N_max] ragged array of shape [J*3]
        pred3d_compared = np.zeros_like(gt3d)  # [F, N, J, 3]
        for frame_idx in range(gt3d.shape[0]):
            for human_idx in range(gt3d.shape[1]):
                if len(pred3d[frame_idx, human_idx]) == 0:
                    continue
                pred3d_compared[frame_idx, human_idx] = pred3d[frame_idx, human_idx].reshape(
                    (-1, 3)
                )

        error = np.linalg.norm(gt3d - pred3d_compared, axis=-1).mean(axis=-1)  # [F, N]
        error = np.swapaxes(error, 0, 1)  # [N, F]

        rec_ratio = pred3d_compared.any(axis=-1).mean(axis=-1)  # [F, N]
        rec_ratio = np.swapaxes(rec_ratio, 0, 1)  # [N, F]

        self.statistics_data = {}
        for idx, data in zip(self.graphicsView_plot_idx_list, [error, rec_ratio]):
            self.statistics_data[f'p{idx}'] = data

        # init common items
        self.person_names = [f'h{i}' for i in range(self.gt_num_humans)]  # record order
        self.symbols = [self.meta_info['symbols'][idx] for idx in range(self.gt_num_humans)]
        self.pens = [self.human_colors[idx] for idx in range(self.gt_num_humans)]

        for idx in self.graphicsView_plot_idx_list:
            self._init_single_plot(idx)

    def _init_single_plot(self, plot_idx):
        """
        Initialize a sinlge plot.
        """
        graphicsView = getattr(self.m_ui, f'graphicsView_plot{plot_idx}')

        self.item_dict['plot'][f'p{plot_idx}_label'] = pg.LabelItem(justify='right')
        graphicsView.addItem(self.item_dict['plot'][f'p{plot_idx}_label'])

        plot = CustomPlotItem(antialias=True)
        plot.create_custom_menu(self.gt_num_humans)
        plot.setID(plot_idx)
        plot.signal_with_id.connect(self.hide_curve)
        self.item_dict['plot'][f'plot{plot_idx}'] = plot
        graphicsView.addItem(plot, row=1, col=0)
        plot.addLegend()
        plot.setDownsampling(mode='peak')
        plot.setClipToView(True)
        plot.setAutoVisible(y=True)
        plot.setYRange(0, 60)  # clip abnormal values

        # vline
        vLine = pg.InfiniteLine(angle=90, movable=False, pen=(69, 230, 0))
        plot.addItem(vLine, ignoreBounds=True)
        self.item_dict['plot'][f'p{plot_idx}_vLine'] = vLine

        # plot marker
        self.item_dict['plot'][f'p{plot_idx}_symbol'] = pg.ScatterPlotItem(size=12)
        plot.addItem(self.item_dict['plot'][f'p{plot_idx}_symbol'])
        self.visibility_dict[f'p{plot_idx}_symbol'] = [True] * self.gt_num_humans

        # curve
        for pen, symbol, name, data in zip(self.pens, self.symbols,
            self.person_names, self.statistics_data[f'p{plot_idx}']):
            self.item_dict['plot'][f'p{plot_idx}_curve_' + name] = plot.plot(
                data, pen=pen, name=name, antialias=True
            )

        # hline
        for name, pen in zip(self.person_names, self.pens):
            hLine = pg.InfiniteLine(angle=0, movable=False, pen=pen)
            plot.addItem(hLine, ignoreBounds=True)
            self.item_dict['plot'][f'p{plot_idx}_hLine_' + name] = hLine

    def update_plot(self):
        # no data loaded or no plot inited
        if not hasattr(self, 'data3d_group'):
            return

        for idx in self.graphicsView_plot_idx_list:
            self._update_single_plot(idx)

    def _update_single_plot(self, plot_idx):
        """
        Update a sinlge plot.
        """
        data = self.statistics_data[f'p{plot_idx}']

        self.item_dict['plot'][f'p{plot_idx}_vLine'].setPos(self.current_frame)

        # update hline pos
        for idx, name in enumerate(self.person_names):
            self.item_dict['plot'][f'p{plot_idx}_hLine_' + name].setPos(data[idx, self.current_frame])

        format_str = (
            "<span style='font-size: 10pt'>x={:d}, "
            + "<span style='color: rgb{:s}'>{:s}={:.1f}</span>, " * self.gt_num_humans
        )

        param_list = [self.current_frame]
        for human_idx in range(self.gt_num_humans):
            param_list.extend(
                [
                    str(tuple(self.pens[human_idx])),
                    self.person_names[human_idx],
                    data[human_idx, self.current_frame],
                ]
            )

        self.item_dict['plot'][f'p{plot_idx}_label'].setText(format_str.format(*param_list))

        # update symbols
        symbos_pos = np.concatenate(
            (
                np.full((len(self.person_names), 1), self.current_frame),
                data[:, [self.current_frame]],
            ),
            axis=1,
        )  # [N, 2]
        visibility = self.visibility_dict[f'p{plot_idx}_symbol']  # list
        if len(visibility):
            pen = ['w' if vis else None for vis in visibility]
            brush = [self.pens[idx] if vis else None for idx, vis in enumerate(visibility)]
        else:
            # init
            pen = 'w'
            brush = self.pens
        self.item_dict['plot'][f'p{plot_idx}_symbol'].setData(
            pos=symbos_pos, pen=pen, brush=brush, symbol=self.symbols
        )

    def hide_curve(self, args_list):
        """
        args_list: [human_idx (checkbox_id), state, plot_id]
        """
        human_idx, state, plot_idx = args_list
        # plot = getattr(self, f'plot{plot_idx}')

        if f'plot{plot_idx}' not in self.item_dict['plot']:
            warnings.warn(f'does not have plot{plot_idx}')
            return

        if human_idx >= self.gt_num_humans:
            warnings.warn(f'does not have curver: h{human_idx}')
            return

        curve_item = self.item_dict['plot'].get(f'p{plot_idx}_curve_' + self.person_names[human_idx])
        if curve_item is None:
            return

        # show/hide curves
        if state == Qt.Unchecked:
            curve_item.setPen(None)
        else:
            curve_item.setPen(self.pens[human_idx])

        # show/hide symbols
        symbols_item = self.item_dict['plot'].get(f'p{plot_idx}_symbol')
        if symbols_item is None:
            return

        if state == Qt.Unchecked:
            self.visibility_dict[f'p{plot_idx}_symbol'][human_idx] = False
        else:
            self.visibility_dict[f'p{plot_idx}_symbol'][human_idx] = True

        # hide once
        # bus here due to pyqtgraph update
        symbols_item.setPointsVisible(self.visibility_dict[f'p{plot_idx}_symbol'])

        # show/hide hline
        hline_item = self.item_dict['plot'].get(f'p{plot_idx}_hLine_h{human_idx}')
        if hline_item is None:
            pass
        else:
            # show/hide curves
            if state == Qt.Unchecked:
                hline_item.setPen(None)
            else:
                hline_item.setPen(self.pens[human_idx])

    def backward_batch(self):
        if not hasattr(self, 'data3d_group'):
            return

        # if 'pred3d' not in self.data3d_group:
        #     return

        if self.current_batch >= 1:
            self.current_batch -= 1
            self.update_3d()

    def forward_batch(self):
        if not hasattr(self, 'data3d_group'):
            return

        # if 'pred3d' not in self.data3d_group:
        #     return

        # ceiling division
        max_batch = -(-self.data3d_group['pred3d'].shape[1] // self.max_num_humans)

        if self.current_batch < max_batch - 1:
            self.current_batch += 1
            self.update_3d()
