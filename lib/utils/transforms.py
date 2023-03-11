import numpy as np

# ======================== 3D =======================================

def world_to_camera_frame(P, R, T):
    """
    Convert points from world to camera coordinates

    Args
        P: Nx3 3d points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
    Returns
        X_cam: Nx3 3d points in camera coordinates
    """

    assert len(P.shape) == 2
    assert P.shape[1] == 3

    X_cam = R.dot( P.T - T ) # rotate and translate

    return X_cam.T

def camera_to_world_frame(P, R, T):
    """Inverse of world_to_camera_frame

    Args
        P: Nx3 points in camera coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
    Returns
        X_cam: Nx3 points in world coordinates
    """

    assert len(P.shape) == 2
    assert P.shape[1] == 3

    X_cam = R.T.dot( P.T ) + T # rotate and translate

    return X_cam.T

def procrustes(A, B, scaling=True, reflection='best'):
    """ A port of MATLAB's `procrustes` function to Numpy.

    $$ \min_{R, T, S} \sum_i^N || A_i - R B_i + T ||^2. $$
    Use notation from [course note]
    (https://fling.seas.upenn.edu/~cis390/dynamic/slides/CIS390_Lecture11.pdf).

    Args:
        A: Matrices of target coordinates.
        B: Matrices of input coordinates. Must have equal numbers of  points
            (rows), but B may have fewer dimensions (columns) than A.
        scaling: if False, the scaling component of the transformation is forced
            to 1
        reflection:
            if 'best' (default), the transformation solution may or may not
            include a reflection component, depending on which fits the data
            best. setting reflection to True or False forces a solution with
            reflection or no reflection respectively.

    Returns:
        d: The residual sum of squared errors, normalized according to a measure
            of the scale of A, ((A - A.mean(0))**2).sum().
        Z: The matrix of transformed B-values.
        tform: A dict specifying the rotation, translation and scaling that
            maps A --> B.
    """
    assert A.shape[0] == B.shape[0]
    n, dim_x = A.shape
    _, dim_y = B.shape

    # remove translation
    A_bar = A.mean(0)
    B_bar = B.mean(0)
    A0 = A - A_bar
    B0 = B - B_bar

    # remove scale
    ssX = (A0**2).sum()
    ssY = (B0**2).sum()
    A_norm = np.sqrt(ssX)
    B_norm = np.sqrt(ssY)
    A0 /= A_norm
    B0 /= B_norm

    if dim_y < dim_x:
        B0 = np.concatenate((B0, np.zeros(n, dim_x - dim_y)), 0)

    # optimum rotation matrix of B
    A = np.dot(A0.T, B0)
    U, s, Vt = np.linalg.svd(A)
    V = Vt.T
    R = np.dot(V, U.T)

    if reflection != 'best':
        # does the current solution use a reflection?
        have_reflection = np.linalg.det(R) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            R = np.dot(V, U.T)

    S_trace = s.sum()
    if scaling:
        # optimum scaling of B
        scale = S_trace * A_norm / B_norm

        # standarised distance between A and scale*B*R + c
        d = 1 - S_trace**2

        # transformed coords
        Z = A_norm * S_trace * np.dot(B0, R) + A_bar
    else:
        scale = 1
        d = 1 + ssY / ssX - 2 * S_trace * B_norm / A_norm
        Z = B_norm * np.dot(B0, R) + A_bar

    # transformation matrix
    if dim_y < dim_x:
        R = R[:dim_y, :]
    translation = A_bar - scale * np.dot(B_bar, R)

    # transformation values
    tform = {'rotation': R, 'scale': scale, 'translation': translation}
    return d, Z, tform

def image_to_camera_frame(pose3d_image_frame, box, camera, rootIdx, root_depth):
    rectangle_3d_size = 2000.0
    ratio = (box[2] - box[0] + 1) / rectangle_3d_size
    pose3d_image_frame = pose3d_image_frame.copy()
    pose3d_image_frame[:, 2] = pose3d_image_frame[:, 2] / ratio + root_depth

    cx, cy, fx, fy = camera['cx'], camera['cy'], camera['fx'], camera['fy']
    pose3d_image_frame[:, 0] = (pose3d_image_frame[:, 0] - cx) / fx
    pose3d_image_frame[:, 1] = (pose3d_image_frame[:, 1] - cy) / fy
    pose3d_image_frame[:, 0] *= pose3d_image_frame[:, 2]
    pose3d_image_frame[:, 1] *= pose3d_image_frame[:, 2]
    return pose3d_image_frame


def align_to_gt(pose, pose_gt):
    """Align pose to ground truth pose.

    Use MLE.
    """
    return procrustes(pose_gt, pose)[1]

