import cv2
import os
import numpy as np


def gen_traj_on_image(pose_in,
                      rec_pose,
                      pred_loc,
                      gt_loc,
                      bbox,
                      edge, image_path=None, st_mage_name=None, obs_len=10, height=1080):
    """
    Args:

    Shapes:
        pose: (C, V)
        bbox  (4, V)
        pred_loc, gt_loc: (T, 2)
    Returns:

    """
    C, V = pose_in.shape  # C, V
    image_list = os.listdir(image_path)
    image_list = sorted(image_list)
    st_index = image_list.index(st_mage_name)

    # read image
    if image_path is not None:
        file_name = os.path.join(image_path, image_list[st_index + obs_len])
        frame = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    else:
        frame = np.ones(shape=[512, 512, 3], dtype=np.uint8)  # white background
        frame[:, 0, :] = 0;
        frame[:, -1, :] = 0;
        frame[0, :, :] = 0;
        frame[-1, :, :] = 0

    # image resize
    H, W, c = frame.shape
    frame = cv2.resize(frame, (height * W // H // 2, height // 2))
    H, W, c = frame.shape
    scale_factor = 2 * height / 1080

    # draw skeleton in
    skeleton = frame.copy()
    for i, j in edge:
        xi = pose_in[0, i]
        yi = pose_in[1, i]
        xj = pose_in[0, j]
        yj = pose_in[1, j]
        if xi + yi == 0 or xj + yj == 0:
            continue
        else:
            xi = int(((xi + 0.5) * (bbox[2] - bbox[0]) + bbox[0]) * W)
            yi = int(((yi + 0.5) * (bbox[3] - bbox[1]) + bbox[1]) * H)
            xj = int(((xj + 0.5) * (bbox[2] - bbox[0]) + bbox[0]) * W)
            yj = int(((yj + 0.5) * (bbox[3] - bbox[1]) + bbox[1]) * H)
        cv2.line(skeleton, (xi, yi), (xj, yj), (0, 255, 255), 1)  # noisy pose in black

    # draw bbox
    xtl = int(bbox[0] * W)
    ytl = int(bbox[1] * H)
    xbl = int(bbox[2] * W)
    ybl = int(bbox[3] * H)
    skeleton = cv2.rectangle(skeleton, (xtl, ytl), (xbl, ybl), (255, 0, 0), 2)

    # draw gt trajectory
    gt_loc[:, 0] = gt_loc[:, 0] * W
    gt_loc[:, 1] = gt_loc[:, 1] * H
    for t in range(gt_loc.shape[0] - 1):
        cv2.line(skeleton, (gt_loc[t, 0], gt_loc[t, 1]), (gt_loc[t + 1, 0], gt_loc[t + 1, 1]), (0, 0, 255), 2)

    # draw pred trajectory
    pred_loc[:, 0] = pred_loc[:, 0]* W
    pred_loc[:, 1] = pred_loc[:, 1]* H
    for t in range(pred_loc.shape[0] - 1):
        cv2.line(skeleton, (pred_loc[t, 0], pred_loc[t, 1]), (pred_loc[t + 1, 0], pred_loc[t + 1, 1]), (0, 255, 0), 2)

    # draw reconstructed pose
    rec_skeleton = frame.copy()
    for i, j in edge:
        xi = rec_pose[0, i]
        yi = rec_pose[1, i]
        xj = rec_pose[0, j]
        yj = rec_pose[1, j]
        if xi + yi == 0 or xj + yj == 0:
            continue
        else:
            xi = int(((xi + 0.5) * (bbox[2] - bbox[0]) + bbox[0]) * W)
            yi = int(((yi + 0.5) * (bbox[3] - bbox[1]) + bbox[1]) * H)
            xj = int(((xj + 0.5) * (bbox[2] - bbox[0]) + bbox[0]) * W)
            yj = int(((yj + 0.5) * (bbox[3] - bbox[1]) + bbox[1]) * H)
        cv2.line(rec_skeleton, (xi, yi), (xj, yj), (0, 255, 255), 1)  # noisy pose in black

    # draw bbox
    rec_skeleton = cv2.rectangle(rec_skeleton, (xtl, ytl), (xbl, ybl), (255, 0, 0), 2)

    img = np.concatenate((skeleton, rec_skeleton), axis=0)
    yield img


def put_text(img, text, position, scale_factor=1):
    t_w, t_h = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_TRIPLEX, scale_factor, thickness=1)[0]
    H, W, _ = img.shape
    position = (int(W * position[1] - t_w * 0.5),
                int(H * position[0] - t_h * 0.5))
    params = (position, cv2.FONT_HERSHEY_TRIPLEX, scale_factor,
              (0, 0, 0))
    cv2.putText(img, text, *params)


def blend(background, foreground, dx=20, dy=10, fy=0.7):
    foreground = cv2.resize(foreground, (0, 0), fx=fy, fy=fy)
    h, w = foreground.shape[:2]
    b, g, r, a = cv2.split(foreground)
    mask = np.dstack((a, a, a))
    rgb = np.dstack((b, g, r))

    canvas = background[-h - dy:-dy, dx:w + dx]
    imask = mask > 0
    canvas[imask] = rgb[imask]
