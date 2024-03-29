import cv2
import os
import numpy as np


def gen_reconstructed_pose_on_image(pose_in, pose_rec, pose_gt, edge, bbox,
                                    image_path=None, st_mage_name=None, height=1080):
    """
    Args:

    Shapes:
        pose (3, T, V)
        bbox (4, T, V)
    Returns:

    """

    C, T, V = pose_in.shape  # C, T, V
    padding_bbox = 3

    if image_path is not None:
        image_list = os.listdir(image_path)
        st_index = image_list.index(st_mage_name)

    for t in range(T):

        # read image
        if image_path is not None:
            file_name = os.path.join(image_path, image_list[st_index + t])
            frame = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
        else:
            frame = np.ones(shape=[512, 512, 3], dtype=np.uint8)  # white background
            frame[:, 0, :] = 0; frame[:, -1, :] = 0; frame[0, :, :] = 0; frame[-1, :, :] = 0

        # image resize
        H, W, c = frame.shape
        frame = cv2.resize(frame, (height * W // H // 2, height // 2))
        H, W, c = frame.shape
        scale_factor = 2 * height / 1080

        # draw skeleton in
        skeleton = frame.copy()
        for i, j in edge:
            xi = pose_in[0, t, i]
            yi = pose_in[1, t, i]
            xj = pose_in[0, t, j]
            yj = pose_in[1, t, j]
            if xi + yi == 0 or xj + yj == 0:
                continue
            else:
                xi = int(((xi + 0.5) * (bbox[t, 2] - bbox[t, 0]) + bbox[t, 0]) * W)
                yi = int(((yi + 0.5) * (bbox[t, 3] - bbox[t, 1]) + bbox[t, 1]) * H)
                xj = int(((xj + 0.5) * (bbox[t, 2] - bbox[t, 0]) + bbox[t, 0]) * W)
                yj = int(((yj + 0.5) * (bbox[t, 3] - bbox[t, 1]) + bbox[t, 1]) * H)
            cv2.line(skeleton, (xi, yi), (xj, yj), (0, 255, 255),
                     int(np.ceil(2 * scale_factor)))  # noisy pose in black

        # draw skeleton rec
        skeleton_rec = frame.copy()
        for i, j in edge:
            xi = pose_rec[0, t, i]
            yi = pose_rec[1, t, i]
            xj = pose_rec[0, t, j]
            yj = pose_rec[1, t, j]
            if xi + yi == 0 or xj + yj == 0:
                continue
            else:
                xi = int(((xi + 0.5) * (bbox[t, 2] - bbox[t, 0]) + bbox[t, 0]) * W)
                yi = int(((yi + 0.5) * (bbox[t, 3] - bbox[t, 1]) + bbox[t, 1]) * H)
                xj = int(((xj + 0.5) * (bbox[t, 2] - bbox[t, 0]) + bbox[t, 0]) * W)
                yj = int(((yj + 0.5) * (bbox[t, 3] - bbox[t, 1]) + bbox[t,1]) * H)
            cv2.line(skeleton_rec, (xi, yi), (xj, yj), (0, 255, 0),
                     int(np.ceil(2 * scale_factor)))  # reconstructed pose in blue

        # draw skeleton gt
        skeleton_gt = frame.copy()
        for i, j in edge:
            xi = pose_gt[0, t, i]
            yi = pose_gt[1, t, i]
            xj = pose_gt[0, t, j]
            yj = pose_gt[1, t, j]
            if xi + yi == 0 or xj + yj == 0:
                continue
            else:
                xi = int(((xi + 0.5) * (bbox[t, 2] - bbox[t, 0]) + bbox[t, 0]) * W)
                yi = int(((yi + 0.5) * (bbox[t, 3] - bbox[t, 1]) + bbox[t, 1]) * H)
                xj = int(((xj + 0.5) * (bbox[t, 2] - bbox[t, 0]) + bbox[t, 0]) * W)
                yj = int(((yj + 0.5) * (bbox[t, 3] - bbox[t, 1]) + bbox[t, 1]) * H)
            cv2.line(skeleton_gt, (xi, yi), (xj, yj), (0, 0, 255),
                     int(np.ceil(2 * scale_factor)))  # gt pose in red

        # draw bbox
        xtl = int(bbox[t, 0]*W) - padding_bbox
        ytl = int(bbox[t, 1]*H) - padding_bbox
        xbl = int(bbox[t, 2]*W) + padding_bbox
        ybl = int(bbox[t, 3]*H) + padding_bbox
        skeleton = cv2.rectangle(skeleton, (xtl, ytl), (xbl, ybl),  (255, 0, 0), 2)
        skeleton_rec = cv2.rectangle(skeleton_rec, (xtl, ytl), (xbl, ybl),  (255, 0, 0), 2)
        skeleton_gt = cv2.rectangle(skeleton_gt, (xtl, ytl), (xbl, ybl),  (255, 0, 0), 2)

        img0 = np.concatenate((frame, skeleton), axis=1)
        img1 = np.concatenate((skeleton_rec, skeleton_gt), axis=1)
        img = np.concatenate((img0, img1), axis=0)

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
