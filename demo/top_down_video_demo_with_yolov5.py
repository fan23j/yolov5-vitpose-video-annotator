# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser

import cv2

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo
import torch
from tqdm import tqdm
import json
import pudb
import numpy as np

# COCO Keypoints mapping
coco_keypoints = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle',
}

def smooth(pose_results, pose_buffer, num_frames=2):
    """
    Simple pose smoothing that simply averages pose_results with
    results from previous num_frames.
    """
    # [[x,y,score],...] - 17 kpts
    current_pose = pose_results[0]['keypoints'].tolist()
    
    # populate head of pose_buffer or shift window
    if len(pose_buffer) < num_frames:
        pose_buffer.append(current_pose)
    else:
        pose_buffer.pop(0)
        # sum x,y of current_pose with poses from previous frames
        for pose in pose_buffer:
            for kpt in coco_keypoints:
                current_pose[kpt][0] += pose[kpt][0]
                current_pose[kpt][1] += pose[kpt][1]
        
        # average
        for kpt in current_pose:
            kpt[0] /= (num_frames)
            kpt[1] /= (num_frames)
        
        pose_buffer.append(current_pose)
        pose_results[0]['keypoints'] = np.array(current_pose)
    
    return pose_results

def main():
    """Visualize the demo video.

    Using yolov5 to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('--pose-config', help='Config file for pose')
    parser.add_argument('--pose-checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    args = parser.parse_args()

    assert args.show or (args.out_video_root != '')
    
    # initialize pose buffer for simple pose smoothing
    pose_buffer = []

    # initialize yolov5 detector
    detector = torch.hub.load("ultralytics/yolov5", 'yolov5l6')
    
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    cap = cv2.VideoCapture(args.video_path)
    assert cap.isOpened(), f'Faild to load video file {args.video_path}'

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fps = cap.get(cv2.CAP_PROP_FPS)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'vis_{os.path.basename(args.video_path)}'), fourcc,
            fps, size)

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None
    
    # initialize results json
    results = []

    for i in tqdm(range(0, length)):
        flag, img = cap.read()
        if not flag:
            break
        # test a single image, the resulting box is (x1, y1, x2, y2)
        dets = detector(img)
        
        # process results
        df = dets.pandas().xyxy[0]
        df = df[df['name'] == 'person']
        detections = df.iloc[:, :5].values.tolist()

        # initialize person detections list
        person_detections = []
        
        # filter detections results
        for bbox in detections:
            if bbox[4] > args.bbox_thr:
                person_detections.append({'bbox': bbox})

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_detections,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)
        
        if len(pose_results) == 0:
            continue
        
        # smooth results
        # pose_results = smooth(pose_results, pose_buffer)

        # show the results
        vis_img = vis_pose_result(
            pose_model,
            img,
            pose_results,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            radius=args.radius,
            thickness=args.thickness,
            show=False)
        

        # add to results json
        results.append({
            "image_id": str(i) +'.jpg',
            "category_id": 1,
            "keypoints": list(np.concatenate(pose_results[0]['keypoints'].tolist()).flat),
        })

        if args.show:
            cv2.imshow('Image', vis_img)

        if save_out_video:
            videoWriter.write(vis_img)

    cap.release()
    if save_out_video:
        videoWriter.release()
    if args.show:
        cv2.destroyAllWindows()
    # write results json
    with open(os.path.join(args.out_video_root, f'vis_{os.path.basename(args.video_path)}.json'), 'w') as out_file:
        json.dump(results, out_file)


if __name__ == '__main__':
    main()
