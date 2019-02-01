import os
import sys
import argparse
import cv2
import math
import time
import numpy as np
from util import padRightDownCorner, processBar
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model.cmu_model import get_testing_model

# find connection in the specified sequence
limbSeq = [[3, 2], [2, 0], [2, 1], [3, 4], [3, 5], \
           [3, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], \
           [3, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17]]

# the sort order of mapIdx encodes the map from joint_pairs in training into limbSeq above
mapIdx = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], \
          [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21], \
          [22, 23], [24, 25], [26, 27], [28, 29], [30, 31], [32, 33]]

# visualize
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], \
          [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], \
          [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


def process(input_image, params, model_params):
    oriImg = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    multiplier = [x * model_params['boxsize'] / oriImg.shape[0] for x in scale_search]

    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], num_joints_and_bkg))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], num_paf))

    for m in range(len(multiplier)):
        scale = multiplier[m]

        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = padRightDownCorner(imageToTest, model_params['stride'],
                                                     model_params['padValue'])

        input_img = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]),
                                 (3, 0, 1, 2))  # required shape (1, width, height, channels)

        output_blobs = model.predict(input_img)

        # extract outputs, resize, and remove padding
        heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3],
                  :]
        heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
        paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0] - pad[2], :imageToTest_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

    all_peaks = []
    peak_counter = 0

    for part in range(num_joints):
        map_ori = heatmap_avg[:, :, part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]

        peaks_binary = np.logical_and.reduce(
            (map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > params['thre1']))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(num_connections):
        score_mid = paf_avg[:, :, [x for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0]]
        candB = all_peaks[limbSeq[k][1]]
        nA = len(candA)
        nB = len(candB)
        if (nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    # failure case when 2 body parts overlaps
                    if norm == 0:
                        continue
                    vec = np.divide(vec, norm)

                    startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                        np.linspace(candA[i][1], candB[j][1], num=mid_num)))

                    vec_x = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                         for I in range(len(startend))])
                    vec_y = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                         for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        0.5 * oriImg.shape[0] / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > params['thre2'])[0]) > 0.8 * len(
                        score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior,
                                                     score_with_dist_prior + candA[i][2] + candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if (i not in connection[:, 3] and j not in connection[:, 4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if (len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, num_joints + 2))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(num_connections):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = limbSeq[k]

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if (subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + \
                              connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    deleteIdx = []
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)

    canvas = input_image.copy()
    # draw connections
    stickwidth = min(max(int(4 * resize_fac), 1), 4)
    conn_count = 0
    for i in range(num_connections):
        for n in range(len(subset)):
            index = subset[n][limbSeq[i]]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)),
                                       (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i])
            conn_count += 1
    # draw keypoints
    for i in range(num_joints):
        for peak in all_peaks[i]:
            cv2.circle(canvas, peak[0:2], 4, colors[i], thickness=-1)
            cv2.putText(canvas, model_params['part_str'][i], peak[0:2], cv2.FONT_HERSHEY_DUPLEX, 0.3,
                        [255, 255, 255], thickness=1)
    # mix
    canvas = cv2.addWeighted(canvas, 0.6, input_image, 0.4, 0)
    if conn_count < 5:
        return False, canvas
    else:
        return True, canvas


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=str, help='input video file name')
    parser.add_argument('-m', '--model', type=str, default='model/keras/mymodel.h5',
                        help='path to the weights file')
    parser.add_argument('-r', '--resize_factor', type=float, default=1, help='minification factor')
    parser.add_argument('-f', '--frame_ratio', type=int, default=1, help='analyze every [n] frames')
    parser.add_argument('-p', '--process_speed', type=int, default=2,
                        help='Int 1 (fastest, lowest quality) to 4 (slowest, highest quality)')
    parser.add_argument('-e', '--end', type=int, default=None, help='Last video frame to analyze')
    parser.add_argument('-s', '--show', action='store_true', help='show video canvas')
    args = parser.parse_args()

    start_datetime = time.strftime("-%m%d%H%M%S", time.localtime())

    dataset_save_dir = 'dataset/gen-by-model'
    os.makedirs(dataset_save_dir, exist_ok=True)

    keras_weights_file = args.model
    frame_rate_ratio = args.frame_ratio
    process_speed = args.process_speed
    resize_fac = args.resize_factor

    print('start processing...')

    # Video input & output
    video = args.video
    if not os.path.exists(video):
        raise FileNotFoundError("File not exist in {}".format(video))
    video_files = {"input": [], "output": []}
    output_dir = 'outputs/video'
    output_format = '.mp4'
    if os.path.isdir(video):
        for root, dirs, files in os.walk(video):
            rel_path = os.path.relpath(root, video)
            for file in files:
                name, ext = os.path.splitext(file)
                if ext in ['.mp4', '.avi', '.mpg', '.mpeg', '.3gp']:
                    input_path = os.path.join(root, file)
                    output_path = os.path.join(output_dir + start_datetime, rel_path, name + output_format)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    video_files["input"].append(input_path)
                    video_files["output"].append(output_path)
    else:
        name, ext = os.path.splitext(video)
        assert ext in ['.mp4', '.avi', '.mpg', '.mpeg', '.3gp']
        output_path = os.path.join(output_dir, os.path.basename(name) + start_datetime + output_format)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        video_files["input"].append(video)
        video_files["output"].append(output_path)

    # load model
    print('[*]Loading model...')
    model = get_testing_model()
    model.load_weights(keras_weights_file)

    # load config
    print('[*]Loading config...')
    params, model_params = config_reader()
    scale_search = params['scale_search']  # [0.5, 1, 1.5, 2]
    scale_search = scale_search[0:process_speed]
    num_joints = len(model_params['part_str'])  # 18
    num_joints_and_bkg = num_joints + 1  # 19
    num_connections = len(model_params['joint_pairs'])  # 17
    num_paf = 2 * num_connections  # 34

    assert num_connections == len(limbSeq) and num_connections == len(mapIdx)

    # Video reader
    for input_video, output_video in zip(video_files["input"], video_files["output"]):
        print('[*]Process video {} into {}'.format(input_video, output_video))
        cam = cv2.VideoCapture(input_video)
        input_fps = cam.get(cv2.CAP_PROP_FPS)
        ret_val, input_image = cam.read()
        if not ret_val:
            continue
        video_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

        ending_frame = args.end
        if ending_frame is None:
            ending_frame = video_length

        # Video writer
        output_fps = input_fps / frame_rate_ratio
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, output_fps, (input_image.shape[1], input_image.shape[0]))

        i = 0  # default is 0
        dsidx = len(os.listdir(dataset_save_dir)) + 1
        tst = time.time()
        while (cam.isOpened()) and ret_val and i < ending_frame:
            if i % frame_rate_ratio == 0:
                tic = time.time()
                # generate image with body parts
                resized_image = cv2.resize(input_image, (0, 0), fx=1 * resize_fac, fy=1 * resize_fac,
                                           interpolation=cv2.INTER_CUBIC)
                ret, canvas = process(resized_image, params, model_params)
                if not ret: # no completion people
                    save_path = os.path.join(dataset_save_dir,
                                             'id{}_{}_frame{}.jpg'.format(dsidx, os.path.split(output_video)[1], i))
                    save_path_view = os.path.join(
                        dataset_save_dir, 'id{}_{}_frame{}_view.jpg'.format(dsidx, os.path.split(output_video)[1], i))
                    cv2.imwrite(save_path, input_image)
                    cv2.imwrite(save_path_view, canvas)
                    dsidx += 1
                if args.show:
                    cv2.imshow('preview', canvas)
                    cv2.waitKey(1)
                toc = time.time()
                processBar(i, ending_frame,
                           '{}/{}, process time:{:.3f}, total time:{:.3f}'.format(i, ending_frame, (toc - tic),
                                                                                  (toc - tst)), length=20)
                out.write(canvas)
            ret_val, input_image = cam.read()
            i += 1
        out.release()
        cv2.destroyAllWindows()
        processBar(ending_frame, ending_frame,
                   '{}/{}, total time:{:.3f}'.format(i, ending_frame, (time.time() - tst)),
                   length=20)
