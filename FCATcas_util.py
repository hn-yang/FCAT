import os
import dlib  # Dlib library for face detection / facial landmarks
import numpy as np  # NumPy for numerical computing
import cv2  # OpenCV for image processing
import math
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
# import try_emd
from PyEMD import EMD, EEMD, CEEMDAN

detector = dlib.get_frontal_face_detector()  # Get the frontal face detector
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Load the 68-point landmark predictor

# Dlib detector and landmark predictor
font = cv2.FONT_HERSHEY_SIMPLEX
landmark0 = []


def temporal_ideal_filter(tensor, low, high, fps, axis=0):
    """Apply an ideal temporal filter in the frequency domain (current implementation mainly uses `high`)."""
    signal_fft = fftpack.fft(tensor, axis=axis)

    frequencies = fftpack.fftfreq(tensor.shape[0], d=1.0 / fps)

    bound_low = (np.abs(frequencies - low)).argmin()
    bound_high = (np.abs(frequencies - high)).argmin()

    signal_fft[bound_high:-bound_high] = 0
    # signal_fft[-bound_low:-1] = 0

    filtered_signal = fftpack.ifft(signal_fft, axis=axis)
    return np.abs(filtered_signal)

def compute_emd_imf_sums(original_signal, signal, output_path, sequence_id, sampling_rate):
    """
    Run EMD on `signal` and return two IMF sums:
    - imf_sum_excluding_first: sum of IMFs excluding the 1st IMF (index 0)
    - imf_sum_excluding_last:  sum of IMFs excluding the last IMF
    """
    # Build time vector (keeps the original formula to avoid changing behavior)
    time_vector = np.arange(len(signal) / sampling_rate)

    signal = np.array(signal)

    imfs = EMD().emd(signal, time_vector)
    # imfs = EEMD().eemd(signal, time_vector)
    # imfs = CEEMDAN().ceemdan(signal, time_vector)

    num_imfs = imfs.shape[0]

    imf_sum_excluding_first = np.zeros(imfs.shape[1])
    imf_sum_excluding_last = np.zeros(imfs.shape[1])

    for imf_index, imf in enumerate(imfs):
        if imf_index != num_imfs - 1:
            imf_sum_excluding_last = np.add(imf_sum_excluding_last, imf)
        if imf_index != 0:
            imf_sum_excluding_first = np.add(imf_sum_excluding_first, imf)

    return imf_sum_excluding_first, imf_sum_excluding_last




def crop_face_region(image, target_size):
    """Detect face landmarks and crop a face ROI based on eye landmarks.

    Returns:
        landmarks_68: 68-point facial landmarks as a numpy matrix (shape: 68x2)
        cropped_resized: cropped face ROI resized to (target_size, target_size)
        top, bottom, left, right: crop box coordinates in the original image
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Detect faces
    face_rects = detector(gray_image, 0)

    # Detect 68 landmarks (use the last detected face to match original behavior)
    landmarks_68 = None
    for face_rect in face_rects:
        landmarks_68 = np.matrix([[p.x, p.y] for p in predictor(image, face_rect).parts()])

    if len(face_rects) != 0:
        cv2.imwrite("img_gray.jpg", gray_image)
    else:
        raise ValueError("Cannot find even one face")

    # Eye corner landmarks (Dlib 68-point model)
    left_eye_corner = landmarks_68[39]
    right_eye_corner = landmarks_68[42]

    # Half of the horizontal distance between the two eye corners
    eye_half_width = int((right_eye_corner[0, 0] - left_eye_corner[0, 0]) / 2)

    eye_center = [
        int((right_eye_corner[0, 0] + left_eye_corner[0, 0]) / 2),
        int((right_eye_corner[0, 1] + left_eye_corner[0, 1]) / 2),
    ]

    # Draw ROI rectangle (kept from original code)
    cv2.rectangle(
        image,
        (eye_center[0] - int(4.5 * eye_half_width), eye_center[1] - int(3.5 * eye_half_width)),
        (eye_center[0] + int(4.5 * eye_half_width), eye_center[1] + int(5.5 * eye_half_width)),
        (0, 0, 255),
        2,
    )

    # Compute crop box
    top = eye_center[1] - int(3 * eye_half_width)
    bottom = eye_center[1] + int(5 * eye_half_width)
    left = eye_center[0] - int(4 * eye_half_width)
    right = eye_center[0] + int(4 * eye_half_width)

    # Clamp to image bounds (kept consistent with original code: only clamps top/left)
    top = max(top, 0)
    left = max(left, 0)

    cropped = image[top:bottom, left:right]
    cropped_resized = cv2.resize(cropped, (target_size, target_size))

    return landmarks_68, cropped_resized, top, bottom, left, right



def get_roi_bounds(start_idx, end_idx, padding, landmarks_68):
    """Compute ROI bounds (x_max, x_min, y_min, y_max) for a slice of landmarks.

    Args:
        start_idx: Start index (inclusive) of landmark slice.
        end_idx: End index (exclusive) of landmark slice.
        padding: Extra pixels to expand the ROI in all directions.
        landmarks_68: Landmark array/matrix where col0 is x and col1 is y (shape: Nx2).

    Returns:
        x_max_padded, x_min_padded, y_min_padded, y_max_padded
    """
    roi_points = landmarks_68[start_idx:end_idx]

    # Find indices of extreme points
    x_max_idx = roi_points[:, 0].argmax(axis=0)
    x_min_idx = roi_points[:, 0].argmin(axis=0)
    y_min_idx = roi_points[:, 1].argmin(axis=0)
    y_max_idx = roi_points[:, 1].argmax(axis=0)

    # Extract extreme coordinates
    x_max = roi_points[x_max_idx, 0]
    x_min = roi_points[x_min_idx, 0]
    y_min = roi_points[y_min_idx, 1]
    y_max = roi_points[y_max_idx, 1]

    # Apply padding and convert to Python ints
    x_max_padded = int((x_max + padding)[0, 0])
    x_min_padded = int((x_min - padding)[0, 0])
    y_min_padded = int((y_min - padding)[0, 0])
    y_max_padded = int((y_max + padding)[0, 0])

    return x_max_padded, x_min_padded, y_min_padded, y_max_padded


def compute_roi_flow_mean(flow_field, top_percent):
    magnitude, _angle = cv2.cartToPolar(
        flow_field[:, :, 0],
        flow_field[:, :, 1],
        angleInDegrees=True,
    )
    magnitude_flat = np.ravel(magnitude)

    dx_flat = np.ravel(flow_field[:, :, 0])
    dy_flat = np.ravel(flow_field[:, :, 1])

    # Indices that sort magnitude from small to large
    sorted_indices = np.argsort(magnitude_flat)

    # Start index of the top `top_percent` subset
    start_idx = int(len(magnitude_flat) * (1 - top_percent))

    sum_dx = 0.0
    sum_dy = 0.0

    for i in range(start_idx, len(sorted_indices)):
        idx = sorted_indices[i]
        sum_dx += dx_flat[idx]
        sum_dy += dy_flat[idx]

    count = len(sorted_indices) - start_idx
    mean_dx = sum_dx / count
    mean_dy = sum_dy / count

    return mean_dx, mean_dy


def update_landmarks_with_fallback(
    gray_image,
    image,
    landmarks_68,
    crop_top,
    crop_left,
    crop_height,
    crop_width,
    target_size,
):
    face_rects = detector(gray_image, 0)

    if len(face_rects) == 0:
        # Map previous landmarks into the resized ROI coordinate system
        landmarks_68[:, 0] = (landmarks_68[:, 0] - crop_left) * (target_size / crop_width)
        landmarks_68[:, 1] = (landmarks_68[:, 1] - crop_top) * (target_size / crop_height)
        current_landmarks = landmarks_68
    else:
        current_landmarks = np.matrix([[p.x, p.y] for p in predictor(image, face_rects[0]).parts()])

    return current_landmarks



def compute_flow_magnitude_series(flow_vectors):
    flow_vectors = np.array(flow_vectors)

    squared_sum = np.sum(flow_vectors ** 2, axis=1)
    magnitudes = np.sqrt(squared_sum)

    return magnitudes


def detect_peaks_by_thresholds(flow_magnitudes, imf_sum_excluding_last, low_threshold, peak_threshold):

    MERGE_GAP = 3
    WINDOW_RADIUS = 30
    IMF_MIN_DELTA = 0.5

    flow_magnitudes = np.array(flow_magnitudes)

    # Shift so the series starts from 0 (keeps original behavior)
    global_min = np.min(flow_magnitudes)
    flow_shifted = flow_magnitudes - global_min

    # ---------- Stage 1: coarse thresholding ----------
    candidate_indices = []
    for idx in range(len(flow_shifted)):
        if flow_shifted[idx] >= low_threshold:
            candidate_indices.append(idx)

    merged_segments_stage1 = []
    if len(candidate_indices) > 0:
        seg_start = candidate_indices[0]
        seg_end = candidate_indices[0]

        for idx in candidate_indices:
            if idx >= seg_end and (idx - seg_end) < MERGE_GAP:
                seg_end = idx
            else:
                merged_segments_stage1.append([seg_start, seg_end])
                seg_start = idx
                seg_end = idx

        merged_segments_stage1.append([seg_start, seg_end])

    segments_stage1 = np.array(merged_segments_stage1)

    # ---------- Stage 2: local-min based refinement ----------
    refined_indices = []
    for seg_idx in range(len(segments_stage1)):
        seg_start = segments_stage1[seg_idx, 0]
        seg_end = segments_stage1[seg_idx, 1]

        for idx in range(seg_start, seg_end):
            left = max(0, idx - WINDOW_RADIUS)
            right = min(len(flow_shifted) - 1, idx + WINDOW_RADIUS)

            local_min_flow = np.min(flow_shifted[left:right])
            local_min_imf = np.min(imf_sum_excluding_last[left:right])

            if (flow_shifted[idx] - local_min_flow > peak_threshold) and (
                imf_sum_excluding_last[idx] - local_min_imf > IMF_MIN_DELTA
            ):
                refined_indices.append(idx)

    # Merge refined indices into final segments
    merged_segments_final = []
    if len(refined_indices) > 0:
        seg_start = refined_indices[0]
        seg_end = refined_indices[0]

        for idx in refined_indices:
            if idx >= seg_end and (idx - seg_end) < MERGE_GAP:
                seg_end = idx
            else:
                merged_segments_final.append([seg_start, seg_end])
                seg_start = idx
                seg_end = idx

        merged_segments_final.append([seg_start, seg_end])

    return np.array(merged_segments_final)


def expand_segments(segments, emd_series):
    WINDOW_RADIUS = 30
    LOCAL_RADIUS = 10
    RATIO_THRESHOLD = 0.33
    DELTA_THRESHOLD = 0.3

    for i in range(len(segments)):
        start = segments[i, 0]
        end = segments[i, 1]

        start_left = max(0, start - WINDOW_RADIUS)
        start_right = min(len(emd_series) - 1, start + WINDOW_RADIUS)
        end_left = max(0, end - WINDOW_RADIUS)
        end_right = min(len(emd_series) - 1, end + WINDOW_RADIUS)

        if end > start:  # end can be equal to start
            segment_peak = np.max(emd_series[start:end])
        else:
            segment_peak = emd_series[start]

        start_window_min = np.min(emd_series[start_left:start_right])
        start_window_min_idx = np.argmin(emd_series[start_left:start_right]) + start_left  # min index around start

        end_window_min = np.min(emd_series[end_left:end_right])
        end_window_min_idx = np.argmin(emd_series[end_left:end_right]) + end_left  # min index around end

        # Expand start
        if start_window_min_idx < start:
            for j in range(start - 1, -1, -1):
                if emd_series[j] - start_window_min < RATIO_THRESHOLD * (segment_peak - start_window_min):
                    start = j
                    break
                if emd_series[j] > emd_series[j + 1]:
                    start = j + 2
                    break
        else:
            left_bound = max(start - LOCAL_RADIUS, 0)
            local_min_idx = np.argmin(emd_series[left_bound:start + 1]) + left_bound  # min index in left LOCAL_RADIUS
            if emd_series[start] - emd_series[local_min_idx] > DELTA_THRESHOLD:
                start = local_min_idx + 1

        # Expand end
        if end_window_min_idx > end:
            for j in range(end + 1, end_window_min_idx):
                if emd_series[j] - end_window_min < RATIO_THRESHOLD * (segment_peak - end_window_min):
                    end = j
                    break
                if emd_series[j] > emd_series[j - 1]:
                    end = j - 2
                    break
        else:
            right_bound = min(end + LOCAL_RADIUS, len(emd_series) - 1)
            local_min_idx = np.argmin(emd_series[end:right_bound + 1]) + end  # min index in right LOCAL_RADIUS
            if emd_series[end] - emd_series[local_min_idx] > DELTA_THRESHOLD:
                end = local_min_idx - 1

        segments[i, 0] = start
        segments[i, 1] = end

    return segments



def compute_main_secondary_flow_mean(flow_field, top_percent):
    # Convert optical flow to polar coordinates (magnitude + angle)
    magnitudes, angles_deg = cv2.cartToPolar(
        flow_field[:, :, 0],
        flow_field[:, :, 1],
        angleInDegrees=True,
    )

    # Angle bins (in degrees)
    angle_bins = [0, 160, 180, 340, 360]
    hist_counts, _ = np.histogram(angles_deg, bins=angle_bins)

    # Indices of the two most frequent bins
    top2_bin_indices = np.argsort(hist_counts)[-2:]

    # Magnitudes in the two dominant angle bins
    bin1_magnitudes = magnitudes[
        (angles_deg >= angle_bins[top2_bin_indices[0]])
        & (angles_deg < angle_bins[top2_bin_indices[0] + 1])
    ]
    bin2_magnitudes = magnitudes[
        (angles_deg >= angle_bins[top2_bin_indices[1]])
        & (angles_deg < angle_bins[top2_bin_indices[1] + 1])
    ]

    bin1_magnitudes = np.ravel(bin1_magnitudes)
    bin2_magnitudes = np.ravel(bin2_magnitudes)

    dx_flat = np.ravel(flow_field[:, :, 0])
    dy_flat = np.ravel(flow_field[:, :, 1])

    # Keep only the strongest subset within each bin (keeps original behavior)
    bin1_sorted_idx = np.argsort(bin1_magnitudes)
    bin2_sorted_idx = np.argsort(bin2_magnitudes)

    bin2_start_idx = int(len(bin2_magnitudes) * 0.8)
    bin1_start_idx = int(len(bin1_magnitudes) * 0.9)

    bin1_sorted_idx = bin1_sorted_idx[bin1_start_idx:]
    bin2_sorted_idx = bin2_sorted_idx[bin2_start_idx:]

    selected_indices = np.concatenate((bin1_sorted_idx, bin2_sorted_idx))

    sum_dx = 0.0
    sum_dy = 0.0
    for idx in selected_indices:
        sum_dx += dx_flat[idx]
        sum_dy += dy_flat[idx]

    mean_dx = sum_dx / len(selected_indices)
    mean_dy = sum_dy / len(selected_indices)

    return mean_dx, mean_dy




def process_flow_segments(flow_vectors, low_threshold, peak_threshold, position_prefix, sequence_id, frame_index, crop_margin, segments_out):
    sampling_rate = 1
    threshold_bias = 0.9

    low_threshold = low_threshold + threshold_bias
    peak_threshold = peak_threshold + threshold_bias

    flow_magnitudes = compute_flow_magnitude_series(flow_vectors)
    flow_magnitudes = np.array(flow_magnitudes)

    position_prefix = position_prefix + str(sequence_id) + "----"

    high_cutoff_hz = 2
    filtered_flow = temporal_ideal_filter(flow_magnitudes[crop_margin:-crop_margin], 1, high_cutoff_hz, 30)

    offset_len = len(filtered_flow) + 2

    _imf_sum_excluding_first, imf_sum_excluding_last = compute_emd_imf_sums(
        flow_magnitudes[crop_margin:-crop_margin],
        filtered_flow,
        position_prefix,
        str(frame_index - offset_len),
        sampling_rate,
    )

    segments = detect_peaks_by_thresholds(filtered_flow, imf_sum_excluding_last, low_threshold, peak_threshold)
    segments = expand_segments(segments, filtered_flow)

    segments = segments + (frame_index - offset_len) + crop_margin
    for i in range(len(segments)):
        segments_out.append(segments[i])

    return segments_out


def merge_overlapping_segments(segments, iou_threshold):
    segments = np.array(segments)

    merged = [[0, 0]]
    for i in range(len(segments)):
        is_new = True

        if i == 0:
            merged = np.vstack((merged, [[segments[i, 0], segments[i, 1]]]))
            continue

        for j in range(1, len(merged)):
            if segments[i, 0] > merged[j, 1] or segments[i, 1] < merged[j, 0]:
                overlap_ratio = 0
            else:
                overlap_start = max(segments[i, 0], merged[j, 0])
                overlap_end = min(segments[i, 1], merged[j, 1])
                overlap_width = overlap_end - overlap_start

                np.seterr(divide="ignore", invalid="ignore")
                overlap_ratio = max(
                    overlap_width / (merged[j, 1] - merged[j, 0]),
                    overlap_width / (segments[i, 1] - segments[i, 0]),
                )

            if overlap_ratio > iou_threshold:
                is_new = False
                merged[j, 1] = max(merged[j, 1], segments[i, 1])
                merged[j, 0] = min(merged[j, 0], segments[i, 0])

        if is_new:
            merged = np.vstack((merged, [[segments[i, 0], segments[i, 1]]]))

    return merged



def extract_roi_flow_segments_v19(base_dir, video_subdir, pre_frames, post_frames, frame_step):
    image_dir = base_dir + video_subdir + "/"
    all_files = os.listdir(image_dir)

    all_files.sort()

    sampled_files = []
    frame_counter = 0
    for filename in all_files:
        if frame_counter % frame_step == 0:
            sampled_files.append(filename)
        frame_counter += 1

    frame_idx = 0  # start position
    window_start = frame_idx - 99
    window_end = frame_idx + 100
    window_move = 100
    is_last_window = True

    all_segments = np.array([[0, 0]])

    while frame_idx < len(sampled_files):
        window_start += window_move
        window_end += window_move

        if window_end > len(sampled_files) and is_last_window:
            window_end = len(sampled_files) - 2
            is_last_window = False

        frame_idx = 0
        is_mid = False

        global window_start_idx
        global window_end_idx
        window_start_idx = window_start
        window_end_idx = window_end

        for filename in sampled_files:
            frame_idx += 1
            if frame_idx >= window_start:
                if frame_idx == window_start:
                    flow_region1 = [[0, 0]]
                    flow_region1_1 = [[0, 0]]
                    flow_region1_2 = [[0, 0]]
                    flow_region1_3 = [[0, 0]]
                    flow_region2 = [[0, 0]]
                    flow_region3 = [[0, 0]]
                    flow_region3_1 = [[0, 0]]
                    flow_region3_2 = [[0, 0]]
                    flow_region3_3 = [[0, 0]]
                    flow_region4 = [[0, 0]]
                    flow_region4_1 = [[0, 0]]
                    flow_region4_2 = [[0, 0]]
                    flow_region4_3 = [[0, 0]]
                    flow_region4_4 = [[0, 0]]
                    flow_region4_5 = [[0, 0]]
                    flow_region5_1 = [[0, 0]]
                    flow_region5_2 = [[0, 0]]
                    flow_region2_1 = [[0, 0]]
                    flow_region6 = [[0, 0]]
                    flow_region7 = [[0, 0]]

                    frame_image = cv2.imread(image_dir + filename)

                    target_size = 256
                    landmarks_68, face_roi, crop_top, crop_bottom, crop_left, crop_right = crop_face_region(
                        frame_image, target_size
                    )
                    # crop_top/crop_bottom/crop_left/crop_right are coordinates in the original image.
                    # face_roi is the cropped face area resized to 256x256; landmarks_68 are landmarks on the original image.

                    gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)

                    landmarks_68 = update_landmarks_with_fallback(
                        gray_roi,
                        face_roi,
                        landmarks_68,
                        crop_top,
                        crop_left,
                        crop_bottom - crop_top,
                        crop_right - crop_left,
                        target_size,
                    )
                    # Landmarks in the resized ROI coordinate system.

                    padding = 0

                    # Left eyebrow ROI
                    roi_x_max, roi_x_min, roi_y_min, roi_y_max = get_roi_bounds(17, 22, 0, landmarks_68)

                    left_brow_points = []  # 3 small ROIs (inner -> outer)
                    left_brow_points.append([landmarks_68[20, 1] - (roi_y_min - 15), landmarks_68[20, 0] - (roi_x_min - 5)])
                    left_brow_points.append([landmarks_68[19, 1] - (roi_y_min - 15), landmarks_68[19, 0] - (roi_x_min - 5)])
                    left_brow_points.append([landmarks_68[18, 1] - (roi_y_min - 15), landmarks_68[18, 0] - (roi_x_min - 5)])

                    prev_gray_left_brow = gray_roi[(roi_y_min - 15):roi_y_max + 5, roi_x_min - 5:roi_x_max]

                    # Right eyebrow ROI
                    roi3_x_max, roi3_x_min, roi3_y_min, roi3_y_max = get_roi_bounds(22, 27, 0, landmarks_68)

                    right_brow_points = []  # 3 small ROIs (inner -> outer)
                    right_brow_points.append([landmarks_68[23, 1] - (roi3_y_min - 15), landmarks_68[23, 0] - roi3_x_min])
                    right_brow_points.append([landmarks_68[24, 1] - (roi3_y_min - 15), landmarks_68[24, 0] - roi3_x_min])
                    right_brow_points.append([landmarks_68[25, 1] - (roi3_y_min - 15), landmarks_68[25, 0] - roi3_x_min])

                    prev_gray_right_brow = gray_roi[(roi3_y_min - 15):roi3_y_max + 5, roi3_x_min:roi3_x_max]

                    # Mouth ROI (5 anchor points)
                    mouth_x_max, mouth_x_min, mouth_y_min, mouth_y_max = get_roi_bounds(48, 67, 0, landmarks_68)

                    mouth_points = []
                    mouth_points.append([landmarks_68[48, 1] - (mouth_y_min - 15), landmarks_68[48, 0] - (mouth_x_min - 20)])
                    mouth_points.append([landmarks_68[54, 1] - (mouth_y_min - 15), landmarks_68[54, 0] - (mouth_x_min - 20)])
                    mouth_points.append([landmarks_68[51, 1] - (mouth_y_min - 15), landmarks_68[51, 0] - (mouth_x_min - 20)])
                    mouth_points.append([landmarks_68[57, 1] - (mouth_y_min - 15), landmarks_68[57, 0] - (mouth_x_min - 20)])
                    mouth_points.append([landmarks_68[62, 1] - (mouth_y_min - 15), landmarks_68[62, 0] - (mouth_x_min - 20)])

                    prev_gray_mouth = gray_roi[(mouth_y_min - 15):mouth_y_max + 10, mouth_x_min - 20:mouth_x_max + 20]

                    # Nose sides ROI
                    nose_side_x_max, nose_side_x_min, nose_side_y_min, nose_side_y_max = get_roi_bounds(30, 36, 0, landmarks_68)

                    nose_side_points = []
                    nose_side_points.append([landmarks_68[31, 1] - (nose_side_y_min - 20), landmarks_68[31, 0] - (nose_side_x_min - 30)])
                    nose_side_points.append([landmarks_68[35, 1] - (nose_side_y_min - 20), landmarks_68[35, 0] - (nose_side_x_min - 30)])

                    prev_gray_nose_sides = gray_roi[
                        (nose_side_y_min - 20):nose_side_y_max + 5,
                        nose_side_x_min - 30:nose_side_x_max + 30,
                    ]

                    # Nose center ROI
                    nose_x_max, nose_x_min, nose_y_min, nose_y_max = get_roi_bounds(29, 31, 13, landmarks_68)
                    prev_gray_nose_center = gray_roi[nose_y_min:nose_y_max, nose_x_min:nose_x_max]

                    # Left outer eye corner region (based on landmark 0 in original code)
                    left_corner_x_min = max(0, landmarks_68[0, 0] - 1)
                    left_corner_x_max = min(target_size, landmarks_68[0, 0] + 25)
                    left_corner_y_min = max(0, landmarks_68[0, 1] - 10)
                    left_corner_y_max = min(target_size, landmarks_68[0, 1] + 10)
                    prev_gray_left_corner = gray_roi[left_corner_y_min:left_corner_y_max, left_corner_x_min:left_corner_x_max]

                    # Right outer eye corner region (based on landmark 16 in original code)
                    right_corner_x_min = max(0, landmarks_68[16, 0] - 25)
                    right_corner_x_max = min(target_size, landmarks_68[16, 0] + 1)
                    right_corner_y_min = max(0, landmarks_68[16, 1] - 10)
                    right_corner_y_max = min(target_size, landmarks_68[16, 1] + 10)
                    prev_gray_right_corner = gray_roi[right_corner_y_min:right_corner_y_max, right_corner_x_min:right_corner_x_max]

                else:
                    current_frame = cv2.imread(image_dir + filename)

                    face_crop = current_frame[crop_top:crop_bottom, crop_left:crop_right]
                    face_roi = cv2.resize(face_crop, (target_size, target_size))
                    gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)

                    # Global motion ROIs
                    nose_center_roi = gray_roi[nose_y_min:nose_y_max, nose_x_min:nose_x_max]
                    left_corner_roi = gray_roi[left_corner_y_min:left_corner_y_max, left_corner_x_min:left_corner_x_max]
                    right_corner_roi = gray_roi[right_corner_y_min:right_corner_y_max,
                                       right_corner_x_min:right_corner_x_max]

                    nose_flow = cv2.calcOpticalFlowFarneback(prev_gray_nose_center, nose_center_roi, None, 0.5, 3, 15,
                                                             5, 7, 1.5, 0)
                    left_corner_flow = cv2.calcOpticalFlowFarneback(prev_gray_left_corner, left_corner_roi, None, 0.5,
                                                                    3, 15, 5, 7, 1.5, 0)
                    right_corner_flow = cv2.calcOpticalFlowFarneback(prev_gray_right_corner, right_corner_roi, None,
                                                                     0.5, 3, 15, 5, 7, 1.5, 0)

                    nose_dx, nose_dy = compute_roi_flow_mean(nose_flow[15:-10, 5:-5, :], 0.7)
                    left_dx, left_dy = compute_roi_flow_mean(left_corner_flow, 0.8)
                    right_dx, right_dy = compute_roi_flow_mean(right_corner_flow, 0.8)

                    global_dx = 0.8 * nose_dx + 0.1 * left_dx + 0.1 * right_dx
                    global_dy = 0.8 * nose_dy + 0.1 * left_dy + 0.1 * right_dy

                    # Face alignment: shift the crop box
                    align_iter = 0
                    while (global_dx ** 2 + global_dy ** 2) > 0.5:
                        align_iter += 1
                        if align_iter > 3:
                            break

                        crop_left += int(round(global_dx))
                        crop_top += int(round(global_dy))
                        crop_right += int(round(global_dx))
                        crop_bottom += int(round(global_dy))

                        crop_left = max(0, crop_left)
                        crop_top = max(0, crop_top)

                        if crop_bottom == 0:
                            print(image_dir + filename)
                            continue

                        current_frame = cv2.imread(image_dir + filename)
                        face_crop = current_frame[crop_top:crop_bottom, crop_left:crop_right]
                        face_roi = cv2.resize(face_crop, (target_size, target_size))
                        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)

                        nose_center_roi = gray_roi[nose_y_min:nose_y_max, nose_x_min:nose_x_max]
                        left_corner_roi = gray_roi[left_corner_y_min:left_corner_y_max,
                                          left_corner_x_min:left_corner_x_max]
                        right_corner_roi = gray_roi[right_corner_y_min:right_corner_y_max,
                                           right_corner_x_min:right_corner_x_max]

                        nose_flow = cv2.calcOpticalFlowFarneback(prev_gray_nose_center, nose_center_roi, None, 0.5, 3,
                                                                 15, 5, 7, 1.5, 0)
                        left_corner_flow = cv2.calcOpticalFlowFarneback(prev_gray_left_corner, left_corner_roi, None,
                                                                        0.5, 3, 15, 5, 7, 1.5, 0)
                        right_corner_flow = cv2.calcOpticalFlowFarneback(prev_gray_right_corner, right_corner_roi, None,
                                                                         0.5, 3, 15, 5, 7, 1.5, 0)

                        nose_dx, nose_dy = compute_roi_flow_mean(nose_flow[15:-10, 5:-5, :], 0.7)
                        left_dx, left_dy = compute_roi_flow_mean(left_corner_flow, 0.8)
                        right_dx, right_dy = compute_roi_flow_mean(right_corner_flow, 0.8)

                        global_dx = 0.8 * nose_dx + 0.1 * left_dx + 0.1 * right_dx
                        global_dy = 0.8 * nose_dy + 0.1 * left_dy + 0.1 * right_dy

                    # Left eyebrow ROI flow,,roi_x_max, roi_x_min, roi_y_min, roi_y_max
                    left_brow_roi = gray_roi[(roi_y_min - 15):roi_y_max + 5,
                                    roi_x_min - 5:roi_x_max]
                    try:
                        left_brow_flow = cv2.calcOpticalFlowFarneback(prev_gray_left_brow, left_brow_roi, None, 0.5, 3,
                                                                      15, 5, 7, 1.5, 0)
                    except:
                        break

                    left_brow_flow = left_brow_flow - np.array([global_dx, global_dy])

                    roi_padding = 10
                    left_brow_points = np.array(left_brow_points)

                    a, b = compute_main_secondary_flow_mean(left_brow_flow[roi_padding:-roi_padding, roi_padding:-roi_padding, :],
                                                 0.2)
                    a1, b1 = compute_main_secondary_flow_mean(
                        left_brow_flow[left_brow_points[0, 0] - 10:left_brow_points[0, 0] + 10,
                        left_brow_points[0, 1] - 10:left_brow_points[0, 1] + 10, :],
                        0.2,
                    )
                    a2, b2 = compute_main_secondary_flow_mean(
                        left_brow_flow[left_brow_points[1, 0] - 10:left_brow_points[1, 0] + 10,
                        left_brow_points[1, 1] - 10:left_brow_points[1, 1] + 10, :],
                        0.2,
                    )
                    a3, b3 = compute_main_secondary_flow_mean(
                        left_brow_flow[left_brow_points[2, 0] - 10:left_brow_points[2, 0] + 10,
                        left_brow_points[2, 1] - 10:left_brow_points[2, 1] + 10, :],
                        0.2,
                    )

                    flow_region1_1.append([a1 - global_dx, b1 - global_dy])
                    flow_region1_2.append([a2 - global_dx, b2 - global_dy])
                    flow_region1_3.append([a3 - global_dx, b3 - global_dy])
                    flow_region1.append([a - global_dx, b - global_dy])

                    # Right eyebrow ROI flow,,roi3_x_max, roi3_x_min, roi3_y_min, roi3_y_max
                    right_brow_roi = gray_roi[(roi3_y_min - 15):roi3_y_max + 5,
                                     roi3_x_min:roi3_x_max]
                    right_brow_flow = cv2.calcOpticalFlowFarneback(prev_gray_right_brow, right_brow_roi, None, 0.5, 3,
                                                                   15, 5, 7, 1.5, 0)

                    roi_padding = 10
                    right_brow_points = np.array(right_brow_points)

                    a, b = compute_main_secondary_flow_mean(right_brow_flow[roi_padding:-roi_padding, roi_padding:-roi_padding, :],
                                                 0.3)
                    a1, b1 = compute_main_secondary_flow_mean(
                        right_brow_flow[right_brow_points[0, 0] - 10:right_brow_points[0, 0] + 10,
                        right_brow_points[0, 1] - 10:right_brow_points[0, 1] + 10, :],
                        0.3,
                    )
                    a2, b2 = compute_main_secondary_flow_mean(
                        right_brow_flow[right_brow_points[1, 0] - 10:right_brow_points[1, 0] + 10,
                        right_brow_points[1, 1] - 10:right_brow_points[1, 1] + 10, :],
                        0.3,
                    )
                    a3, b3 = compute_main_secondary_flow_mean(
                        right_brow_flow[right_brow_points[2, 0] - 10:right_brow_points[2, 0] + 10,
                        right_brow_points[2, 1] - 10:right_brow_points[2, 1] + 10, :],
                        0.3,
                    )

                    flow_region3_1.append([a1 - global_dx, b1 - global_dy])
                    flow_region3_2.append([a2 - global_dx, b2 - global_dy])
                    flow_region3_3.append([a3 - global_dx, b3 - global_dy])
                    flow_region3.append([a - global_dx, b - global_dy])

                    # Mouth ROI flow
                    mouth_roi = gray_roi[(mouth_y_min - 15):mouth_y_max + 10, mouth_x_min - 20:mouth_x_max + 20]
                    mouth_flow = cv2.calcOpticalFlowFarneback(prev_gray_mouth, mouth_roi, None, 0.5, 3, 15, 5, 7, 1.5,
                                                              0)

                    roi_padding = 10
                    mouth_points = np.array(mouth_points)

                    a, b = compute_main_secondary_flow_mean(mouth_flow[roi_padding:-roi_padding, roi_padding:-roi_padding, :], 0.3)
                    a1, b1 = compute_main_secondary_flow_mean(
                        mouth_flow[mouth_points[0, 0] - 10:mouth_points[0, 0] + 10,
                        mouth_points[0, 1] - 10:mouth_points[0, 1] + 20, :],
                        0.2,
                    )
                    a2, b2 = compute_main_secondary_flow_mean(
                        mouth_flow[mouth_points[1, 0] - 10:mouth_points[1, 0] + 10,
                        mouth_points[1, 1] - 20:mouth_points[1, 1] + 10, :],
                        0.2,
                    )
                    a3, b3 = compute_main_secondary_flow_mean(
                        mouth_flow[mouth_points[2, 0] - 10:mouth_points[2, 0] + 10,
                        mouth_points[2, 1] - 10:mouth_points[2, 1] + 10, :],
                        0.2,
                    )
                    a4, b4 = compute_main_secondary_flow_mean(
                        mouth_flow[mouth_points[3, 0] - 10:mouth_points[3, 0] + 10,
                        mouth_points[3, 1] - 10:mouth_points[3, 1] + 10, :],
                        0.2,
                    )
                    a5, b5 = compute_main_secondary_flow_mean(
                        mouth_flow[mouth_points[4, 0] - 10:mouth_points[4, 0] + 10,
                        mouth_points[4, 1] - 10:mouth_points[4, 1] + 10, :],
                        0.2,
                    )

                    flow_region4_1.append([a1 - global_dx, b1 - global_dy])
                    flow_region4_2.append([a2 - global_dx, b2 - global_dy])
                    flow_region4_3.append([a3 - global_dx, b3 - global_dy])
                    flow_region4_4.append([a4 - global_dx, b4 - global_dy])
                    flow_region4_5.append([a5 - global_dx, b5 - global_dy])
                    flow_region4.append([a - global_dx, b - global_dy])

                    # Nose-sides ROI flow
                    nose_sides_roi = gray_roi[(nose_side_y_min - 20):nose_side_y_max + 5,
                                     nose_side_x_min - 30:nose_side_x_max + 30]
                    nose_sides_flow = cv2.calcOpticalFlowFarneback(prev_gray_nose_sides, nose_sides_roi, None, 0.5, 3,
                                                                   15, 5, 7, 1.5, 0)

                    nose_side_points = np.array(nose_side_points)

                    a1, b1 = compute_main_secondary_flow_mean(
                        nose_sides_flow[nose_side_points[0, 0] - 20:nose_side_points[0, 0] + 5,
                        nose_side_points[0, 1] - 20:nose_side_points[0, 1] + 10, :],
                        0.2,
                    )
                    a2, b2 = compute_main_secondary_flow_mean(
                        nose_sides_flow[nose_side_points[1, 0] - 20:nose_side_points[1, 0] + 5,
                        nose_side_points[1, 1] - 10:nose_side_points[1, 1] + 20, :],
                        0.2,
                    )

                    flow_region5_1.append([a1 - global_dx, b1 - global_dy])
                    flow_region5_2.append([a2 - global_dx, b2 - global_dy])

                    roi_padding = 5

            if frame_idx == window_end:
                window_len = window_end - window_start + 1

                segments_out = []
                micro_segments = []
                macro_segments = []

                crop_margin = 1

                segments_out = process_flow_segments(flow_region1_1, 1.5, 1.9, "left_eye", 1, frame_idx, crop_margin,
                                                     segments_out)
                segments_out = process_flow_segments(flow_region1_2, 1.5, 1.9, "left_eye", 2, frame_idx, crop_margin,
                                                     segments_out)
                segments_out = process_flow_segments(flow_region1_3, 1.5, 1.9, "left_eye", 3, frame_idx, crop_margin,
                                                     segments_out)

                segments_out = process_flow_segments(flow_region3_1, 1.5, 1.9, "right_eye", 1, frame_idx, crop_margin,
                                                     segments_out)
                segments_out = process_flow_segments(flow_region3_2, 1.5, 1.9, "right_eye", 2, frame_idx, crop_margin,
                                                     segments_out)
                segments_out = process_flow_segments(flow_region3_3, 1.5, 1.9, "right_eye", 3, frame_idx, crop_margin,
                                                     segments_out)

                segments_out = process_flow_segments(flow_region4_1, 1.4, 1.85, "mouth", 1, frame_idx, crop_margin,
                                                     segments_out)
                segments_out = process_flow_segments(flow_region4_2, 1.4, 1.85, "mouth", 2, frame_idx, crop_margin,
                                                     segments_out)
                segments_out = process_flow_segments(flow_region4_3, 1.4, 1.85, "mouth", 3, frame_idx, crop_margin,
                                                     segments_out)
                segments_out = process_flow_segments(flow_region4_4, 1.4, 1.85, "mouth", 4, frame_idx, crop_margin,
                                                     segments_out)
                segments_out = process_flow_segments(flow_region4_5, 1.4, 1.85, "mouth", 5, frame_idx, crop_margin,
                                                     segments_out)

                segments_out = process_flow_segments(flow_region5_1, 1.4, 2.1, "nose", 1, frame_idx, crop_margin, segments_out)
                segments_out = process_flow_segments(flow_region5_2, 1.4, 2.1, "nose", 2, frame_idx, crop_margin, segments_out)

                segments_out = np.array(merge_overlapping_segments(segments_out, 0.1))
                segments_out = np.array(merge_overlapping_segments(segments_out, 0.1))

                segments_relative = segments_out - (frame_idx - window_len)

                window_move = 100
                for seg_idx in range(len(segments_relative)):
                    if segments_relative[seg_idx, 0] < 100 and segments_relative[seg_idx, 1] > 100:
                        if segments_relative[seg_idx, 1] < 150:
                            window_move = segments_relative[seg_idx, 1] + 20
                        elif segments_relative[seg_idx, 0] > 50:
                            window_move = segments_relative[seg_idx, 0] - 20
                        else:
                            tmp_end = min(189, segments_relative[seg_idx, 1])
                            window_move = tmp_end + 10

                all_segments = np.vstack((all_segments, segments_out))
                break

    print("All segments:")

    filtered_segments = []
    filtered_segments_merged = []

    for seg_idx in range(len(all_segments)):
        if 12 <= (all_segments[seg_idx, 1] - all_segments[seg_idx, 0]) <= 200:
            filtered_segments.append([all_segments[seg_idx, 0], all_segments[seg_idx, 1]])

    filtered_segments.sort()
    filtered_segments = np.array(merge_overlapping_segments(filtered_segments, 0.1))
    filtered_segments = np.array(merge_overlapping_segments(filtered_segments, 0.1))

    prev_start = 0
    prev_end = 0
    for seg_idx in range(len(filtered_segments)):
        if filtered_segments[seg_idx, 1] != 0:
            if prev_start != 0:
                if (filtered_segments[seg_idx, 0] - prev_end) < 15 and (
                        (filtered_segments[seg_idx, 1] - filtered_segments[seg_idx, 0]) > 15
                ):
                    c = 1
                    continue
            filtered_segments_merged.append([filtered_segments[seg_idx, 0], filtered_segments[seg_idx, 1]])
            prev_start = filtered_segments[seg_idx, 0]
            prev_end = filtered_segments[seg_idx, 1]

    filtered_segments_merged = np.array(filtered_segments_merged)
    return filtered_segments_merged

