import os
import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.utils import register_keras_serializable
from collections import deque
import tifffile
import argparse
from pathlib import Path

@register_keras_serializable()
class CentralFrameExtractionLayer(layers.Layer):
    def __init__(self, sequence_length, **kwargs):
        super(CentralFrameExtractionLayer, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.center = sequence_length // 2

    def call(self, inputs):
        # Input shape: (batch, H, W, sequence_length, channels)
        # Output shape: (batch, H, W, channels)
        return inputs[:, :, :, self.center, :]

    def get_config(self):
        config = super(CentralFrameExtractionLayer, self).get_config()
        config.update({'sequence_length': self.sequence_length})
        return config

def build_3d_unet(input_shape, sequence_length):
    inputs = layers.Input(input_shape)
    # Encoding path
    c1 = layers.Conv3D(32, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv3D(32, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPool3D(pool_size=(2, 2, 1))(c1)
    c2 = layers.Conv3D(64, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv3D(64, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPool3D(pool_size=(2, 2, 1))(c2)
    c3 = layers.Conv3D(128, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv3D(128, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPool3D(pool_size=(2, 2, 1))(c3)
    # Bridge
    c4 = layers.Conv3D(256, 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv3D(256, 3, activation='relu', padding='same')(c4)
    # Decoding path
    u1 = layers.Conv3DTranspose(128, 2, strides=(2, 2, 1), padding='same')(c4)
    u1 = layers.concatenate([u1, c3])
    c5 = layers.Conv3D(128, 3, activation='relu', padding='same')(u1)
    c5 = layers.Conv3D(128, 3, activation='relu', padding='same')(c5)
    u2 = layers.Conv3DTranspose(64, 2, strides=(2, 2, 1), padding='same')(c5)
    u2 = layers.concatenate([u2, c2])
    c6 = layers.Conv3D(64, 3, activation='relu', padding='same')(u2)
    c6 = layers.Conv3D(64, 3, activation='relu', padding='same')(c6)
    u3 = layers.Conv3DTranspose(32, 2, strides=(2, 2, 1), padding='same')(c6)
    u3 = layers.concatenate([u3, c1])
    c7 = layers.Conv3D(32, 3, activation='relu', padding='same')(u3)
    c7 = layers.Conv3D(32, 3, activation='relu', padding='same')(c7)
    outputs_seq = layers.Conv3D(1, 1, activation='relu', padding='same')(c7)
    outputs_central_frame = CentralFrameExtractionLayer(sequence_length)(outputs_seq)
    return Model(inputs=inputs, outputs=outputs_central_frame)


def analyze_noise_regions(frames, noise_regions):
    noise_pixels = []
    for region in noise_regions:
        x1, y1, x2, y2 = region
        for frame in frames:
            noise_pixels.extend(frame[y1:y2, x1:x2].flatten())
    if not noise_pixels:
        raise ValueError("No pixels found in noise regions. Check region definitions in config.")
    noise_pixels = np.array(noise_pixels)
    background_level = np.median(noise_pixels)
    return background_level
    
def pad_to_divisible(image_stack, divisor=8):
    num_frames, h, w = image_stack.shape

    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded_stack = np.pad(
        image_stack,
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode='reflect'
    )
    print(f"Original HxW: {h}x{w}. Padded to divisible-by-{divisor} size: {padded_stack.shape[1]}x{padded_stack.shape[2]}")
    return padded_stack, (pad_top, pad_bottom, pad_left, pad_right)

def pad_video_sequence(frames, sequence_length):
    pad_size = sequence_length // 2
    if pad_size == 0:
        return frames
    padded_frames = np.pad(frames, ((pad_size, pad_size), (0, 0), (0, 0)), mode='reflect')
    print(f"Applied reflection padding of size {pad_size} to video sequence (temporal).")
    return padded_frames
    
def evaluate_video(input_tiff_path, training_run_folder, output_path):
    print("--- Step 1: Loading Configuration ---")
    config_path = os.path.join(training_run_folder, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    print("Configuration loaded successfully.")

    print("\n--- Step 2: Loading U-Net Model ---")
    sequence_length = config['sequence_length']
    input_shape = (None, None, sequence_length, config['channels'])
    unet_model = build_3d_unet(input_shape, sequence_length)

    best_weights_path = os.path.join(training_run_folder, "models", "unet_best.weights.h5")
    if not os.path.exists(best_weights_path):
         raise FileNotFoundError(f"The required weights file was not found: {best_weights_path}")

    unet_model.load_weights(best_weights_path)
    print(f"Model built with flexible dimensions and loaded weights from: {best_weights_path}")

    print("\n--- Step 3: Loading and Preprocessing Video ---")
    try:
        video_frames = tifffile.imread(input_tiff_path)
    except Exception as e:
        print(f"Tifffile failed ({e}), falling back to OpenCV.")
        ret, video_frames = cv2.imreadmulti(input_tiff_path, flags=cv2.IMREAD_UNCHANGED)
        if not ret:
            raise IOError(f"Could not read the input TIFF file: {input_tiff_path}")

    video_frames = np.array(video_frames, dtype=np.float32)
    original_shape = video_frames.shape
    num_frames_orig = original_shape[0]
    print(f"Video loaded: {num_frames_orig} frames, {original_shape[1]}x{original_shape[2]} resolution.")

    background_level = analyze_noise_regions(video_frames, config['noise_analysis_regions'])
    print(f"Estimated background level: {background_level:.2f}")
    video_frames -= background_level
    print("Background subtracted from all frames.")

    spatially_padded_frames, spatial_pads = pad_to_divisible(video_frames, 8)
    temporally_padded_frames = pad_video_sequence(spatially_padded_frames, sequence_length)

    print("\n--- Step 4: Denoising Video ---")
    denoised_frames = []

    for i in range(num_frames_orig):
        frame_sequence = temporally_padded_frames[i : i + sequence_length]
        model_input = np.transpose(frame_sequence, (1, 2, 0))
        model_input = np.expand_dims(model_input, axis=(0, -1))

        denoised_center_frame = unet_model.predict(model_input, verbose=0)
        denoised_frames.append(np.squeeze(denoised_center_frame))

        print(f"\rProcessing frame {i + 1}/{num_frames_orig}...", end="")

    print("\nVideo denoising complete.")
    print("\n--- Step 5: Post-processing and Saving Denoised Video ---")
    denoised_video = np.array(denoised_frames)

    pad_top, pad_bottom, pad_left, pad_right = spatial_pads
    h_new, w_new = denoised_video.shape[1], denoised_video.shape[2]
    
    crop_y_end = h_new - pad_bottom if pad_bottom > 0 else h_new
    crop_x_end = w_new - pad_right if pad_right > 0 else w_new

    denoised_video_cropped = denoised_video[:, pad_top:crop_y_end, pad_left:crop_x_end]
    print(f"Cropped denoised video back to original size: {denoised_video_cropped.shape}")

    denoised_video_cropped += background_level
    denoised_video_final = np.clip(denoised_video_cropped, 0, 65535).astype(np.uint16)

    num_frames, height, width = denoised_video_final.shape
    imagej_hyperstack = denoised_video_final.reshape(num_frames, 1, 1, height, width)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    tifffile.imwrite(
        output_path,
        imagej_hyperstack,
        imagej=True,
        metadata={'axes': 'TZCYX'} 
    )

    print(f"\n✅ Denoised video successfully saved to: {output_path}")

def main(args):
    input_path = Path(args.input_file)
    model_folder_path = Path(args.model_folder)
    config_path = model_folder_path / 'config.json'

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return
    if not model_folder_path.exists():
        print(f"Error: Model folder not found: {model_folder_path}")
        return
    if not config_path.exists():
        print(f"Error: config.json not found in model folder: {config_path}")
        print("Please ensure --model_folder points to the root of a training run.")
        return

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    try:
        evaluate_video(str(input_path), str(model_folder_path), args.output_file)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Denoise a video (multi-page TIFF) using a trained RL-U-Net model."
    )

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input (multi-page) TIFF file to denoise."
    )
    parser.add_argument(
        "--model_folder",
        type=str,
        required=True,
        help="Path to the training run folder (this folder should contain config.json and the models/ subdirectory)."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the denoised output (multi-page) TIFF file."
    )

    args = parser.parse_args()
    main(args)
