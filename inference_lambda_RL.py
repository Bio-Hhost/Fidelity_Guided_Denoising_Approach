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

def evaluate_video(input_tiff_path, training_run_folder, output_path):
    print("--- Step 1: Loading Configuration ---")
    config_path = os.path.join(training_run_folder, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)
    print("Configuration loaded successfully.")

    print("\n--- Step 2: Loading U-Net Model ---")

    final_model_path = os.path.join(training_run_folder, "models", "unet_final.keras")
    best_weights_path = os.path.join(training_run_folder, "models", "unet_best.weights.h5")

    if os.path.exists(final_model_path):
        print(f"Loading full model from: {final_model_path}")
        unet_model = tf.keras.models.load_model(final_model_path, custom_objects={
            'CentralFrameExtractionLayer': CentralFrameExtractionLayer
        })
    elif os.path.exists(best_weights_path):
        print("Final model not found. Building model from config and loading best weights.")

        input_shape = (config['img_height'], config['img_width'], config['sequence_length'], config['channels'])
        unet_model = build_3d_unet(input_shape, config['sequence_length'])
        unet_model.load_weights(best_weights_path)
        print(f"Loaded best weights from: {best_weights_path}")
    else:
        raise FileNotFoundError(f"No trained model or weights file found in {os.path.join(training_run_folder, 'models')}")

    print("Model loaded successfully.")
    unet_model.summary()

    print("\n--- Step 3: Loading and Preprocessing Video ---")
    try:
        video_frames = tifffile.imread(input_tiff_path)
    except Exception as e:
        print(f"Tifffile failed ({e}), falling back to OpenCV.")
        ret, video_frames = cv2.imreadmulti(input_tiff_path, flags=cv2.IMREAD_UNCHANGED)
        if not ret:
            raise IOError(f"Could not read the input TIFF file: {input_tiff_path}")

    video_frames = np.array(video_frames, dtype=np.float32)
    num_frames, height, width = video_frames.shape
    print(f"Video loaded: {num_frames} frames, {height}x{width} resolution.")

    background_level = analyze_noise_regions(video_frames, config['noise_analysis_regions'])
    print(f"Estimated background level: {background_level:.2f}")
    video_frames -= background_level
    print("Background subtracted from all frames.")

    print("\n--- Step 4: Denoising Video ---")
    sequence_length = config['sequence_length']
    center_offset = sequence_length // 2

    denoised_frames = []
    frame_sequence = deque(maxlen=sequence_length)

    for i in range(num_frames):
        pad_idx = np.clip(i, 0, num_frames - 1)
        
        if i == 0:
            for _ in range(sequence_length):
                frame_sequence.append(video_frames[0])
        else:
            frame_to_add_idx = min(i + center_offset, num_frames - 1)
            frame_sequence.append(video_frames[frame_to_add_idx])
            
        # At i=0, sequence is [F0, F0, F0, F0, F0] (for seq_len=5)
        # At i=1, sequence is [F0, F0, F0, F0, F3]
        # At i=2, sequence is [F0, F0, F0, F3, F4]
        # At i=3, sequence is [F0, F0, F3, F4, F5]
        # ...
        
        current_sequence = []
        for j in range(-center_offset, center_offset + 1):
            frame_index = np.clip(i + j, 0, num_frames - 1)
            current_sequence.append(video_frames[frame_index])
            
        model_input = np.stack(current_sequence, axis=-1)
        model_input = np.expand_dims(model_input, axis=(0, -1))
        
        if i == 0:
            for _ in range(sequence_length):
                frame_sequence.append(video_frames[0])
        else:
            frame_sequence.append(video_frames[min(i + center_offset, num_frames - 1)])
        
        model_input = np.array(frame_sequence) # Shape (T, H, W)
        model_input = np.transpose(model_input, (1, 2, 0)) # Shape (H, W, T)
        model_input = np.expand_dims(model_input, axis=(0, -1)) # Shape (1, H, W, T, 1)

        denoised_center_frame = unet_model.predict(model_input, verbose=0)
        denoised_center_frame = np.squeeze(denoised_center_frame)
        denoised_frames.append(denoised_center_frame)

        print(f"\rProcessing frame {i + 1}/{num_frames}...", end="")

    print("\nVideo denoising complete.")

    print("\n--- Step 5: Saving Denoised Video ---")
    denoised_video = np.array(denoised_frames)
    denoised_video += background_level
    denoised_video = np.clip(denoised_video, 0, 65535).astype(np.uint16)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tifffile.imwrite(output_path, denoised_video, imagej=True)
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
    except (FileNotFoundError, IOError, ValueError, Exception) as e:
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
