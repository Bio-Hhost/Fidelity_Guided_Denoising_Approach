import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable
import os
import tifffile
import argparse
from pathlib import Path

@register_keras_serializable()
class CentralFrameExtractionLayer(Layer):
    def __init__(self, sequence_length, **kwargs):
        super(CentralFrameExtractionLayer, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.center = sequence_length // 2

    def call(self, inputs):
        return inputs[:, :, :, self.center, :]

    def get_config(self):
        config = super(CentralFrameExtractionLayer, self).get_config()
        config.update({'sequence_length': self.sequence_length})
        return config

@register_keras_serializable()
class PoissonGaussianNLLLoss(tf.keras.losses.Loss):
    def __init__(self, gain, sigma_sq, epsilon=1e-7, **kwargs):
        super(PoissonGaussianNLLLoss, self).__init__(**kwargs)
        self.gain = gain
        self.sigma_sq = sigma_sq
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        y_true_values = y_true[..., 0]
        mask = y_true[..., 1]
        y_pred = tf.nn.relu(y_pred) + self.epsilon
        y_true_values = tf.expand_dims(y_true_values, axis=-1)

        term1 = (y_true_values - self.gain * y_pred) ** 2 / (2 * self.sigma_sq)
        term2 = self.gain * y_pred
        term3 = -y_true_values * tf.math.log(self.gain * y_pred + self.epsilon)
        loss = term1 + term2 + term3

        masked_loss = loss * tf.expand_dims(mask, axis=-1)
        return tf.reduce_sum(masked_loss) / (tf.reduce_sum(mask) + 1e-7)

    def get_config(self):
        config = super(PoissonGaussianNLLLoss, self).get_config()
        config.update({
            'gain': self.gain,
            'sigma_sq': self.sigma_sq,
            'epsilon': self.epsilon,
        })
        return config

@register_keras_serializable()
class PoissonGaussianNLLLossWithGeometry(tf.keras.losses.Loss):
    def __init__(self, gain, sigma_sq, lambda_geo=0.1, epsilon=1e-7, **kwargs):
        super().__init__(**kwargs)
        self.gain = gain
        self.sigma_sq = sigma_sq
        self.epsilon = epsilon
        self.lambda_geo = lambda_geo

    def call(self, y_true, y_pred):
        mask = y_true[..., 1]
        y_true_values = y_true[..., 0]
        y_pred = tf.nn.relu(y_pred) + self.epsilon

        y_true_values = tf.expand_dims(y_true_values, axis=-1)

        term1 = (y_true_values - self.gain * y_pred) ** 2 / (2.0 * self.sigma_sq)
        term2 = self.gain * y_pred
        term3 = -y_true_values * tf.math.log(self.gain * y_pred + self.epsilon)

        nll_loss = term1 + term2 + term3
        masked_loss = nll_loss * tf.expand_dims(mask, axis=-1)
        masked_loss_mean = tf.reduce_sum(masked_loss) / (tf.reduce_sum(mask) + 1e-7)

        unmask = 1.0 - mask
        fidelity = tf.square(y_pred - y_true_values) * tf.expand_dims(unmask, axis=-1)
        fidelity_mean = tf.reduce_sum(fidelity) / (tf.reduce_sum(unmask) + 1e-7)

        return masked_loss_mean + self.lambda_geo * fidelity_mean

    def get_config(self):
        config = super().get_config()
        config.update({
            'gain': self.gain,
            'sigma_sq': self.sigma_sq,
            'epsilon': self.epsilon,
            'lambda_geo': self.lambda_geo
        })
        return config

def load_frames(filepath):
    ret, frames = cv2.imreadmulti(str(filepath), flags=cv2.IMREAD_UNCHANGED)
    if not ret:
        print(f"Error loading TIFF file: {filepath}")
        return None
    return np.array(frames)

def prepare_sequence(frames, idx, sequence_length):
    half_len = sequence_length // 2
    start_idx = max(0, idx - half_len)
    end_idx = min(len(frames), idx + half_len + 1)

    sequence = [frames[i] for i in range(start_idx, end_idx)]

    while len(sequence) < sequence_length:
        if start_idx > 0:
            sequence.insert(0, frames[0])
            start_idx -= 1
        elif end_idx < len(frames):
            sequence.append(frames[-1])
        else:
             if len(sequence) < sequence_length:
                 sequence.insert(0, frames[0])
             if len(sequence) < sequence_length:
                 sequence.append(frames[-1])

    if len(sequence) > sequence_length:
        center = len(sequence) // 2
        start = center - half_len
        sequence = sequence[start : start + sequence_length]

    return np.stack(sequence, axis=-1) # shape: (H, W, sequence_length)

def denoise_video(model, frames, sequence_length,
                  batch_size, background_level):
    print("Preparing frames for denoising...")
    frames_processed = frames.astype(np.float32) - background_level
    denoised_frames = []
    num_frames = len(frames)

    print(f"Starting denoising loop for {num_frames} frames...")
    for idx in range(0, num_frames, batch_size):
        batch_sequences = []
        batch_indices = range(idx, min(idx + batch_size, num_frames))

        for frame_idx in batch_indices:
            seq = prepare_sequence(frames_processed, frame_idx, sequence_length)
            batch_sequences.append(seq)

        num_to_pad = batch_size - len(batch_sequences)
        if num_to_pad > 0:
            pad_sequence = batch_sequences[-1] 
            for _ in range(num_to_pad):
                batch_sequences.append(pad_sequence)

        X = np.array(batch_sequences)[..., np.newaxis]

        Y_pred = model.predict(X, verbose=0) # shape: (batch_size, H, W, 1)
        Y_pred = np.squeeze(Y_pred, axis=-1)  # shape: (batch_size, H, W)

        num_actual_frames = len(batch_indices)
        for i in range(num_actual_frames):
            denoised_frames.append(Y_pred[i])

        print(f"  Processed frames {idx} to {min(idx + batch_size, num_frames)} / {num_frames}")

    denoised_frames = np.array(denoised_frames)
    denoised_frames = denoised_frames + background_level
    print("Denoising complete.")
    return denoised_frames

def main(args):
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    print(f"Loading raw video from: {args.input_file}")
    frames = load_frames(args.input_file)
    if frames is None:
        return
    print(f"Loaded {len(frames)} frames with shape: {frames[0].shape} (H,W)")

    print(f"Loading noise parameters from: {args.noise_params_file}")
    try:
        noise_params = np.load(args.noise_params_file, allow_pickle=True).item()
        background_level = noise_params['background_level']
        gaussian_variance = noise_params['gaussian_variance']
        gain_estimate = noise_params['gain_estimate']
        print(f"  Background: {background_level:.2f}")
        print(f"  Variance:   {gaussian_variance:.2f}")
        print(f"  Gain:       {gain_estimate:.2f}")
    except Exception as e:
        print(f"Error: Could not load or parse noise parameters file.")
        print(e)
        return
    
    print(f"Loading model from: {args.model_file}")
    custom_objects = {
        'CentralFrameExtractionLayer': CentralFrameExtractionLayer,
        'PoissonGaussianNLLLoss': PoissonGaussianNLLLoss,
        'PoissonGaussianNLLLossWithGeometry': PoissonGaussianNLLLossWithGeometry,
        # 'poisson_gaussian_nll_loss': poisson_gaussian_nll_loss # part of a class
    }
    try:
        model = tf.keras.models.load_model(args.model_file, custom_objects=custom_objects)
    except Exception as e:
        print(f"Error: Could not load model.")
        print("This often happens if the model was trained with a custom loss")
        print("(e.g., PoissonGaussianNLLLossWithGeometry) but you are not")
        print("loading it with that loss in custom_objects.")
        print(e)
        return

    model.summary()

    print("Denoising video...")
    denoised_frames = denoise_video(
        model, frames,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        background_level=background_level
    )

    print(f"Clipping values to [{args.clip_min}, {args.clip_max}] and converting to {args.output_dtype}")
    output_dtype = np.dtype(args.output_dtype)
    denoised_frames = np.clip(denoised_frames, args.clip_min, args.clip_max).astype(output_dtype)

    print(f"Saving denoised video to: {args.output_file}")
    try:
        tifffile.imwrite(
            args.output_file,
            denoised_frames,
            imagej=True,
            metadata={'axes': 'TYX'} # T=time, Y=height, X=width
        )
        print("Successfully saved denoised video.")
    except Exception as e:
        print(f"Error: Could not save output TIFF file.")
        print(e)
        return

    print("\n===== Video Statistics =====")
    print(f"Num frames: {len(frames)}")
    print(f"Original range: [{frames.min()}, {frames.max()}] (dtype: {frames.dtype})")
    print(f"Denoised range: [{denoised_frames.min()}, {denoised_frames.max()}] (dtype: {denoised_frames.dtype})")
    print(f"Original mean: {frames.mean():.2f}")
    print(f"Denoised mean: {denoised_frames.mean():.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Denoise a video (multi-page TIFF) using a trained 3D U-Net model."
    )

    # --- I/O Arguments ---
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input (multi-page) TIFF file to denoise."
    )
    parser.add_argument(
        "--model_file",
        type=str,
        required=True,
        help="Path to the trained .keras model file."
    )
    parser.add_argument(
        "--noise_params_file",
        type=str,
        required=True,
        help="Path to the .npy file containing the noise parameters (background, variance, gain)."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the denoised output (multi-page) TIFF file."
    )

    # --- Model & Inference Arguments ---
    parser.add_argument(
        "--sequence_length",
        type=int,
        required=True,
        help="Number of frames in each sequence (must be odd). MUST match the sequence_length used for training."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference. Adjust based on GPU memory."
    )

    # --- Output Formatting Arguments ---
    parser.add_argument(
        "--clip_min",
        type=float,
        default=0.0,
        help="Minimum value to clip the output pixels to."
    )
    parser.add_argument(
        "--clip_max",
        type=float,
        default=65535.0,
        help="Maximum value to clip the output pixels to (e.g., 65535 for uint16)."
    )
    parser.add_argument(
        "--output_dtype",
        type=str,
        default='uint16',
        help="Numpy data type for the output file (e.g., 'uint16', 'uint8', 'float32')."
    )

    # --- Misc Arguments ---
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )

    args = parser.parse_args()

    if args.sequence_length % 2 == 0:
        parser.error("--sequence_length must be an odd number (e.g., 1, 3, 5).")

    if not Path(args.input_file).exists():
        parser.error(f"Input file not found: {args.input_file}")
    if not Path(args.model_file).exists():
        parser.error(f"Model file not found: {args.model_file}")
    if not Path(args.noise_params_file).exists():
        parser.error(f"Noise parameters file not found: {args.noise_params_file}")

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    main(args)
