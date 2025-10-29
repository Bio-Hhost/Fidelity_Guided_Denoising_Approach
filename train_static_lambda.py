import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable
import random
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import argparse
from pathlib import Path

@register_keras_serializable()
class CentralFrameExtractionLayer(Layer):
    def __init__(self, sequence_length, **kwargs):
        super(CentralFrameExtractionLayer, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.center = sequence_length // 2

    def call(self, inputs):
        #Input shape: (batch, height, width, sequence, channels)
        #Output shape: (batch, height, width, channels)
        return inputs[:, :, :, self.center, :]

    def get_config(self):
        config = super(CentralFrameExtractionLayer, self).get_config()
        config.update({'sequence_length': self.sequence_length})
        return config

class VideoDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, frames, batch_size=4, sequence_length=5, mask_ratio=0.10, background_level=0):
        self.frames = np.array(frames).astype(np.float32)
        self.frames = self.frames - background_level

        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.mask_ratio = mask_ratio 

        self.indices = np.arange(len(self.frames) - self.sequence_length + 1)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        X = []
        Y = []
        for index in batch_indices:
            sequence = self.frames[index:index + self.sequence_length]
            volume = np.stack(sequence, axis=-1)
            mask = self.generate_blind_spot_mask(volume.shape[:-1])

            input_volume = volume.copy()
            target_volume = volume.copy()

            center_frame_idx = self.sequence_length // 2
            input_volume[..., center_frame_idx][mask == 1] = 0

            # Y shape: (H, W, 2) where [..., 0] is ground truth, [..., 1] is mask
            target_with_mask = np.stack([target_volume[..., center_frame_idx], mask], axis=-1)

            X.append(input_volume)
            Y.append(target_with_mask)

        # X shape: (batch, H, W, T) -> add channel dim: (batch, H, W, T, 1)
        # Y shape: (batch, H, W, 2)
        X = np.array(X)[..., np.newaxis]
        Y = np.array(Y)
        return X, Y

    def generate_blind_spot_mask(self, shape):
        mask = np.random.choice([0, 1], size=shape, p=[1 - self.mask_ratio, self.mask_ratio]).astype(np.float32)
        return mask

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


@register_keras_serializable()
class PoissonGaussianNLLLoss(tf.keras.losses.Loss):
    def __init__(self, gain, sigma_sq, epsilon=1e-7, **kwargs):
        super(PoissonGaussianNLLLoss, self).__init__(**kwargs)
        self.gain = gain
        self.sigma_sq = sigma_sq
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        y_pred = tf.nn.relu(y_pred) + self.epsilon
        term1 = (y_true - self.gain * y_pred) ** 2 / (2.0 * self.sigma_sq)
        term2 = self.gain * y_pred
        term3 = -y_true * tf.math.log(self.gain * y_pred + self.epsilon)
        nll_loss = term1 + term2 + term3
        return tf.reduce_mean(nll_loss)

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
        mask = y_true[..., 1]  # 1 where masked
        y_true_values = y_true[..., 0]

        y_pred = tf.nn.relu(y_pred) + self.epsilon
        y_true_values_expanded = tf.expand_dims(y_true_values, axis=-1)

        term1 = (y_true_values_expanded - self.gain * y_pred) ** 2 / (2.0 * self.sigma_sq)
        term2 = self.gain * y_pred
        term3 = -y_true_values_expanded * tf.math.log(self.gain * y_pred + self.epsilon)

        nll_loss = term1 + term2 + term3

        #apply only where mask=1
        #Expand mask dim to match nll_loss (H, W, 1)
        mask_expanded = tf.expand_dims(mask, axis=-1)
        masked_loss = nll_loss * mask_expanded
        masked_loss_mean = tf.reduce_sum(masked_loss) / (tf.reduce_sum(mask_expanded) + 1e-7)

        unmask = 1.0 - mask  # 1 where not masked

        fidelity = tf.square(y_pred - y_true_values_expanded)
        fidelity = fidelity * tf.expand_dims(unmask, axis=-1)

        fidelity_mean = tf.reduce_sum(fidelity) / (tf.reduce_sum(unmask) + 1e-7)
        total_loss = masked_loss_mean + self.lambda_geo * fidelity_mean
        return total_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'gain': self.gain,
            'sigma_sq': self.sigma_sq,
            'epsilon': self.epsilon,
            'lambda_geo': self.lambda_geo
        })
        return config

def build_3d_unet(input_shape, sequence_length):
    inputs = layers.Input(input_shape)

    # Encoding path
    conv1 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPool3D(pool_size=(2, 2, 1))(conv1)  # No pooling in time dim

    conv2 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPool3D(pool_size=(2, 2, 1))(conv2)

    conv3 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPool3D(pool_size=(2, 2, 1))(conv3)

    # Bridge
    conv4 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)

    # Decoding path
    up1 = layers.Conv3DTranspose(128, (2, 2, 1), strides=(2, 2, 1), padding='same')(conv4)
    up1 = layers.concatenate([up1, conv3])
    conv5 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up1)
    conv5 = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv5)

    up2 = layers.Conv3DTranspose(64, (2, 2, 1), strides=(2, 2, 1), padding='same')(conv5)
    up2 = layers.concatenate([up2, conv2])
    conv6 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up2)
    conv6 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv6)

    up3 = layers.Conv3DTranspose(32, (2, 2, 1), strides=(2, 2, 1), padding='same')(conv6)
    up3 = layers.concatenate([up3, conv1])
    conv7 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up3)
    conv7 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv7)

    outputs = layers.Conv3D(1, (1, 1, 1), activation='relu', padding='same')(conv7)

    #extract the central frame
    outputs = CentralFrameExtractionLayer(sequence_length)(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def load_frames(filepath):
    ret, frames = cv2.imreadmulti(str(filepath), flags=cv2.IMREAD_UNCHANGED)
    if not ret:
        print(f"Error loading TIFF file: {filepath}")
        return None
    return np.array(frames)

def validate_region(region, frame_shape):
    height, width = frame_shape
    x1, y1, x2, y2 = region

    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        return False
    if (x2 - x1) < 5 or (y2 - y1) < 5:  # Minimum size check
        return False
    return True

def analyze_noise_regions(frames, noise_regions, plot=False):
    if frames.ndim == 0 or len(frames) == 0:
        raise ValueError("Input frames array is empty or invalid.")

    height, width = frames[0].shape
    valid_regions = [region for region in noise_regions
                     if validate_region(region, (height, width))]

    if not valid_regions:
        raise ValueError("No valid noise regions found or provided.")

    noise_pixels = []
    region_stats = []

    for region_idx, (x1, y1, x2, y2) in enumerate(valid_regions):
        region_pixels_all_frames = []
        for frame in frames:
            region = frame[y1:y2, x1:x2]
            if region.size > 0:
                region_pixels_all_frames.extend(region.flatten())

        if not region_pixels_all_frames:
            print(f"Warning: Region {region_idx} is empty. Skipping.")
            continue

        region_pixels = np.array(region_pixels_all_frames)
        region_mean = np.mean(region_pixels)
        region_var = np.var(region_pixels, ddof=1)

        region_stats.append({
            'region': region_idx,
            'mean': region_mean,
            'variance': region_var,
            'size': len(region_pixels)
        })
        noise_pixels.extend(region_pixels)

    if not noise_pixels:
        raise ValueError("No valid pixels found in noise regions")

    noise_pixels = np.array(noise_pixels)

    background_level = np.median(noise_pixels)
    mad = np.median(np.abs(noise_pixels - background_level)) 
    gaussian_variance = (mad * 1.4826) ** 2 

    if plot:
        plt.figure(figsize=(10, 5))
        bins = np.linspace(np.min(noise_pixels), np.max(noise_pixels), 50)
        plt.hist(noise_pixels, bins=bins, density=True, alpha=0.7, label='Histogram')
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, background_level, np.sqrt(gaussian_variance))
        plt.plot(x, p, 'k', linewidth=2, label='Fitted Normal')
        plt.title('Noise Distribution')
        plt.legend()
        plt.savefig("noise_distribution_plot.png")
        print("Saved noise distribution plot to noise_distribution_plot.png")
        plt.close()

    return background_level, gaussian_variance, region_stats


def analyze_intensity_variance_relationship(frames, background_level, patch_size=32, plot=False):
    means = []
    variances = []
    height, width = frames[0].shape

    for frame in frames:
        frame_corrected = frame - background_level
        for y in range(0, height - patch_size, patch_size):
            for x in range(0, width - patch_size, patch_size):
                patch = frame_corrected[y:y + patch_size, x:x + patch_size]
                if patch.size != patch_size * patch_size:
                    continue
                patch_mean = np.mean(patch)
                patch_variance = np.var(patch, ddof=1)

                if not np.isnan(patch_mean) and not np.isnan(patch_variance):
                    means.append(patch_mean)
                    variances.append(patch_variance)

    if not means:
        raise ValueError("No valid patches found for analysis")

    means = np.array(means)
    variances = np.array(variances)

    #remove outliers and negative means
    q1_mean = np.percentile(means, 25)
    q3_mean = np.percentile(means, 75)
    iqr_mean = q3_mean - q1_mean
    mean_mask = (means >= q1_mean - 1.5 * iqr_mean) & \
                (means <= q3_mean + 1.5 * iqr_mean)

    q1_var = np.percentile(variances, 25)
    q3_var = np.percentile(variances, 75)
    iqr_var = q3_var - q1_var
    var_mask = (variances >= q1_var - 1.5 * iqr_var) & \
               (variances <= q3_var + 1.5 * iqr_var)

    valid_mask = mean_mask & var_mask & (means > 0)

    if not np.any(valid_mask):
        raise ValueError("No valid data points after filtering for gain estimation")

    means_filtered = means[valid_mask]
    variances_filtered = variances[valid_mask]

    X = means_filtered.reshape(-1, 1)
    y = variances_filtered

    linear_reg = LinearRegression()
    linear_reg.fit(X, y)

    gain_estimate = linear_reg.coef_[0]
    intercept = linear_reg.intercept_

    if gain_estimate <= 0:
        print(f"Warning: Estimated gain is non-positive ({gain_estimate:.2f}). Clamping to 1.0.")
        gain_estimate = 1.0

    if plot:
        plt.figure(figsize=(8, 6))
        plt.scatter(means_filtered, variances_filtered, alpha=0.1, s=1, label="Data (filtered)")
        plt.plot(means_filtered, linear_reg.predict(X), 'r-', label=f'Linear Fit (Gain={gain_estimate:.2f})')
        plt.xlabel('Mean Intensity (Corrected)')
        plt.ylabel('Variance')
        plt.title('Variance vs. Mean Intensity')
        plt.legend()
        plt.savefig("gain_estimation_plot.png")
        print("Saved gain estimation plot to gain_estimation_plot.png")
        plt.close()

    return {
        'gain_linear': gain_estimate,
        'intercept': intercept
    }

def sequential_split(frames, train_ratio=0.7, val_ratio=0.15):
    total_frames = len(frames)
    train_end = int(total_frames * train_ratio)
    val_end = train_end + int(total_frames * val_ratio)
    if val_end >= total_frames - 1:
        val_end = total_frames - 2 
        if train_end >= val_end:
            train_end = val_end - 1

    if train_end <= 0:
        raise ValueError("Not enough frames to create a training set.")

    train_frames = frames[:train_end]
    val_frames = frames[train_end:val_end]
    test_frames = frames[val_end:]

    print(f"Data split: {len(train_frames)} train, {len(val_frames)} val, {len(test_frames)} test frames.")
    return train_frames, val_frames, test_frames

def main(args):
    # seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    print("Loading frames...")
    all_frames_list = []
    for f_path in args.input_files:
        frames = load_frames(Path(f_path))
        if frames is not None:
            all_frames_list.append(frames)

    if not all_frames_list:
        print("Error: No valid TIFF files were loaded. Exiting.")
        return

    all_frames = np.concatenate(all_frames_list, axis=0)
    print(f"Loaded a total of {len(all_frames)} frames.")

    if len(all_frames) < args.sequence_length + 2:
        print(
            f"Error: Not enough frames ({len(all_frames)}) to create train/val/test splits with sequence length {args.sequence_length}. Need at least {args.sequence_length + 2}.")
        return

    if args.background_level is not None and args.gaussian_variance is not None and args.gain_estimate is not None:
        print("Using provided noise parameters.")
        background_level = args.background_level
        gaussian_variance = args.gaussian_variance
        gain_estimate = args.gain_estimate
    else:
        print("\nEstimating noise parameters from data...")
        # Default noise regions from the original script
        # These may need to be adjusted for different datasets
        noise_regions = [
            (0, 190, 50, 250),  # bottom left
            (200, 190, 250, 250),  # bottom right
            (0, 10, 40, 50),  # top left
            (220, 10, 250, 50)  # top right
        ]
        try:
            background_level, gaussian_variance, _ = analyze_noise_regions(
                all_frames, noise_regions, plot=args.plot_noise
            )
            print(f"Estimated Background Level: {background_level:.2f}")
            print(f"Estimated Gaussian Variance: {gaussian_variance:.2f}")

            print("\nEstimating gain...")
            results = analyze_intensity_variance_relationship(
                all_frames, background_level, plot=args.plot_noise
            )
            gain_estimate = results['gain_linear']
            print(f"Estimated Gain: {gain_estimate:.2f}")

        except ValueError as e:
            print(f"Error during noise estimation: {e}")
            print("Please check your data or provide noise parameters manually.")
            return

    noise_params = {
        'background_level': background_level,
        'gaussian_variance': gaussian_variance,
        'gain_estimate': gain_estimate
    }
    np.save(args.output_noise_params, noise_params)
    print(f"Saved noise parameters to {args.output_noise_params}")

    train_frames, val_frames, test_frames = sequential_split(
        all_frames, train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )

    frame_height, frame_width = train_frames[0].shape
    input_shape = (frame_height, frame_width, args.sequence_length, 1)

    print(f"Model input shape will be: {input_shape}")

    train_generator = VideoDataGenerator(
        train_frames,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        mask_ratio=args.mask_ratio,
        background_level=background_level
    )
    val_generator = VideoDataGenerator(
        val_frames,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        mask_ratio=args.mask_ratio,
        background_level=background_level
    )

    print("Building 3D U-Net model...")
    model = build_3d_unet(input_shape, args.sequence_length)
    model.summary()

    opt = tf.keras.optimizers.Adam(
        learning_rate=args.learning_rate,
        clipnorm=1.0  # From original script
    )

    loss_fn = PoissonGaussianNLLLossWithGeometry(
        gain=gain_estimate,
        sigma_sq=gaussian_variance,
        lambda_geo=args.lambda_geo,
        epsilon=1e-7
    )

    model.compile(optimizer=opt, loss=loss_fn)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=args.es_patience,
        restore_best_weights=True,
        min_delta=1e-5  # Slightly larger delta
    )

    model_checkpoint = ModelCheckpoint(
        filepath=args.output_model,
        monitor="val_loss",
        save_best_only=True,
        mode="min"
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=args.lr_patience,
        min_lr=1e-7,  # Lower min_lr
        verbose=1
    )
    callbacks_list = [early_stopping, model_checkpoint, reduce_lr]

    print("Starting training...")
    history = model.fit(
        x=train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        callbacks=callbacks_list,
        workers=max(1, os.cpu_count() // 2) 
    )

    np.save(args.output_history, history.history)
    print(f"Training complete. Best model saved to {args.output_model}")
    print(f"Training history saved to {args.output_history}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a 3D U-Net for video denoising with blind-spot masking."
    )

    # --- I/O Arguments ---
    parser.add_argument(
        "--input_files",
        nargs='+',
        required=True,
        help="One or more paths to the input (multi-page) TIFF files."
    )
    parser.add_argument(
        "--output_model",
        type=str,
        required=True,
        help="Path to save the best trained model (.keras file)."
    )
    parser.add_argument(
        "--output_history",
        type=str,
        default="training_history.npy",
        help="Path to save the training history (.npy file)."
    )
    parser.add_argument(
        "--output_noise_params",
        type=str,
        default="noise_parameters.npy",
        help="Path to save the estimated noise parameters (.npy file)."
    )

    # --- Data & Model Arguments ---
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=1,
        help="Number of frames in each video sequence (must be odd, e.g., 1, 3, 5)."
    )
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.05,
        help="Ratio of pixels to mask in the central frame for self-supervision."
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Fraction of data to use for training."
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Fraction of data to use for validation."
    )

    # --- Training Arguments ---
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Maximum number of training epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Initial learning rate for the Adam optimizer."
    )
    parser.add_argument(
        "--lambda_geo",
        type=float,
        default=0.1,
        help="Weighting factor (lambda) for the geometry/fidelity loss term."
    )
    parser.add_argument(
        "--es_patience",
        type=int,
        default=30,
        help="Patience (epochs) for Early Stopping."
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=10,
        help="Patience (epochs) for ReduceLROnPlateau."
    )

    # --- Noise Estimation Arguments ---
    parser.add_argument(
        "--background_level",
        type=float,
        default=None,
        help="Manually provide background level. (Skips estimation if all 3 noise params are set)"
    )
    parser.add_argument(
        "--gaussian_variance",
        type=float,
        default=None,
        help="Manually provide Gaussian variance. (Skips estimation if all 3 noise params are set)"
    )
    parser.add_argument(
        "--gain_estimate",
        type=float,
        default=None,
        help="Manually provide gain estimate. (Skips estimation if all 3 noise params are set)"
    )
    parser.add_argument(
        "--plot_noise",
        action='store_true',
        help="Save plots for noise distribution and gain estimation."
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
        parser.error("--sequence_length must be an odd number (e.g., 3, 5, 7).")
    if args.train_ratio + args.val_ratio >= 1.0:
        parser.error("--train_ratio and --val_ratio must sum to less than 1.0.")

    Path(args.output_model).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_history).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_noise_params).parent.mkdir(parents=True, exist_ok=True)

    main(args)
