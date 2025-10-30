import os
import cv2
import json
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.optimize import curve_fit
from tensorflow.keras import layers, Model, optimizers
import random
from collections import deque
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable
import matplotlib.pyplot as plt
from scipy.stats import norm
import argparse
from pathlib import Path

def analyze_noise_regions(frames, noise_regions):
    noise_pixels = []
    for region in noise_regions:
        x1, y1, x2, y2 = region
        for frame in frames:
            noise_pixels.extend(frame[y1:y2, x1:x2].flatten())
    if not noise_pixels:
        raise ValueError("No pixels found in noise regions. Check region definitions.")
    noise_pixels = np.array(noise_pixels)
    background_level = np.median(noise_pixels)
    mad = np.median(np.abs(noise_pixels - background_level))
    gaussian_variance = (mad * 1.4826) ** 2
    return background_level, gaussian_variance

def analyze_intensity_variance_relationship(frames, background_level, patch_size=32):
    means, variances = [], []
    height, width = frames[0].shape
    for frame in frames:
        frame_corrected = frame.astype(np.float32) - background_level
        for y in range(0, height - patch_size, patch_size):
            for x in range(0, width - patch_size, patch_size):
                patch = frame_corrected[y:y+patch_size, x:x+patch_size]
                if patch.size > 0:
                    means.append(np.mean(patch))
                    variances.append(np.var(patch, ddof=1))
    if not means:
        raise ValueError("Could not extract patches for gain estimation.")
    means, variances = np.array(means), np.array(variances)
    valid_mask = ~np.isnan(means) & ~np.isnan(variances) & (means > 0)
    if not np.any(valid_mask):
        raise ValueError("No valid patches after filtering for gain estimation.")
    gain_estimate = np.polyfit(means[valid_mask], variances[valid_mask], 1)[0]
    return max(gain_estimate, 1e-6)

def zoom_spot_loc(video_frame, spot_position, region_size):
    x_int_raw, y_int_raw = spot_position[0], spot_position[1]
    h, w = video_frame.shape
    half_size = region_size // 2
    y1_ideal, y2_ideal = int(round(y_int_raw)) - half_size, int(round(y_int_raw)) + half_size + 1
    x1_ideal, x2_ideal = int(round(x_int_raw)) - half_size, int(round(x_int_raw)) + half_size + 1
    y1_clipped, y2_clipped = max(0, y1_ideal), min(h, y2_ideal)
    x1_clipped, x2_clipped = max(0, x1_ideal), min(w, x2_ideal)
    region = video_frame[y1_clipped:y2_clipped, x1_clipped:x2_clipped]
    return region, (x1_clipped, y1_clipped)

def rotated_2d_gaussian(coords, amplitude, x0, y0, sigma_x, sigma_y, theta_deg, offset):
    """Defines a 2D Gaussian function with rotation."""
    (x, y) = coords
    theta = np.deg2rad(theta_deg)
    X_centered, Y_centered = x - x0, y - y0
    x_prime = X_centered * np.cos(theta) + Y_centered * np.sin(theta)
    y_prime = -X_centered * np.sin(theta) + Y_centered * np.cos(theta)
    sx2, sy2 = sigma_x**2, sigma_y**2
    exponent = (x_prime**2) / (2 * sx2 + 1e-7) + (y_prime**2) / (2 * sy2 + 1e-7)
    g = offset + amplitude * np.exp(-exponent)
    return g

def fit_rotated_gaussian_2d(region, global_x1, global_y1):
    h, w = region.shape
    if h * w < 4: return False, None
    Y_mesh, X_mesh = np.mgrid[global_y1:global_y1+h, global_x1:global_x1+w]
    z_flat = region.ravel().astype(float)
    min_val, max_val = float(np.min(region)), float(np.max(region))
    amp_guess = max(max_val - min_val, 1e-6)
    initial_guess = (amp_guess, global_x1+w/2.0, global_y1+h/2.0, 1.0, 1.0, 0.0, min_val)
    bounds = ([0, -np.inf, -np.inf, 0.2, 0.2, -180, -np.inf], [np.inf, np.inf, np.inf, w, h, 180, np.inf])
    def fit_func(coords, *args): return rotated_2d_gaussian(coords, *args).ravel()
    try:
        popt, _ = curve_fit(fit_func, (X_mesh, Y_mesh), z_flat, p0=initial_guess, bounds=bounds, maxfev=3000, method='trf', ftol=1e-3, xtol=1e-3)
        if popt[0] <= 0 or popt[3] < 0.2 or popt[4] < 0.2: return False, None
        return True, {'fit_amplitude': popt[0], 'fit_sx': popt[3], 'fit_sy': popt[4]}
    except Exception: return False, None

def create_background_mask_from_spots(frame_shape, spots_df, exclude_radius_px, config):
    h, w = frame_shape
    bg_mask = np.ones((h, w), dtype=bool)
    if spots_df is None or spots_df.empty: return bg_mask
    y_coords_grid, x_coords_grid = np.ogrid[:h, :w]
    for _, spot in spots_df.iterrows():
        spot_x, spot_y = spot[config.csv_x_col], spot[config.csv_y_col]
        dist_sq_from_spot = (x_coords_grid - spot_x)**2 + (y_coords_grid - spot_y)**2
        bg_mask[dist_sq_from_spot <= exclude_radius_px**2] = False
    return bg_mask

def find_csv_header(config):
    with open(config.spots_csv_path, 'r') as f:
        for i, line in enumerate(f):
            if config.csv_x_col in line and config.csv_frame_col in line:
                print(f"Found CSV header at line {i}")
                return i
    print("Warning: Could not find CSV header. Defaulting to row 0.")
    return 0 

def process_spot_data(df, config):
    cols_to_convert = [
        config.csv_frame_col, config.csv_x_col, config.csv_y_col,
        config.csv_snr_col, config.csv_quality_col
    ]
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    original_rows = len(df)
    df.dropna(subset=[config.csv_frame_col, config.csv_x_col, config.csv_y_col], inplace=True)
    if original_rows > len(df):
        print(f"Removed {original_rows - len(df)} rows due to invalid/NaN values in essential columns.")

    df[config.csv_frame_col] = df[config.csv_frame_col].astype(int)
    return {frame_idx: group_df for frame_idx, group_df in df.groupby(config.csv_frame_col)}

def estimate_noise_parameters(all_frames, regions):
    background_level, gaussian_variance = analyze_noise_regions(all_frames, regions)
    gain_estimate = analyze_intensity_variance_relationship(all_frames, background_level)
    return {'gain': gain_estimate, 'sigma_sq': gaussian_variance, 'background_level': background_level}

def create_data_generators(all_frames, spots_map, config):
    num_frames = len(all_frames)
    indices = np.arange(num_frames)
    np.random.shuffle(indices)
    split_idx = int(num_frames * config.data_split['train'])
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]

    train_generator = RLDataGenerator(
        all_frames[train_indices], spots_map, config, is_validation=False
    )
    val_generator = RLDataGenerator(
        all_frames[val_indices], spots_map, config, is_validation=True
    )
    return train_generator, val_generator

class RLDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, frames, spots_map, config, mask_ratio=0.1, is_validation=False):
        self.frames = frames.astype(np.float32)
        self.spots_map = spots_map
        self.sequence_length = config.sequence_length
        self.batch_size = config.unet_batch_size
        self.mask_ratio = mask_ratio
        self.is_validation = is_validation 

        self.h, self.w = self.frames[0].shape
        self.center_offset = self.sequence_length // 2

        num_frames = len(self.frames)
        self.indices = np.arange(num_frames - self.sequence_length + 1)
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        X_seq_b, Y_true_center_b, mask_b = [], [], []
        X_center_noisy_b, tm_spots_list_b = [], []

        for seq_start_index in batch_indices:
            sequence = self.frames[seq_start_index : seq_start_index + self.sequence_length]
            X_seq = np.stack(sequence, axis=-1) # Shape: (H, W, Seq)
            X_seq = np.expand_dims(X_seq, axis=-1) # Shape: (H, W, Seq, 1)

            center_frame_global_idx = seq_start_index + self.center_offset
            center_frame_noisy = self.frames[center_frame_global_idx]

            mask = np.random.choice([0, 1], size=(self.h, self.w), p=[1 - self.mask_ratio, self.mask_ratio]).astype(np.float32)

            X_seq_masked = X_seq.copy()
            center_frame_in_seq = X_seq_masked[:, :, self.center_offset, 0]
            center_frame_in_seq[mask == 1] = 0 

            spots_df = self.spots_map.get(center_frame_global_idx, pd.DataFrame())

            X_seq_b.append(X_seq_masked)
            Y_true_center_b.append(center_frame_noisy.copy())
            mask_b.append(mask)
            X_center_noisy_b.append(center_frame_noisy)
            tm_spots_list_b.append(spots_df)

        return (
            np.array(X_seq_b),
            (np.array(Y_true_center_b), np.array(mask_b)),
            tm_spots_list_b,
            np.array(X_center_noisy_b)
        )

    def on_epoch_end(self):
        if not self.is_validation:
            np.random.shuffle(self.indices)

def calculate_self_supervised_reward_batch(X_center_batch_tf, y_pred_batch_tf, trackmate_spots_batch, config, noise_params):
    batch_size = X_center_batch_tf.shape[0]
    batch_per_sample_rewards = []
    batch_reward_details_list = []
    X_np = X_center_batch_tf.numpy().squeeze()
    y_pred_np = y_pred_batch_tf.numpy().squeeze()
    
    for i in range(batch_size):
        X_frame, y_pred_frame = X_np[i], y_pred_np[i]
        tm_spots = trackmate_spots_batch[i]
        frame_rewards_dict = {'bg_smoothness': 0.0, 'spot_sharpness': 0.0, 'intensity_consistency': 0.0, 'fit_success_rate': 0.0}

        if config.rl_reward_config['weights'].get('bg_smoothness', 0) > 0:
            bg_mask = create_background_mask_from_spots(X_frame.shape, tm_spots, config.rl_reward_config.get('spot_exclude_radius_px', 5), config)
            if np.any(bg_mask):
                var_X_bg = np.var(X_frame[bg_mask])
                var_y_pred_bg = np.var(y_pred_frame[bg_mask])
                frame_rewards_dict['bg_smoothness'] = var_X_bg - var_y_pred_bg

        num_tm_spots = len(tm_spots) if tm_spots is not None else 0
        sharpness_vals, intensity_vals, fit_success_count = [], [], 0
        if num_tm_spots > 0:
            for _, spot_row in tm_spots.iterrows():
                tm_pos = (spot_row[config.csv_x_col], spot_row[config.csv_y_col])
                patch_ypred, (gx1, gy1) = zoom_spot_loc(y_pred_frame, tm_pos, config.rl_reward_config['fit_region_size'])
                fit_ok, params = fit_rotated_gaussian_2d(patch_ypred, gx1, gy1)
                if fit_ok:
                    fit_success_count += 1
                    sharpness_vals.append(-(params['fit_sx'] + params['fit_sy']))
                    patch_X, (gx1_X, gy1_X) = zoom_spot_loc(X_frame, tm_pos, config.rl_reward_config['fit_region_size'])
                    fit_ok_X, params_X = fit_rotated_gaussian_2d(patch_X, gx1_X, gy1_X)
                    if fit_ok_X:
                        intensity_vals.append(-abs(params['fit_amplitude'] - params_X['fit_amplitude']))

            if sharpness_vals: frame_rewards_dict['spot_sharpness'] = np.mean(sharpness_vals)
            if intensity_vals: frame_rewards_dict['intensity_consistency'] = np.mean(intensity_vals)
            frame_rewards_dict['fit_success_rate'] = fit_success_count / num_tm_spots if num_tm_spots > 0 else 0

        total_reward = sum(weight * frame_rewards_dict.get(key, 0.0) for key, weight in config.rl_reward_config['weights'].items())
        batch_per_sample_rewards.append(total_reward)
        batch_reward_details_list.append(frame_rewards_dict)

    return tf.convert_to_tensor(batch_per_sample_rewards, dtype=tf.float32), batch_reward_details_list

def calculate_snr_reward(denoised_images_tf, trackmate_spots_list, config):
    denoised_images_np = denoised_images_tf.numpy().squeeze()
    batch_rewards = []

    signal_radius = config.rl_reward_config.get('signal_radius', 2)
    bg_inner_radius = config.rl_reward_config.get('bg_inner_radius', 3)
    bg_outer_radius = config.rl_reward_config.get('bg_outer_radius', 5)

    for i in range(denoised_images_np.shape[0]): 
        image = denoised_images_np[i]
        spots_df = trackmate_spots_list[i]

        if spots_df is None or spots_df.empty:
            batch_rewards.append(0.0)
            continue

        spot_snrs = []
        h, w = image.shape
        y_coords, x_coords = np.ogrid[:h, :w]

        for _, spot in spots_df.iterrows():
            x_c, y_c = int(round(spot[config.csv_x_col])), int(round(spot[config.csv_y_col]))

            signal_mask = (x_coords - x_c)**2 + (y_coords - y_c)**2 <= signal_radius**2
            bg_mask_outer = (x_coords - x_c)**2 + (y_coords - y_c)**2 <= bg_outer_radius**2
            bg_mask_inner = (x_coords - x_c)**2 + (y_coords - y_c)**2 < bg_inner_radius**2
            background_mask = bg_mask_outer & ~bg_mask_inner

            if x_c < bg_outer_radius or x_c >= w - bg_outer_radius or \
               y_c < bg_outer_radius or y_c >= h - bg_outer_radius:
                continue 

            signal_pixels = image[signal_mask]
            background_pixels = image[background_mask]

            if background_pixels.size > 1:
                mean_signal = np.mean(signal_pixels)
                mean_bg = np.mean(background_pixels)
                std_bg = np.std(background_pixels)

                if std_bg > 1e-6: 
                    snr = (mean_signal - mean_bg) / std_bg
                    spot_snrs.append(snr)

        if not spot_snrs:
            batch_rewards.append(0.0)
        else:
            #the reward is the average SNR. We clip to prevent extreme values.
            avg_snr = np.mean(spot_snrs)
            batch_rewards.append(np.clip(avg_snr, -10, 50))

    return tf.constant(batch_rewards, dtype=tf.float32), {}

@register_keras_serializable()
class CentralFrameExtractionLayer(Layer):
    def __init__(self, sequence_length, **kwargs):
        super(CentralFrameExtractionLayer, self).__init__(**kwargs)
        self.sequence_length = sequence_length
        self.center = sequence_length // 2
    def call(self, inputs): return inputs[:, :, :, self.center, :]
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

class ReplayBuffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        self.buffer_capacity, self.batch_size, self.buffer = buffer_capacity, batch_size, deque(maxlen=buffer_capacity)
    def record(self, experience): self.buffer.append(experience)
    def sample(self):
        if len(self.buffer) < self.batch_size: return None
        batch_indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in batch_indices:
            s, a, r, s_n, d = self.buffer[idx]
            states.append(s); actions.append(a); rewards.append(r); next_states.append(s_n); dones.append(d)
        return (tf.convert_to_tensor(np.array(x), dtype=tf.float32) for x in [states, actions, rewards, next_states, dones])
    def __len__(self): return len(self.buffer)

class ActorNetwork(Model):
    def __init__(self, state_dim, action_dim, action_bounds):
        super().__init__()
        self.action_bounds = action_bounds
        self.fc1 = layers.Dense(256, activation='relu')
        self.fc2 = layers.Dense(256, activation='relu')
        self.action_out = layers.Dense(action_dim, activation='tanh')
    def call(self, states):
        x = self.fc1(states)
        x = self.fc2(x)
        action_unscaled = self.action_out(x)
        return (action_unscaled + 1.0) / 2.0 * (self.action_bounds[1] - self.action_bounds[0]) + self.action_bounds[0]

class CriticNetwork(Model):
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()
        self.state_fc1 = layers.Dense(256, activation='relu')
        self.concat = layers.Concatenate()
        self.combined_fc1 = layers.Dense(256, activation='relu')
        self.q_out = layers.Dense(1) 
    def call(self, states, actions):
        state_features = self.state_fc1(states)
        x = self.concat([state_features, actions])
        x = self.combined_fc1(x)
        q_value = self.q_out(x)
        return q_value

class DDPGAgent:
    def __init__(self, config, noise_params):
        self.config, self.noise_params = config, noise_params
        self.state_dim, self.action_dim, self.action_bounds = config.state_dim, config.action_dim, config.lambda_geo_bounds
        self.gamma, self.tau, self.batch_size = config.gamma, config.tau, config.rl_batch_size
        self.actor = ActorNetwork(self.state_dim, self.action_dim, self.action_bounds)
        self.target_actor = ActorNetwork(self.state_dim, self.action_dim, self.action_bounds); self.target_actor.set_weights(self.actor.get_weights())
        self.actor_optimizer = optimizers.Adam(learning_rate=config.actor_lr)
        self.critic = CriticNetwork(self.state_dim, self.action_dim)
        self.target_critic = CriticNetwork(self.state_dim, self.action_dim); self.target_critic.set_weights(self.critic.get_weights())
        self.critic_optimizer = optimizers.Adam(learning_rate=config.critic_lr)
        self.replay_buffer = ReplayBuffer(config.buffer_capacity, config.rl_batch_size)
        self.action_noise_stddev = config.action_noise_stddev_fraction * (self.action_bounds[1] - self.action_bounds[0])
    def record_experience(self, s, a, r, s_n, d): self.replay_buffer.record((s, a, r, s_n, d))
    def _update_target(self, main_net, target_net):
        for main_v, target_v in zip(main_net.variables, target_net.variables): target_v.assign(self.tau * main_v + (1.0 - self.tau) * target_v)

    def learn(self):
        exp_batch = self.replay_buffer.sample()
        if exp_batch is None: return None, None, None, None
        states_b, actions_b, rewards_b, next_states_b, dones_b = exp_batch
        rewards_b, dones_b = tf.reshape(rewards_b, [-1, 1]), tf.cast(tf.reshape(dones_b, [-1, 1]), dtype=tf.float32)

        with tf.GradientTape() as critic_tape:
            target_actions = self.target_actor(next_states_b, training=False)
            target_q = self.target_critic(next_states_b, target_actions, training=False)
            y_targets = rewards_b + self.gamma * target_q * (1.0 - dones_b)
            current_q = self.critic(states_b, actions_b, training=True)
            critic_loss = tf.keras.losses.Huber(delta=1.0)(y_targets, current_q)
        critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        critic_grads, critic_grad_norm = tf.clip_by_global_norm(critic_grads, 1.0)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        with tf.GradientTape() as actor_tape:
            pred_actions = self.actor(states_b, training=True)
            actor_loss = -tf.reduce_mean(self.critic(states_b, pred_actions, training=False))
        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        actor_grads, actor_grad_norm = tf.clip_by_global_norm(actor_grads, 1.0)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        self._update_target(self.actor, self.target_actor); self._update_target(self.critic, self.target_critic)
        return actor_loss, tf.reduce_mean(critic_loss), actor_grad_norm, critic_grad_norm

def calculate_unet_loss_manual(y_true, y_pred, mask, lambda_geo, noise_params):
    y_pred_relu = tf.nn.relu(y_pred) + 1e-7
    y_true_exp = tf.expand_dims(y_true, axis=-1)
    mask_exp = tf.expand_dims(mask, axis=-1)
    unmask_exp = 1.0 - mask_exp
    term1 = (y_true_exp - noise_params['gain'] * y_pred_relu)**2 / (2.0 * noise_params['sigma_sq'] + 1e-7)
    term2 = noise_params['gain'] * y_pred_relu
    term3 = -y_true_exp * tf.math.log(noise_params['gain'] * y_pred_relu)
    nll_loss = (term1 + term2 + term3) * mask_exp
    masked_loss_mean = tf.reduce_sum(nll_loss, axis=[1, 2, 3]) / (tf.reduce_sum(mask_exp, axis=[1, 2, 3]) + 1e-7)
    fidelity_loss = tf.square(y_pred_relu - y_true_exp) * unmask_exp
    fidelity_mean = tf.reduce_sum(fidelity_loss, axis=[1, 2, 3]) / (tf.reduce_sum(unmask_exp, axis=[1, 2, 3]) + 1e-7)
    total_loss = masked_loss_mean + tf.reshape(lambda_geo, [-1, 1]) * fidelity_mean
    return tf.reduce_mean(total_loss)

def create_rl_state_for_batch(X_center_batch_tf, trackmate_spots_batch, config):
    batch_states = []
    X_np = X_center_batch_tf.numpy().squeeze()
    for i in range(X_np.shape[0]):
        frame_np, spots_df = X_np[i], trackmate_spots_batch[i]
        num_tm_spots, mean_tm_snr, mean_tm_quality = 0.0, 0.0, 0.0
        if spots_df is not None and not spots_df.empty:
            num_tm_spots = len(spots_df)
            if config.csv_snr_col in spots_df.columns: mean_tm_snr = spots_df[config.csv_snr_col].mean()
            if config.csv_quality_col in spots_df.columns: mean_tm_quality = spots_df[config.csv_quality_col].mean()
        state_features = [np.mean(frame_np), np.std(frame_np), float(num_tm_spots), mean_tm_snr, mean_tm_quality]
        batch_states.append(state_features)
    return tf.convert_to_tensor(np.array(batch_states), dtype=tf.float32)

def generate_debug_images(unet, data_generator, epoch, output_dir):
    try:
        X_seq_b, (Y_true_center_b, _), _, X_center_noisy_b = data_generator[0]
        y_pred_b = unet(X_seq_b, training=False)
        idx = 0
        noisy = X_center_noisy_b[idx]
        denoised = y_pred_b[idx].numpy().squeeze()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        vmin = np.percentile(noisy, 1)
        vmax = np.percentile(noisy, 99)

        axes[0].imshow(noisy, cmap='gray', vmin=vmin, vmax=vmax)
        axes[0].set_title(f'Noisy Input (Frame {data_generator.indices[idx]})')
        axes[0].axis('off')

        axes[1].imshow(denoised, cmap='gray', vmin=vmin, vmax=vmax)
        axes[1].set_title(f'Denoised Output (Epoch {epoch+1})')
        axes[1].axis('off')

        axes[2].imshow(noisy - denoised, cmap='viridis')
        axes[2].set_title('Difference')
        axes[2].axis('off')

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"debug_comparison_epoch_{epoch+1:04d}.png"))
        plt.close(fig)
    except Exception as e:
        print(f"Warning: Could not generate debug image for epoch {epoch+1}. Error: {e}")

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    OUTPUT_DIR = os.path.join(args.base_output_path, f"training_run_{timestamp}")
    MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"--- Starting New Run ---"); print(f"All outputs will be saved to: {OUTPUT_DIR}")
    with open(os.path.join(OUTPUT_DIR, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4, default=str)

    try:
        _, all_frames = cv2.imreadmulti(args.tiff_path, flags=cv2.IMREAD_UNCHANGED)
        all_frames = np.array(all_frames, dtype=np.float32)
        print(f"Loaded {len(all_frames)} frames from {args.tiff_path}")
        
        all_spots_df = pd.read_csv(args.spots_csv_path, header=find_csv_header(args))
        trackmate_spots_by_frame = process_spot_data(all_spots_df, args)
        print(f"Processed {len(all_spots_df)} spots for {len(trackmate_spots_by_frame)} frames.")
        
        print("\nEstimating noise parameters...")
        NOISE_PARAMS = estimate_noise_parameters(all_frames, args.noise_analysis_regions)
        print(f"Estimated Noise Params: {NOISE_PARAMS}")
        all_frames -= NOISE_PARAMS['background_level']
        
    except Exception as e:
        print(f"ERROR during data loading or noise estimation. Details: {e}"); return

    train_generator, val_generator = create_data_generators(all_frames, trackmate_spots_by_frame, args)
    print(f"\nData split: {len(train_generator.frames)} train frames, {len(val_generator.frames)} validation frames.")

    input_shape = (args.img_height, args.img_width, args.sequence_length, args.channels)
    unet_model = build_3d_unet(input_shape, args.sequence_length)
    ddpg_agent = DDPGAgent(args, NOISE_PARAMS)
    unet_optimizer = optimizers.Adam(learning_rate=args.unet_lr)
    print("U-Net and DDPG Agent initialized.")

    print("\n--- Starting RL Warm-up Phase ---")
    previous_rl_states_numpy, previous_actions_numpy, previous_rewards_numpy = None, None, None
    for epoch in range(args.rl_warmup_epochs):
        print(f"Warm-up Epoch {epoch + 1}/{args.rl_warmup_epochs}")
        for step in range(args.steps_per_epoch):
            X_seq_b, (Y_true_b, Y_mask_b), tm_list_b, X_center_b = train_generator[step % len(train_generator)]
            current_rl_states_tf = create_rl_state_for_batch(tf.constant(X_center_b, dtype=tf.float32), tm_list_b, args)

            random_actions = tf.random.uniform(shape=(X_seq_b.shape[0], args.action_dim), minval=args.lambda_geo_bounds[0], maxval=args.lambda_geo_bounds[1])

            with tf.GradientTape() as unet_tape:
                y_pred_b = unet_model(X_seq_b, training=True)
                unet_loss = calculate_unet_loss_manual(Y_true_b, y_pred_b, Y_mask_b, random_actions, NOISE_PARAMS)
            unet_grads = unet_tape.gradient(unet_loss, unet_model.trainable_variables)
            unet_grads_clipped, _ = tf.clip_by_global_norm(unet_grads, 1.0)
            unet_optimizer.apply_gradients(zip(unet_grads_clipped, unet_model.trainable_variables))

            rewards_tf, _ = calculate_snr_reward(y_pred_b, tm_list_b, args)
            
            if previous_rl_states_numpy is not None and previous_rl_states_numpy.shape[0] == current_rl_states_tf.shape[0]:
                for i in range(len(previous_rl_states_numpy)):
                    ddpg_agent.record_experience(previous_rl_states_numpy[i], previous_actions_numpy[i], previous_rewards_numpy[i], current_rl_states_tf[i].numpy(), False)
            previous_rl_states_numpy, previous_actions_numpy, previous_rewards_numpy = current_rl_states_tf.numpy(), random_actions.numpy(), rewards_tf.numpy()
        print(f"Replay buffer size after warm-up epoch: {len(ddpg_agent.replay_buffer)}")

    print("\n--- Starting DDPG + U-Net Training ---")
    history = []
    best_val_reward = -np.inf
    epochs_without_improvement = 0

    for epoch in range(args.total_epochs):
        print(f"\nEpoch {epoch+1}/{args.total_epochs}")
        metrics = {k: tf.keras.metrics.Mean() for k in ['unet_loss', 'reward', 'actor_loss', 'critic_loss', 'actor_grad', 'critic_grad', 'unet_grad']}
        current_noise_stddev = ddpg_agent.action_noise_stddev * (args.noise_decay ** epoch)

        for step in range(args.steps_per_epoch):
            X_seq_b, (Y_true_b, Y_mask_b), tm_list_b, X_center_b = train_generator[step % len(train_generator)]
            current_rl_states_tf = create_rl_state_for_batch(tf.constant(X_center_b, dtype=tf.float32), tm_list_b, args)
            
            actions_deterministic = ddpg_agent.actor(current_rl_states_tf, training=False)
            noise = tf.random.normal(shape=actions_deterministic.shape, stddev=current_noise_stddev)
            actions_noisy = tf.clip_by_value(actions_deterministic + noise, *args.lambda_geo_bounds)
            
            with tf.GradientTape() as unet_tape:
                y_pred_b = unet_model(X_seq_b, training=True)
                unet_loss = calculate_unet_loss_manual(Y_true_b, y_pred_b, Y_mask_b, actions_noisy, NOISE_PARAMS)
            unet_grads = unet_tape.gradient(unet_loss, unet_model.trainable_variables)
            unet_grads_clipped, unet_grad_norm = tf.clip_by_global_norm(unet_grads, 1.0)
            unet_optimizer.apply_gradients(zip(unet_grads_clipped, unet_model.trainable_variables))
            
            rewards_tf, _ = calculate_snr_reward(y_pred_b, tm_list_b, args)
            
            if previous_rl_states_numpy is not None and previous_rl_states_numpy.shape[0] == current_rl_states_tf.shape[0]:
                for i in range(len(previous_rl_states_numpy)):
                    ddpg_agent.record_experience(previous_rl_states_numpy[i], previous_actions_numpy[i], previous_rewards_numpy[i], current_rl_states_tf[i].numpy(), False)
            previous_rl_states_numpy, previous_actions_numpy, previous_rewards_numpy = current_rl_states_tf.numpy(), actions_noisy.numpy(), rewards_tf.numpy()

            learn_results = ddpg_agent.learn()

            metrics['unet_loss'].update_state(unet_loss); metrics['unet_grad'].update_state(unet_grad_norm); metrics['reward'].update_state(tf.reduce_mean(rewards_tf))
            if learn_results and all(res is not None for res in learn_results):
                actor_loss, critic_loss, actor_grad, critic_grad = learn_results
                metrics['actor_loss'].update_state(actor_loss); metrics['critic_loss'].update_state(critic_loss); metrics['actor_grad'].update_state(actor_grad); metrics['critic_grad'].update_state(critic_grad)

        log_entry = {name: meter.result().numpy() for name, meter in metrics.items()}
        print(f"End of Epoch {epoch+1} -> " + ", ".join([f"{k}: {v:.4f}" for k, v in log_entry.items()]))
        history.append(log_entry)
        
        generate_debug_images(unet_model, val_generator, epoch, OUTPUT_DIR)
        
        current_epoch_reward = log_entry['reward']
        if current_epoch_reward > best_val_reward:
            print(f"New best reward: {current_epoch_reward:.4f} (previously {best_val_reward:.4f}). Saving best models.")
            best_val_reward = current_epoch_reward
            epochs_without_improvement = 0
            unet_model.save_weights(os.path.join(MODELS_DIR, "unet_best.weights.h5"))
            ddpg_agent.actor.save_weights(os.path.join(MODELS_DIR, "actor_best.weights.h5"))
            ddpg_agent.critic.save_weights(os.path.join(MODELS_DIR, "critic_best.weights.h5"))
        else:
            epochs_without_improvement += 1
            print(f"No improvement in reward for {epochs_without_improvement} epoch(s). Best was {best_val_reward:.4f}.")
        
        if epochs_without_improvement > 0 and epochs_without_improvement % args.lr_scheduler_patience == 0:
            current_lr = unet_optimizer.learning_rate.numpy()
            if current_lr > args.min_lr:
                new_lr = current_lr * args.lr_scheduler_factor
                ddpg_agent.actor_optimizer.learning_rate.assign(new_lr)
                ddpg_agent.critic_optimizer.learning_rate.assign(new_lr)
                unet_optimizer.learning_rate.assign(new_lr)
                print(f"Reduced learning rate to {new_lr:.7f}.")
        
        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"Stopping training early after {epochs_without_improvement} epochs without improvement.")
            break

    print("\n--- Training Finished ---")
    print("Saving final models and training history...")
    unet_model.save(os.path.join(MODELS_DIR, "unet_final.keras"))
    ddpg_agent.actor.save_weights(os.path.join(MODELS_DIR, "actor_final.weights.h5"))
    ddpg_agent.critic.save_weights(os.path.join(MODELS_DIR, "critic_final.weights.h5"))

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(OUTPUT_DIR, "training_history.csv"), index_label="epoch")
    print(f"All outputs saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a 3D U-Net with RL-based hyperparameter tuning.")
    
    # --- I/O Arguments ---
    parser.add_argument("--tiff_path", type=str, required=True, help="Path to the input (multi-page) TIFF file.")
    parser.add_argument("--spots_csv_path", type=str, required=True, help="Path to the TrackMate spots CSV file.")
    parser.add_argument("--base_output_path", type=str, required=True, help="Base directory to save all training runs.")
    
    # --- CSV Column Names ---
    parser.add_argument("--csv_frame_col", type=str, default="FRAME", help="Column name for frame index in CSV.")
    parser.add_argument("--csv_x_col", type=str, default="POSITION_X", help="Column name for spot X coordinate.")
    parser.add_argument("--csv_y_col", type=str, default="POSITION_Y", help="Column name for spot Y coordinate.")
    parser.add_argument("--csv_snr_col", type=str, default="SNR_CH1", help="Column name for spot SNR.")
    parser.add_argument("--csv_quality_col", type=str, default="QUALITY", help="Column name for spot Quality.")
    
    # --- Data & Model Parameters ---
    parser.add_argument("--sequence_length", type=int, default=5, help="Number of frames in each sequence (must be odd).")
    parser.add_argument("--img_height", type=int, default=256, help="Image height (must match data).")
    parser.add_argument("--img_width", type=int, default=256, help="Image width (must match data).")
    parser.add_argument("--channels", type=int, default=1, help="Number of channels (default 1).")
    parser.add_argument("--train_split", type=float, default=0.8, help="Fraction of data for training.")
    parser.add_argument("--val_split", type=float, default=0.2, help="Fraction of data for validation.")
    
    # --- Training Hyperparameters ---
    parser.add_argument("--total_epochs", type=int, default=100, help="Total number of epochs to run *after* warmup.")
    parser.add_argument("--steps_per_epoch", type=int, default=100, help="Number of training steps per epoch.")
    parser.add_argument("--unet_batch_size", type=int, default=4, help="Batch size for the U-Net.")
    parser.add_argument("--rl_batch_size", type=int, default=32, help="Batch size for the DDPG agent's replay buffer.")
    parser.add_argument("--buffer_capacity", type=int, default=20000, help="Capacity of the DDPG replay buffer.")
    parser.add_argument("--unet_lr", type=float, default=1e-4, help="Learning rate for the U-Net optimizer.")
    parser.add_argument("--actor_lr", type=float, default=1e-4, help="Learning rate for the DDPG Actor.")
    parser.add_argument("--critic_lr", type=float, default=3e-4, help="Learning rate for the DDPG Critic.")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor for the DDPG agent.")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update parameter for target networks.")
    parser.add_argument("--action_noise_stddev_fraction", type=float, default=0.05, help="Std dev of action noise as a fraction of action range.")
    parser.add_argument("--lambda_geo_bounds", type=float, nargs=2, default=[0.01, 0.5], help="Min and max bounds for the lambda_geo action.")
    
    # --- RL State & Reward ---
    parser.add_argument("--state_dim", type=int, default=5, help="Dimension of the RL state vector.")
    parser.add_argument("--action_dim", type=int, default=1, help="Dimension of the RL action vector (lambda_geo).")
    parser.add_argument("--signal_radius", type=int, default=2, help="Radius of circle for spot signal (SNR reward).")
    parser.add_argument("--bg_inner_radius", type=int, default=4, help="Inner radius of background ring (SNR reward).")
    parser.add_argument("--bg_outer_radius", type=int, default=6, help="Outer radius of background ring (SNR reward).")

    # --- Noise Model ---
    parser.add_argument("--noise_decay", type=float, default=0.999, help="Factor to decay action noise by each epoch.")
    
    # --- Training Control ---
    parser.add_argument("--rl_warmup_epochs", type=int, default=5, help="Number of epochs to run with random actions to fill buffer.")
    parser.add_argument("--early_stopping_patience", type=int, default=15, help="Patience (epochs) for Early Stopping.")
    parser.add_argument("--lr_scheduler_patience", type=int, default=7, help="Patience (epochs) for ReduceLROnPlateau.")
    parser.add_argument("--lr_scheduler_factor", type=float, default=0.5, help="Factor to reduce LR by.")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate.")
    
    # --- Other ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    # --- Post-process args to reconstruct nested dictionaries ---
    args.data_split = {"train": args.train_split, "val": args.val_split}
    
    # This is the active reward config used by calculate_snr_reward
    args.rl_reward_config = {
        'signal_radius': args.signal_radius,
        'bg_inner_radius': args.bg_inner_radius,
        'bg_outer_radius': args.bg_outer_radius
    }
    
    # Hard-coded noise regions
    args.noise_analysis_regions = [
        (0, 190, 50, 250), (200, 190, 250, 250),
        (0, 10, 40, 50), (220, 10, 250, 50)
    ]

    # --- Validate arguments ---
    if args.sequence_length % 2 == 0:
        parser.error("--sequence_length must be an odd number (e.g., 3, 5).")
    if args.train_split + args.val_split > 1.0:
        parser.error("--train_split and --val_split must sum to 1.0 or less.")
    
    # --- Create output directory ---
    Path(args.base_output_path).mkdir(parents=True, exist_ok=True)

    main(args)
