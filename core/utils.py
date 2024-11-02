"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
from os.path import join as ospj
import json
import glob
from shutil import copyfile
from matplotlib import pyplot as plt

from tqdm import tqdm
import ffmpeg

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils

import scipy.io as sio
from scipy import signal
from scipy.signal import butter, lfilter
import pandas as pd


# from torchviz import make_dot


def save_json(json_file, filename):
    with open(filename, 'w') as f:
        json.dump(json_file, f, indent=4, sort_keys=False)


def print_network(network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    # instead of print(network), print the number of parameters
    print("Number of parameters of %s: %i" % (name, num_params))
    


def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def get_FFT(x):
    fft = abs(np.fft.fft(x))
    # calculate the frequencies
    freqs = np.fft.fftfreq(len(x)) * 250
    return fft, freqs

def save_image(x, ncol, filename):
    # print([item.shape for item in x])
    x = [item.cpu() for item in x]
    np.save(filename + f"_b", x)
    # ss = int(np.ceil(np.sqrt(len(x))))

    # fig, axs = plt.subplots(ss, ss, figsize=(10, 10))

    # for i in range(len(x)):
    #     ex = i // ss
    #     ne = i % ss
    #     for j in range(x[i].shape[0]):
    #         x_ = x[i][j, :]
    #         # plot the fft of the signal
    #         fft, freqs = get_FFT(x_)
    #         axs[ex, ne].plot(freqs, fft)

    #     axs[ex, ne].set_title(f"FFT of signal {str(i)}")
    #     axs[ex, ne].set_xlabel("Frequency")
    #     axs[ex, ne].set_ylabel("Amplitude")
    #     axs[ex, ne].set_xlim(0, 22)

    # plt.xlabel("Frequency")
    # plt.ylabel("Amplitude")
    # plt.xlim(0, 22)
    # plt.savefig(filename + ".png")
    # plt.close()    



@torch.no_grad()
def translate_and_reconstruct(nets, args, x_src, y_src, x_ref, y_ref, filename):
    N, _, _ = x_src.size()
    s_ref = nets.style_encoder(x_ref, y_ref)
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    x_fake = nets.generator(x_src, s_ref, masks=masks)
    s_src = nets.style_encoder(x_src, y_src)
    masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    x_rec = nets.generator(x_fake, s_src, masks=masks)
    x_concat = [x_src, x_ref, x_fake, x_rec]
    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N, filename)
    del x_concat


@torch.no_grad()
def translate_using_latent(nets, args, x_src, y_trg_list, z_trg_list, psi, filename):
    N, _, _ = x_src.size()
    latent_dim = z_trg_list[0].size(1)
    x_concat = [x_src]
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None

    for i, y_trg in enumerate(y_trg_list):
        z_many = torch.randn(10000, latent_dim).to(x_src.device)
        y_many = torch.LongTensor(10000).to(x_src.device).fill_(y_trg[0])
        s_many = nets.mapping_network(z_many, y_many)
        s_avg = torch.mean(s_many, dim=0, keepdim=True)
        s_avg = s_avg.repeat(N, 1)

        for z_trg in z_trg_list:
            s_trg = nets.mapping_network(z_trg, y_trg)
            s_trg = torch.lerp(s_avg, s_trg, psi)
            x_fake = nets.generator(x_src, s_trg, masks=masks)
            x_concat += [x_fake]

    x_concat = torch.cat(x_concat, dim=0)
    save_image(x_concat, N, filename)


@torch.no_grad()
def translate_using_reference(nets, args, x_src, x_ref, y_ref, filename):
    # print(f"Shape of source image: {x_src.shape}")
    N, C, H = x_src.size()
    wb = torch.ones(1, C, H).to(x_src.device)
    x_src_with_wb = torch.cat([wb, x_src], dim=0)

    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    s_ref = nets.style_encoder(x_ref, y_ref)

    # print(f"Shape of reference image: {x_ref.shape}")
    # # Label
    # print(f"Shape of reference label: {y_ref.shape}, value: {y_ref}")

    # make_dot(nets.style_encoder(x_ref, y_ref)).render("style_encoder", format="png")
    
    s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1)
    print(f"Shape of reference style: {s_ref_list.shape}")

    # print(f"Into de torch: {x_ref.shape} ==> {x_ref}")

    x_concat = [x_src_with_wb]
    for i, s_ref in enumerate(s_ref_list):
        
        x_fake = nets.generator(x_src, s_ref, masks=masks)
        x_fake_with_ref = torch.cat([x_ref[i:i+1], x_fake], dim=0)
        x_concat += [x_fake_with_ref]
        
    print(f"saving into {filename + '_%02d' % i}")  
    save_image(x_concat, N+1, filename + '_%02d' % i)

    del x_concat


@torch.no_grad()
def debug_image(nets, args, inputs, step):
    x_src, y_src = inputs.x_src, inputs.y_src
    x_ref, y_ref = inputs.x_ref, inputs.y_ref

    device = inputs.x_src.device
    N = inputs.x_src.size(0)

    # translate and reconstruct (reference-guided)
    filename = ospj(args.sample_dir, '%06d_cycle_consistency.jpg' % (step))
    translate_and_reconstruct(nets, args, x_src, y_src, x_ref, y_ref, filename)

    # latent-guided image synthesis
    y_trg_list = [torch.tensor(y).repeat(N).to(device)
                  for y in range(min(args.num_domains, 5))]
    z_trg_list = torch.randn(args.num_outs_per_domain, 1, args.latent_dim).repeat(1, N, 1).to(device)
    for psi in [0.5, 0.7, 1.0]:
        filename = ospj(args.sample_dir, '%06d_latent_psi_%.1f.jpg' % (step, psi))
        translate_using_latent(nets, args, x_src, y_trg_list, z_trg_list, psi, filename)

    # reference-guided image synthesis
    filename = ospj(args.sample_dir, '%06d_reference.jpg' % (step))
    translate_using_reference(nets, args, x_src, x_ref, y_ref, filename)


# ======================= #
# Video-related functions #
# ======================= #


def sigmoid(x, w=1):
    return 1. / (1 + np.exp(-w * x))


def get_alphas(start=-5, end=5, step=0.5, len_tail=10):
    return [0] + [sigmoid(alpha) for alpha in np.arange(start, end, step)] + [1] * len_tail


def interpolate(nets, args, x_src, s_prev, s_next):
    ''' returns T x C x H x W '''
    B = x_src.size(0)
    frames = []
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    alphas = get_alphas()

    for alpha in alphas:
        s_ref = torch.lerp(s_prev, s_next, alpha)
        x_fake = nets.generator(x_src, s_ref, masks=masks)
        entries = torch.cat([x_src.cpu(), x_fake.cpu()], dim=2)
        frame = torchvision.utils.make_grid(entries, nrow=B, padding=0, pad_value=-1).unsqueeze(0)
        frames.append(frame)
    frames = torch.cat(frames)
    return frames


def slide(entries, margin=32):
    """Returns a sliding reference window.
    Args:
        entries: a list containing two reference images, x_prev and x_next, 
                 both of which has a shape (1, 3, 256, 256)
    Returns:
        canvas: output slide of shape (num_frames, 3, 256*2, 256+margin)
    """
    _, C, H = entries[0].shape
    alphas = get_alphas()
    T = len(alphas) # number of frames

    canvas = - torch.ones((T, C, H*2))
    merged = torch.cat(entries, dim=2)  # (1, 3, 512, 256)
    for t, alpha in enumerate(alphas):
        top = int(H * (1 - alpha))  # top, bottom for canvas
        bottom = H * 2
        m_top = 0  # top, bottom for merged
        m_bottom = 2 * H - top
        canvas[t, :, top:bottom] = merged[:, :, m_top:m_bottom]
    return canvas


@torch.no_grad()
def video_ref(nets, args, x_src, x_ref, y_ref, fname):
    video = []
    s_ref = nets.style_encoder(x_ref, y_ref)
    s_prev = None
    for data_next in tqdm(zip(x_ref, y_ref, s_ref), 'video_ref', len(x_ref)):
        x_next, y_next, s_next = [d.unsqueeze(0) for d in data_next]
        if s_prev is None:
            x_prev, y_prev, s_prev = x_next, y_next, s_next
            continue
        if y_prev != y_next:
            x_prev, y_prev, s_prev = x_next, y_next, s_next
            continue

        interpolated = interpolate(nets, args, x_src, s_prev, s_next)
        entries = [x_prev, x_next]
        slided = slide(entries)  # (T, C, 256*2)
        frames = torch.cat([slided, interpolated], dim=2).cpu()  # (T, C, 256*(batch+1))
        video.append(frames)
        x_prev, y_prev, s_prev = x_next, y_next, s_next

    # append last frame 10 time
    for _ in range(10):
        video.append(frames[-1:])
    video = tensor2ndarray255(torch.cat(video))
    save_video(fname, video)


@torch.no_grad()
def video_latent(nets, args, x_src, y_list, z_list, psi, fname):
    latent_dim = z_list[0].size(1)
    s_list = []
    for i, y_trg in enumerate(y_list):
        z_many = torch.randn(10000, latent_dim).to(x_src.device)
        y_many = torch.LongTensor(10000).to(x_src.device).fill_(y_trg[0])
        s_many = nets.mapping_network(z_many, y_many)
        s_avg = torch.mean(s_many, dim=0, keepdim=True)
        s_avg = s_avg.repeat(x_src.size(0), 1)

        for z_trg in z_list:
            s_trg = nets.mapping_network(z_trg, y_trg)
            s_trg = torch.lerp(s_avg, s_trg, psi)
            s_list.append(s_trg)

    s_prev = None
    video = []
    # fetch reference images
    for idx_ref, s_next in enumerate(tqdm(s_list, 'video_latent', len(s_list))):
        if s_prev is None:
            s_prev = s_next
            continue
        if idx_ref % len(z_list) == 0:
            s_prev = s_next
            continue
        frames = interpolate(nets, args, x_src, s_prev, s_next).cpu()
        video.append(frames)
        s_prev = s_next
    for _ in range(10):
        video.append(frames[-1:])
    video = tensor2ndarray255(torch.cat(video))
    save_video(fname, video)


def save_video(fname, images, output_fps=30, vcodec='libx264', filters=''):
    assert isinstance(images, np.ndarray), "images should be np.array: NHWC"
    num_frames, height, width, channels = images.shape
    stream = ffmpeg.input('pipe:', format='rawvideo', 
                          pix_fmt='rgb24', s='{}x{}'.format(width, height))
    stream = ffmpeg.filter(stream, 'setpts', '2*PTS')  # 2*PTS is for slower playback
    stream = ffmpeg.output(stream, fname, pix_fmt='yuv420p', vcodec=vcodec, r=output_fps)
    stream = ffmpeg.overwrite_output(stream)
    process = ffmpeg.run_async(stream, pipe_stdin=True)
    for frame in tqdm(images, desc='writing video to %s' % fname):
        process.stdin.write(frame.astype(np.uint8).tobytes())
    process.stdin.close()
    process.wait()


def tensor2ndarray255(images):
    images = torch.clamp(images * 0.5 + 0.5, 0, 1)
    return images.cpu().numpy().transpose(0, 2, 3, 1) * 255


"""
# ======================== #
# Signal-related functions #
# ======================== #
"""
def read_mat_file(file_path):
    data = sio.loadmat(file_path)
    return data


def read_mat_file_as_df(file_path, columns):
    data = read_mat_file(file_path)
    data_values = data['data']
    # shape --> (64, 1500, 40, 6)
    data_values = data_values[:, 125:(1250 + 125), :, :]
    data_values = data_values.reshape((64, 1250*40*6), order='F')
    data_labels = np.tile(np.arange(0, 40), 6).repeat(1250)
    data_values = data_values.swapaxes(0, 1)
    data_df = pd.DataFrame(data_values)
    data_df.columns = ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "M1", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "M2", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "CB1"
                       , "O1", "Oz", "O2", "CB2"]
    data_df['labels'] = data_labels

    if columns is not None:
        data_df = data_df[columns + ['labels']]
    return data_df


def read_all_mat_files_and_save_as_csv(columns=None, labels = None):
    for i in range(1, 36):
        file_path = "./data/raw/S" + str(i) + ".mat"
        data_df = read_mat_file_as_df(file_path, columns)

        if labels is not None:
            # Return only the data with a label in the labels list
            data_df = data_df[data_df['labels'].isin(labels)]
            
        data_df.to_csv("./data/processed/S" + str(i) + ".csv", index=False)



def segment_signal_into_windows(df, window_size, fs = 250, padding = 140, n_blocks = 6, block_size = 1250):
    # Get the n_samples of padding (which comes in ms)
    sp = int(padding * fs / 1000)
    N = df.shape[0]
    dim = df.shape[1]
    # K = int(N/window_size)
    segments = np.empty((n_blocks, window_size, dim))
    
    for i in range(n_blocks):
        # print((i*block_size+sp), (i*block_size+window_size+sp))
        segment = df[(i*block_size+sp):(i*block_size+window_size+sp),:]
        segments[i] = np.vstack(segment)
    return segments


def segment_signal_into_overlapping_windows(df, window_size, overlap):
    N = df.shape[0]
    dim = df.shape[1]
    K = int(N/window_size)
    segments = np.empty((K, window_size, dim))
    for i in range(K):
        segment = df[i*overlap : i*overlap+window_size,:]
        segments[i] = np.vstack(segment)
    return segments


# def butter_bandpass(lowcut, highcut, fs, order=6):
#     nyq = 0.5 * fs
#     low = float(lowcut) / nyq
#     high = float(highcut) / nyq
#     b, a = butter(order, [low, high], btype='band', analog=False)
#     return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = float(lowcut) / nyq
    high = float(highcut) / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    y = lfilter(b, a, data, axis=0)
    return y


def apply_filter_to_segments(segments, lowcut, highcut, fs, order=6, segment_dim=1250):
    N = segments.shape[0]
    dim = segments.shape[2]
    filtered_segments = np.empty((N, segment_dim, dim))
    for i in range(N):
        segment = segments[i]
        filtered_segments[i] = butter_bandpass_filter(segment, lowcut, highcut, fs, order=order)
    return filtered_segments


def apply_filter_to_signal(signal, lowcut, highcut, fs, order=6):
    dim = signal.shape[1]
    filtered_signal = np.empty((signal.shape[0], dim))
    filtered_signal = butter_bandpass_filter(signal, lowcut, highcut, fs, order=order)
    return filtered_signal


def apply_filter_to_df(df, lowcut, highcut, fs, order=6):
    dim = df.shape[1]
    filtered_df = np.empty((df.shape[0], dim))
    filtered_df = butter_bandpass_filter(df, lowcut, highcut, fs, order=order)
    return filtered_df


def apply_filter_to_df_with_labels(df, lowcut, highcut, fs, order=6):
    dim = df.shape[1]
    filtered_df = np.empty((df.shape[0], dim))
    filtered_df = butter_bandpass_filter(df, lowcut, highcut, fs, order=order)
    filtered_df['labels'] = df['labels']
    return filtered_df


def segment_and_filter_all_subjects(src_dir, target_dir, window_size, lowcut, highcut, fs, order=6):
    # save the segments on the target dir / label name
    df = pd.read_csv(os.path.join(src_dir, "S1.csv"))
    labels = df['labels'].unique()
    for label in labels:
        label_dir = os.path.join(target_dir, str(label))
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        
        for i in range(1, 36):
            file_path = os.path.join(src_dir, "S" + str(i) + ".csv")
            df = pd.read_csv(file_path)
            df = df[df['labels'] == label]
            segments = segment_signal_into_windows(df.drop('labels', axis = 1).values, window_size)
            filtered_segments = apply_filter_to_segments(segments, lowcut, highcut, fs, order=order, segment_dim=window_size)
            for j in range(filtered_segments.shape[0]):
                segment = filtered_segments[j].swapaxes(0, 1)

                if segment.shape[1] == window_size:
                    np.save(os.path.join(label_dir, "S" + str(i) + "_" + str(j) + ".npy"), segment)
                else:
                    print("segment with shape " + str(segment.shape) + " not saved")
