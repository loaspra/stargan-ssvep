import os
import random
import shutil
import numpy as np
import argparse
from munch import Munch
from torch.backends import cudnn
import torch
from core.utils2 import generate_ref_signal, fbcca
from core.utils import segment_and_filter_all_subjects
from core.data_loader import get_train_loader, get_test_loader
from core.solver import Solver
import scipy.signal as signal

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Batch run script')
parser.add_argument('--skip-data', action='store_true', help='Skip data processing steps')
args = parser.parse_args()

WINDOW_SIZES = [1024]
w_sizes = ["4"]

for window_size in WINDOW_SIZES:
    print(f"Processing window size: {window_size}")
    # Constants
    SRC_DIR = './data/processed'
    TARGET_DIR = './data/final'
    LOWCUT = 6
    HIGHCUT = 54
    FS = 250
    ORDER = 6
    N = window_size
    N_HARMONICS = 3
    N_SUBBANDS = 6
    FREQS = [8, 9, 10, 11, 12, 13, 14, 15]
    LOWEST_FREQ = 2
    UPMOST_FREQ = 54
    W = 2.2
    # IDX_FREQS = [0, 2, 4, 6]  # [8, 10, 12, 14]
    IDX_FREQS = [0, 1, 2, 3, 4, 5, 6, 7]  # [8, 10, 12, 14]
    DATA_DIR = "./data/final"
    DATA_SRC = "./data/"
    FREQ_PHASE_SRC = "./data/raw/Freq_Phase.mat"
    SPLIT_RATIO = 0.9  # for train/validation

    if not args.skip_data:
        # Step 0: Purge the data directory and target_dir
        shutil.rmtree(DATA_DIR, ignore_errors=True)
        shutil.rmtree(TARGET_DIR, ignore_errors=True)
        shutil.rmtree(os.path.join(DATA_SRC, "train"), ignore_errors=True)
        shutil.rmtree(os.path.join(DATA_SRC, "val"), ignore_errors=True)

        # Step 1: Segment and filter all subjects
        os.makedirs(TARGET_DIR, exist_ok=True)
        segment_and_filter_all_subjects(SRC_DIR, TARGET_DIR, window_size, LOWCUT, HIGHCUT, FS, ORDER)

        # Step 2: Train-Test split and calculate accuracies
        ref_signals = generate_ref_signal(FREQ_PHASE_SRC, freqs=FREQS, N=N, n_harmonics=N_HARMONICS, fs=FS)
        accuracies = np.zeros((len(FREQS), 35, 6)) # 35 subjects, 6 trials

        for i in range(1, 36):
            for idx_freq, (freq_name, freq) in enumerate(zip(FREQS, IDX_FREQS)):
                actual_freq = IDX_FREQS.index(freq)
                label_dir = os.path.join(DATA_DIR, str(freq))
                for j in range(0, 5):
                    file_path = os.path.join(label_dir, f"S{i}_{j}.npy")
                    if os.path.exists(file_path):
                        try:
                            segment = np.load(file_path)
                        except FileNotFoundError:
                            print(f"File not found: {file_path}")
                            continue
                        dominant_freqs = []
                        for channel in segment:
                            # print(f"channel.shape: {channel.shape}")
                            freqs, psd = signal.welch(channel, fs=FS, nperseg=250)
                            dominant_freq = freqs[np.argmax(psd)]
                            # print(f"FREQ: {freq_name}, dominant_freq: {dominant_freq}, np.argmax(psd): {np.argmax(psd)}")
                            dominant_freqs.append(dominant_freq)
                        if dominant_freqs.count(freq_name) > 1:
                            accuracies[idx_freq, i-1, j] = 1
            # break
        # print(accuracies)
        # exit(0)

        # acc_means = np.mean(accuracies, axis=1)
        # perfect_idx = np.where(acc_means == 1)[0]
        
        # if len(perfect_idx) == 0:
        #     sorted_indices = np.argsort(acc_means)[::-1]
        #     top_50_percent_idx = sorted_indices[:len(sorted_indices) // 2]
        #     perfect_idx = top_50_percent_idx
        
        # print(perfect_idx)

        # Split the data
        labels = os.listdir(TARGET_DIR)
        print(f"Labels: {labels}")
        for i, label in enumerate(labels):
            label_dir = os.path.join(TARGET_DIR, label)
            files = os.listdir(label_dir)
            files = [file for file in files if accuracies[i, int(file.split("_")[0][1:]) - 1, int(file.split("_")[1][0:1])] == 1]
            print(f"Using a total of {len(files)} samples to process for label {label}")
            random.shuffle(files)
            split_idx = int(len(files) * SPLIT_RATIO)
            train_files = files[:split_idx]
            val_files = files[split_idx:]
            train_dir = os.path.join(DATA_SRC, "train", label)
            print(f"Creating directory: {train_dir}")
            val_dir = os.path.join(DATA_SRC, "val", label)
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)

            for file in train_files:
                shutil.copy(os.path.join(label_dir, file), os.path.join(train_dir, file))
            for file in val_files:
                shutil.copy(os.path.join(label_dir, file), os.path.join(val_dir, file))

        # Rename the labels folders to 0, 1, 2 ... N instead of 2, 4, 6, 8
        train_files = sorted(os.listdir("data/train/"))
        val_files = sorted(os.listdir("data/val/"))
        for file in train_files:
            shutil.move(f"data/train/{file}", f"data/train/{train_files.index(file)}")
        for file in val_files:
            shutil.move(f"data/val/{file}", f"data/val/{val_files.index(file)}")

        # # from the DATA_SRC + train/val folders, remove the folders that does not correspond to the sequence (0, 1, 2, 3)
        # for folder in os.listdir("data/train/"):
        #     if folder not in [str(i) for i in range(4)]:
        #         shutil.rmtree(f"data/train/{folder}")
        # for folder in os.listdir("data/val/"):
        #     if folder not in [str(i) for i in range(4)]:
        #         shutil.rmtree(f"data/val/{folder}")

    # Final step: Train the model
    def subdirs(dname):
        return [d for d in os.listdir(dname) if os.path.isdir(os.path.join(dname, d))]

    class Args:
        img_size = window_size
        num_domains = 8
        latent_dim = 16
        hidden_dim = 512
        style_dim = 64
        lambda_reg = 1
        lambda_cyc = 1
        lambda_sty = 1
        lambda_ds = 1
        ds_iter = 5000
        w_hpf = 0  # For SSVEP
        randcrop_prob = 0.5
        total_iters = 5000
        resume_iter = 0
        batch_size = 8
        val_batch_size = 8
        lr = 5e-4
        f_lr = 1e-6
        beta1 = 0.0
        beta2 = 0.99
        weight_decay = 1e-4
        num_outs_per_domain = 10
        mode = 'train'
        num_workers = 16
        seed = 777
        train_img_dir = 'data/train'
        val_img_dir = 'data/val'
        sample_dir = 'expr/samples'
        checkpoint_dir = 'expr/checkpoints'
        eval_dir = 'expr/eval'
        result_dir = 'expr/results'
        print_every = 10
        sample_every = 250
        save_every = 1000
        eval_every = 2500
        runType = f'w_{w_sizes[WINDOW_SIZES.index(window_size)]}s'

    args = Args()
    args.skip_data = False  # Add this line to initialize skip_data attribute

    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    solver = Solver(args)

    assert len(subdirs(args.train_img_dir)) == args.num_domains
    assert len(subdirs(args.val_img_dir)) == args.num_domains

    loaders = Munch(
        src=get_train_loader(root=args.train_img_dir, which='source', img_size=args.img_size, batch_size=args.batch_size, prob=args.randcrop_prob, num_workers=args.num_workers),
        ref=get_train_loader(root=args.train_img_dir, which='reference', img_size=args.img_size, batch_size=args.batch_size, prob=args.randcrop_prob, num_workers=args.num_workers),
        val=get_test_loader(root=args.val_img_dir, img_size=args.img_size, batch_size=args.val_batch_size, shuffle=True, num_workers=args.num_workers)
    )

    solver.train(loaders)