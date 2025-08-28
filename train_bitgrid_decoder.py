import os
import numpy as np
import torch
from datasets import load_dataset
from bitgrid import BitGridSensor, image_to_bitgrid, phase_reconstruction
from neural_network import NeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import joblib
import argparse


def build_dataset(model, sensor, ds, n_samples=2000, grid=16, device='cpu'):
    X_feats = []
    Y_bits = []
    device_t = torch.device(device)
    model.to(device_t)
    model.eval()

    count = 0
    for item in tqdm(ds, desc='Building dataset'):
        if count >= n_samples:
            break
        try:
            pil = item['img']
            arr = np.asarray(pil.convert('L').resize((128,128)), dtype=np.float32) / 255.0
            gt_bits = image_to_bitgrid(arr, g=grid, threshold=0.5)
            emb = sensor.encode_image(arr)
            if isinstance(emb, torch.Tensor):
                emb_t = emb.unsqueeze(0).to(device_t)
            else:
                emb_t = torch.tensor(emb).unsqueeze(0).to(device_t)
            a = torch.full((1,1), 0.5, dtype=torch.float32).to(device_t)
            emo = torch.zeros(1, dtype=torch.long).to(device_t)
            with torch.no_grad():
                _, phase = model(emb_t, a, emo, emo, epoch=None)
            if isinstance(phase, torch.Tensor):
                ph = phase.squeeze(0).cpu().numpy()
            else:
                ph = np.asarray(phase)
            recon_norm, recon_smoothed, threshold = phase_reconstruction(ph, g=grid)
            feat = recon_norm.flatten().astype(np.float32)
            X_feats.append(feat)
            Y_bits.append(gt_bits.astype(np.int8))
            count += 1
        except Exception:
            continue
    X = np.stack(X_feats, axis=0)
    Y = np.stack(Y_bits, axis=0)
    return X, Y


def train_decoder(ckpt='pokrov_model.pt', n_samples=2000, grid=16, device='cpu', out_path='bitgrid_decoder.joblib'):
    if not os.path.exists(ckpt):
        print(f'Checkpoint not found: {ckpt}')
        return
    data = torch.load(ckpt, map_location='cpu')
    ms = data.get('model_state', data)
    input_size = data.get('input_size', 128)

    model = NeuralNetwork(input_size, 32, 32, 20)
    try:
        model.load_state_dict(ms)
    except Exception:
        model_state = {k.replace('module.', ''): v for k, v in ms.items()}
        model.load_state_dict(model_state)

    sensor = BitGridSensor(dim=input_size, grid=grid, threshold=0.5)

    ds = load_dataset('cifar10', split='train')
    n_samples = min(n_samples, len(ds))
    ds_small = ds.select(list(range(n_samples)))

    X, Y = build_dataset(model, sensor, ds_small, n_samples=n_samples, grid=grid, device=device)
    print(f'Built dataset X={X.shape}, Y={Y.shape}')

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    base = LogisticRegression(solver='liblinear', max_iter=200)
    clf = MultiOutputClassifier(base, n_jobs=-1)
    print('Training LogisticRegression decoder...')
    clf.fit(X_train, Y_train)

    Yp = clf.predict(X_val)
    bit_acc = accuracy_score(Y_val.flatten(), Yp.flatten())

    def iou_bits(a, b):
        a = a.astype(bool)
        b = b.astype(bool)
        inter = np.logical_and(a, b).sum(axis=1)
        union = np.logical_or(a, b).sum(axis=1)
        iou = np.where(union == 0, 1.0, inter / union)
        return iou

    ious = iou_bits(Y_val, Yp)
    mean_iou = float(np.mean(ious))

    print(f'Decoder results on val: mean IoU={mean_iou:.4f}, bit_acc={bit_acc:.4f}, samples={Y_val.shape[0]}')

    joblib.dump({'clf': clf, 'grid': grid}, out_path)
    print(f'Saved decoder to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='pokrov_model.pt')
    parser.add_argument('--n-samples', type=int, default=2000)
    parser.add_argument('--grid', type=int, default=16)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--out', type=str, default='bitgrid_decoder.joblib')
    args = parser.parse_args()
    train_decoder(ckpt=args.ckpt, n_samples=args.n_samples, grid=args.grid, device=args.device, out_path=args.out)
