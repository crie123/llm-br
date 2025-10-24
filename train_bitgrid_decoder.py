import os
import argparse
import joblib
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

from bitgrid import BitGridDecoder, BitGridSensor, image_to_bitgrid, bitgrid_to_phase
from datasets import load_dataset


def create_synthetic_dataset(n_samples, grid_size=16, noise_level=0.1):
    X = np.random.rand(n_samples, grid_size * grid_size)
    Y = (X + np.random.normal(0, noise_level, X.shape) > 0.5).astype(float)
    return X, Y


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=1) + self.eps
        den = probs.sum(dim=1) + targets.sum(dim=1) + self.eps
        dice = num / den
        return 1.0 - dice.mean()


class MLPDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=(512, 256), output_dim=256, drop=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(drop))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def extract_features_from_cifar(n_samples=None, grid=16, dim=128):
    """Extract features from CIFAR10 dataset using BitGridSensor."""
    ds = load_dataset('cifar10', split='train')
    total = len(ds) if n_samples is None else min(n_samples, len(ds))
    sensor = BitGridSensor(dim=dim, grid=grid)

    feats = []
    targets = []

    pbar = tqdm(range(total), desc='Extracting features')
    for i in pbar:
        item = ds[i]
        pil = item['img']
        arr = np.asarray(pil.convert('L').resize((dim, dim)), dtype=np.float32) / 255.0
        # bitgrid (target)
        bits = image_to_bitgrid(arr, g=grid, threshold=0.5)
        bits_flat = bits.reshape(-1).astype(np.float32)

        # phase embedding from bits (fixed length)
        phase_tensor = bitgrid_to_phase(bits_flat, dim)
        phase_vec = np.asarray(phase_tensor).ravel().astype(np.float32)

        # reconstructions and moments come from running phase_reconstruction indirectly via sensor.encode_image
        # sensor.encode_image is safe now but may run phase_reconstruction; we only need moments and recon histogram
        features = sensor.encode_image(arr)
        recon_flat = features['reconstruction'].ravel().astype(np.float32)
        moments = features['moments'].astype(np.float32)
        hist = features['hist_orig'].astype(np.float32)

        # Concatenate features: recon_flat (grid*grid) + phase_proj (dim) + moments (4) + hist (bins)
        feat_vec = np.concatenate([recon_flat, phase_vec, moments, hist])
        feats.append(feat_vec)
        targets.append(bits_flat)

    X = np.stack(feats, axis=0)
    Y = np.stack(targets, axis=0)
    return X, Y


def evaluate_decoder(model, scaler, pca, X_val, Y_val, device):
    model.eval()
    with torch.no_grad():
        X_proc = X_val.copy()
        if scaler is not None:
            X_proc = scaler.transform(X_proc)
        if pca is not None:
            X_proc = pca.transform(X_proc)
        X_t = torch.tensor(X_proc, dtype=torch.float32).to(device)
        logits = model(X_t)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(float)

    # IoU per sample
    ious = []
    for i in range(len(Y_val)):
        a = preds[i].astype(bool)
        b = Y_val[i].astype(bool)
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        if union == 0:
            iou = 1.0 if inter == 0 else 0.0
        else:
            iou = inter / union
        ious.append(iou)
    bit_acc = (preds == Y_val).mean()
    return float(np.mean(ious)), float(bit_acc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-samples', type=int, default=2000, help='Number of CIFAR train samples to use (or synthetic fallback)')
    parser.add_argument('--grid', type=int, default=16)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--pca-components', type=int, default=0, help='If >0, apply PCA to features')
    parser.add_argument('--hidden', type=str, default='512,256')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--out', type=str, default='bitgrid_decoder.pt')
    args = parser.parse_args()

    # Try to extract from CIFAR; if fails, fallback to synthetic
    try:
        X, Y = extract_features_from_cifar(args.n_samples, grid=args.grid, dim=args.dim)
    except Exception as e:
        print('CIFAR extraction failed, falling back to synthetic dataset:', e)
        X, Y = create_synthetic_dataset(args.n_samples, grid_size=args.grid, noise_level=0.1)

    print('Dataset shapes:', X.shape, Y.shape)

    # Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = None
    if args.pca_components and args.pca_components > 0:
        pca = PCA(n_components=args.pca_components)
        X_proc = pca.fit_transform(X_scaled)
    else:
        X_proc = X_scaled

    X_train, X_val, Y_train, Y_val = train_test_split(X_proc, Y, test_size=0.2, random_state=42)

    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]
    hidden_dims = tuple(int(h) for h in args.hidden.split(',')) if args.hidden else (512, 256)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLPDecoder(input_dim, hidden_dims=hidden_dims, output_dim=output_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()

    best_iou = -1.0
    best_state = None

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = bce(logits, yb) + 0.5 * dice(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_loader.dataset)

        mean_iou, bit_acc = evaluate_decoder(model, scaler, pca, X_val, Y_val, device)
        print(f'Epoch {epoch}/{args.epochs} loss={epoch_loss:.6f} val_iou={mean_iou:.4f} val_bit_acc={bit_acc:.4f}')

        # checkpoint best
        if mean_iou > best_iou:
            best_iou = mean_iou
            best_state = model.state_dict()

    if best_state is not None:
        torch.save({'model_state': best_state, 'input_dim': input_dim, 'output_dim': output_dim, 'hidden': hidden_dims}, args.out)
        # save preprocessing
        prep = {'scaler': scaler, 'pca': pca, 'grid': args.grid, 'dim': args.dim}
        joblib.dump(prep, 'bitgrid_decoder_meta.joblib')
        print('Saved decoder to', args.out)
        print('Saved preprocessing to bitgrid_decoder_meta.joblib')

    # final evaluation
    if best_state is not None:
        model.load_state_dict(best_state)
        final_iou, final_acc = evaluate_decoder(model, scaler, pca, X_val, Y_val, device)
        print(f'Final best val IoU={final_iou:.4f}, bit_acc={final_acc:.4f}')


if __name__ == '__main__':
    main()
