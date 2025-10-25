import os
import numpy as np
import torch
from datasets import load_dataset
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from bitgrid import BitGridSensor, image_to_bitgrid, phase_to_bitgrid
from neural_network import NeuralNetwork
import matplotlib.pyplot as plt
import argparse
import logging
from tqdm import tqdm
from collections import defaultdict
import hashlib
import joblib
from train_bitgrid_decoder import MLPDecoder


def plot_bitgrid_metrics(ious, accs, bins: int = 20, out_prefix='bitgrid_metrics'):
    ious = np.array(ious)
    accs = np.array(accs)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.hist(ious, bins=bins, color='#4C72B0', alpha=0.8)
    plt.title('IoU Distribution')
    plt.xlabel('IoU')
    plt.ylabel('Count')

    plt.subplot(1,2,2)
    plt.hist(accs, bins=bins, color='#DD8452', alpha=0.8)
    plt.title('Bit Accuracy Distribution')
    plt.xlabel('Bit Accuracy')
    plt.ylabel('Count')

    plt.tight_layout()
    hist_path = out_prefix + '_histograms.png'
    plt.savefig(hist_path)
    plt.close()

    # Summary plot (means over samples)
    plt.figure(figsize=(6,3))
    plt.bar(['mean_iou','mean_bit_acc'], [np.mean(ious), np.mean(accs)], color=['#4C72B0','#DD8452'])
    plt.title('Bitgrid Recognition Summary')
    plt.ylabel('Value')
    plt.ylim(0,1)
    summary_path = out_prefix + '_summary.png'
    plt.savefig(summary_path)
    plt.close()

    return hist_path, summary_path


def iou_bits(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def bit_accuracy(a: np.ndarray, b: np.ndarray) -> float:
    return float((a == b).sum()) / float(a.size)


def load_checkpoint(path):
    data = torch.load(path, map_location='cpu')
    if 'model_state' in data:
        meta = data
    else:
        # older save format: assume full state dict
        meta = {'model_state': data}
    return meta


def save_example_pairs(orig_list, rec_list, out_prefix='cifar_bitgrid_examples', grid=16, max_examples=10):
    """Save side-by-side original vs reconstructed bitgrid images for the first max_examples pairs."""
    n = min(len(orig_list), len(rec_list), max_examples)
    for i in range(n):
        orig = np.asarray(orig_list[i]).reshape(grid, grid)
        rec = np.asarray(rec_list[i]).reshape(grid, grid)
        combined = np.hstack([orig, rec])
        path = f"{out_prefix}_{i}.png"
        plt.imsave(path, combined, cmap='gray', vmin=0, vmax=1)


def _ensure_flat_bits(bits, g=16):
    a = np.asarray(bits).ravel().astype(np.float32)
    total = g * g
    if a.size >= total:
        return a[:total]
    out = np.zeros(total, dtype=np.float32)
    out[:a.size] = a
    return out


def load_trained_decoder(decoder_pt='bitgrid_decoder.pt', meta_joblib='bitgrid_decoder_meta.joblib', device='cpu'):
    if not os.path.exists(decoder_pt) or not os.path.exists(meta_joblib):
        return None, None, None
    try:
        prep = joblib.load(meta_joblib)
        scaler = prep.get('scaler', None)
        pca = prep.get('pca', None)
        data = torch.load(decoder_pt, map_location='cpu')
        model_state = data.get('model_state', data)
        input_dim = data.get('input_dim', None)
        output_dim = data.get('output_dim', None)
        hidden = tuple(data.get('hidden', (512, 256)))
        if input_dim is None or output_dim is None:
            return None, scaler, pca
        model = MLPDecoder(input_dim, hidden_dims=hidden, output_dim=output_dim).to(device)
        model.load_state_dict(model_state)
        model.eval()
        return model, scaler, pca
    except Exception:
        return None, None, None


def main(ckpt='pokrov_model.pt', n_eval=200, batch_size=16, device_str='cpu'):
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    logger = logging.getLogger('eval')

    ckpt_path = ckpt
    if not os.path.exists(ckpt_path):
        logger.error('checkpoint not found: %s', ckpt_path)
        return

    meta = load_checkpoint(ckpt_path)
    ms = meta['model_state']
    input_size = meta.get('input_size', 128)
    hidden_size = meta.get('hidden_size', 32)
    output_size = meta.get('output_size', 32)
    num_layers = meta.get('num_layers', 20)

    model = NeuralNetwork(input_size, hidden_size, output_size, num_layers)
    try:
        model.load_state_dict(ms)
    except Exception:
        model_state = {k.replace('module.', ''): v for k, v in ms.items()}
        model.load_state_dict(model_state)
    model.eval()

    # decide device
    device = torch.device(device_str if (device_str.startswith('cuda') and torch.cuda.is_available()) else 'cpu')
    model.to(device)
    logger.info('Using device: %s', device)

    sensor = BitGridSensor(dim=input_size, grid=16, threshold=0.5)

    # try loading trained decoder
    decoder_model, decoder_scaler, decoder_pca = load_trained_decoder(decoder_pt='bitgrid_decoder.pt', meta_joblib='bitgrid_decoder_meta.joblib', device=device)
    if decoder_model is not None:
        logger.info('Loaded trained bitgrid decoder and will use it for reconstruction')
    else:
        logger.info('No trained decoder found, falling back to phase->bitgrid')

    ds = load_dataset('cifar10', split='test')
    n = min(len(ds), n_eval)
    ious = []
    accs = []

    # per-class metrics
    per_class_ious = defaultdict(list)
    per_class_accs = defaultdict(list)

    # collect example bit pairs for saving
    example_orig = []
    example_rec = []
    max_examples = 10

    # diagnostics: track unique reconstructed masks and phase stats
    seen_hashes = {}
    debug_phases = []

    indices = list(range(n))
    pbar = tqdm(range(0, n, batch_size), desc='Evaluating', unit='batch')

    for start in pbar:
        end = min(start + batch_size, n)
        batch_indices = indices[start:end]

        emb_list = []
        bits_orig_list = []
        feat_list = []
        labels = []
        # prepare batch
        for i in batch_indices:
            try:
                item = ds[i]
                pil = item['img']
                arr = np.asarray(pil.convert('L').resize((128,128)), dtype=np.float32) / 255.0
                bits_orig = image_to_bitgrid(arr, g=16, threshold=0.5)
                bits_orig = _ensure_flat_bits(bits_orig, g=16)
                features = sensor.encode_image(arr)
                # phase vector for model
                ph_vec = np.asarray(features['phase']).ravel()
                emb_list.append(ph_vec)
                bits_orig_list.append(bits_orig)
                # prepare decoder features: recon_flat + phase + moments + hist
                recon_flat = features['reconstruction'].ravel().astype(np.float32)
                phase_vec = ph_vec.astype(np.float32)
                moments = features['moments'].astype(np.float32)
                hist = features['hist_orig'].astype(np.float32)
                feat_vec = np.concatenate([recon_flat, phase_vec, moments, hist])
                feat_list.append(feat_vec)
                labels.append(int(item.get('label', item.get('labels', 0))))
            except Exception as e:
                logger.warning('Skipping sample %d due to error: %s', i, str(e))

        if len(emb_list) == 0:
            continue

        emb_t = torch.tensor(np.stack(emb_list, axis=0), dtype=torch.float32).to(device)
        batch_size = emb_t.size(0)
        a = torch.full((batch_size, 1), 0.5, dtype=torch.float32).to(device)
        emo = torch.zeros(batch_size, dtype=torch.long).to(device)

        with torch.no_grad():
            output, phase = model(emb_t, a, emo, emo, epoch=None)
            phs = phase.cpu().numpy() if isinstance(phase, torch.Tensor) else np.asarray(phase)

        # If decoder available, run predictions for this batch
        decoder_preds = None
        if decoder_model is not None and len(feat_list) > 0:
            try:
                X_dec = np.stack(feat_list, axis=0)
                if decoder_scaler is not None:
                    X_dec = decoder_scaler.transform(X_dec)
                if decoder_pca is not None:
                    X_dec = decoder_pca.transform(X_dec)
                with torch.no_grad():
                    X_t = torch.tensor(X_dec, dtype=torch.float32).to(device)
                    logits = decoder_model(X_t)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    decoder_preds = (probs >= 0.5).astype(np.float32)
            except Exception as e:
                logger.warning('Decoder prediction failed for batch: %s', str(e))

        # process each sample in batch
        for j in range(len(bits_orig_list)):
            bits_orig = bits_orig_list[j]
            ph = phs[j]
            try:
                if decoder_preds is not None:
                    bits_rec = _ensure_flat_bits(decoder_preds[j], g=16)
                else:
                    bits_rec = phase_to_bitgrid(ph, g=16)
            except Exception as e:
                logger.warning('phase_to_bitgrid/decoder failed for batch sample %d: %s', start + j, str(e))
                continue

            # --- diagnostics ---
            try:
                # log phase statistics
                ph_arr = np.asarray(ph)
                logger.debug('sample %d phase mean=%.6f std=%.6f', start + j, float(ph_arr.mean()), float(ph_arr.std()))

                # hash reconstructed bits to detect duplicates
                h = hashlib.sha1(bits_rec.tobytes()).hexdigest()
                seen_hashes[h] = seen_hashes.get(h, 0) + 1

                if os.environ.get('BITGRID_DEBUG') == '1' and len(debug_phases) < 50:
                    debug_phases.append({'idx': start + j, 'phase_mean': float(ph_arr.mean()), 'phase_std': float(ph_arr.std()), 'bits_sum': int(bits_rec.sum()), 'hash': h})

                # warn if this hash already seen many times
                if seen_hashes.get(h, 0) > 5:
                    logger.warning('reconstructed mask hash %s seen %d times (sample %d)', h, seen_hashes[h], start + j)
            except Exception as e:
                logger.debug('diagnostics failed: %s', str(e))

            if len(example_orig) < max_examples:
                example_orig.append(bits_orig.copy())
                example_rec.append(bits_rec.copy())

            iou = iou_bits(bits_orig, bits_rec)
            acc = bit_accuracy(bits_orig, bits_rec)
            ious.append(iou)
            accs.append(acc)

            label = labels[j]
            per_class_ious[label].append(iou)
            per_class_accs[label].append(acc)

        pbar.set_postfix({'mean_iou': np.mean(ious) if ious else 0.0})

    if len(ious) == 0:
        logger.error('No samples evaluated')
        return

    mean_iou = float(np.mean(ious))
    mean_acc = float(np.mean(accs))
    logger.info('mean IoU: %.4f, mean bit accuracy: %.4f', mean_iou, mean_acc)

    # dump diagnostics summary
    try:
        logger.info('unique reconstructed masks: %d, top hashes: %s', len(seen_hashes), str(sorted(seen_hashes.items(), key=lambda x: -x[1])[:5]))
        if os.environ.get('BITGRID_DEBUG') == '1' and debug_phases:
            np.save('bitgrid_debug_phases.npy', np.array(debug_phases, dtype=object))
            logger.info('saved debug phases to bitgrid_debug_phases.npy')
    except Exception as e:
        logger.debug('failed to save diagnostics: %s', str(e))

    hist_path, summary_path = plot_bitgrid_metrics(ious, accs, bins=20, out_prefix='cifar_bitgrid')
    np.savez('cifar_bitgrid_metrics.npz', ious=np.array(ious), accs=np.array(accs))

    # save example images and per-sample CSV
    save_example_pairs(example_orig, example_rec, out_prefix='cifar_bitgrid_example', grid=16, max_examples=max_examples)
    np.savetxt('cifar_bitgrid_per_sample.csv', np.column_stack([np.array(ious), np.array(accs)]), delimiter=',', header='iou,bit_acc', comments='')

    # save per-class summary
    class_rows = []
    for label in sorted(per_class_ious.keys()):
        cls_ious = np.array(per_class_ious[label])
        cls_accs = np.array(per_class_accs[label])
        mean_cls_iou = float(np.mean(cls_ious)) if cls_ious.size else 0.0
        mean_cls_acc = float(np.mean(cls_accs)) if cls_accs.size else 0.0
        class_rows.append((label, mean_cls_iou, mean_cls_acc, cls_ious.size))

    np.savetxt('cifar_bitgrid_per_class.csv', np.array(class_rows), delimiter=',', header='label,mean_iou,mean_bit_acc,count', comments='')

    logger.info('plots saved: %s %s', hist_path, summary_path)
    logger.info('metrics saved: cifar_bitgrid_metrics.npz')
    logger.info('per-sample CSV saved: cifar_bitgrid_per_sample.csv')
    logger.info('per-class CSV saved: cifar_bitgrid_per_class.csv')
    logger.info('example images saved: cifar_bitgrid_example_0.png ...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='pokrov_model.pt')
    parser.add_argument('--n-eval', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cpu', help="'cpu' or 'cuda' (will fallback to cpu if cuda not available)")
    args = parser.parse_args()
    main(ckpt=args.ckpt, n_eval=args.n_eval, batch_size=args.batch_size, device_str=args.device)
