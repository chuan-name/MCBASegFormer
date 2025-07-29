import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from network import mcbasegformer
import imageio
from utils.dataloader import test_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./checkpoints/MCBASegFormer/MCBASegFormer.pth')

for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
    data_path = './dataset/TestDataset/{}'.format(_data_name)
    save_path = './results/MCBASegFormer/{}/'.format(_data_name)
    opt = parser.parse_args()
    model = mcbasegformer()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        if gt.max() > 1:
            gt = (gt > 127).astype(np.float32)

        image = image.cuda()

        predicts = model(image)
        res = predicts[0]
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.data.cpu().numpy().squeeze()

        if i == 0:
            print(f"Raw prediction range: [{res.min():.4f}, {res.max():.4f}]")
            print(f"GT range: [{gt.min():.4f}, {gt.max():.4f}]")

        res = 1 / (1 + np.exp(-res))
        print(f"[{name}] after sigmoid: min={res.min():.4f}, max={res.max():.4f}, mean={res.mean():.4f}")

        imageio.imwrite(save_path + name, (res * 255).astype(np.uint8))
