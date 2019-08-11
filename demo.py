import os
import cv2
import sys
import time
import collections
import torch
import argparse
import numpy as np
import random
from PIL import Image

from torch.autograd import Variable
from torch.utils import data
import torchvision.transforms as transforms

import models
import util
# c++ version pse based on opencv 3+
from pse import pse

random.seed(123456)


def get_img(img_path):
    try:
        img = cv2.imread(img_path)
        img = img[:, :, [2, 1, 0]]
    except Exception as e:
        print(img_path)
        raise
    return img


def scale(img, long_size=2240):
    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img


class DemoDataLoader(data.Dataset):
    def __init__(self, input_path, part_id=0, part_num=1, long_size=2240):
        data_dirs = [input_path]

        self.img_paths = []

        for data_dir in data_dirs:
            img_names = util.io.ls(data_dir, '.jpg')
            img_names.extend(util.io.ls(data_dir, '.png'))

            img_paths = []
            for idx, img_name in enumerate(img_names):
                img_path = data_dir + img_name
                img_paths.append(img_path)

            self.img_paths.extend(img_paths)

        part_size = len(self.img_paths) / part_num
        l = int(part_id * part_size)
        r = int((part_id + 1) * part_size)
        self.img_paths = self.img_paths[l:r]
        self.long_size = long_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        img = get_img(img_path)

        scaled_img = scale(img, self.long_size)
        scaled_img = Image.fromarray(scaled_img)
        scaled_img = scaled_img.convert('RGB')
        scaled_img = transforms.ToTensor()(scaled_img)
        scaled_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(scaled_img)

        return img[:, :, [2, 1, 0]], scaled_img


def debug(idx, img_paths, imgs, output_root):
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    col = []
    for i in range(len(imgs)):
        row = []
        for j in range(len(imgs[i])):
            # img = cv2.copyMakeBorder(imgs[i][j], 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 0, 0])
            row.append(imgs[i][j])
        res = np.concatenate(row, axis=1)
        col.append(res)
    res = np.concatenate(col, axis=0)
    img_name = img_paths[idx].split('/')[-1]
    print(idx, '/', len(img_paths), img_name)
    cv2.imwrite(output_root + img_name, res)


def write_result_as_txt(image_name, bboxes, path):
    filename = util.io.join_path(path, 'res_%s.txt' % (image_name))
    lines = []
    for b_idx, bbox in enumerate(bboxes):
        values = [int(v) for v in bbox]
        line = "%d, %d, %d, %d, %d, %d, %d, %d\n" % tuple(values)
        lines.append(line)
    util.io.write_lines(filename, lines)


def test(args):
    data_loader = DemoDataLoader(long_size=args.long_size, input_path=args.input_dir)
    test_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=True)

    # Setup Model
    if args.arch == "resnet50":
        model = models.resnet50(pretrained=True, num_classes=7, scale=args.scale)
    elif args.arch == "resnet101":
        model = models.resnet101(pretrained=True, num_classes=7, scale=args.scale)
    elif args.arch == "resnet152":
        model = models.resnet152(pretrained=True, num_classes=7, scale=args.scale)

    for param in model.parameters():
        param.requires_grad = False

    model = model.cuda()

    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(("Loading model and optimizer from checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)

            # model.load_state_dict(checkpoint['state_dict'])
            d = collections.OrderedDict()
            for key, value in list(checkpoint['state_dict'].items()):
                tmp = key[7:]
                d[tmp] = value
            model.load_state_dict(d)

            print(("Loaded checkpoint '{}' (epoch {})"
                   .format(args.resume, checkpoint['epoch'])))
            sys.stdout.flush()
        else:
            print(("No checkpoint found at '{}'".format(args.resume)))
            sys.stdout.flush()

    model.eval()

    total_frame = 0.0
    total_time = 0.0
    with torch.no_grad():
        for idx, (org_img, img) in enumerate(test_loader):
            print(('progress: %d / %d' % (idx, len(test_loader))))
            sys.stdout.flush()

            img = Variable(img.cuda())
            org_img = org_img.numpy().astype('uint8')[0]
            text_box = org_img.copy()

            torch.cuda.synchronize()
            start = time.time()

            outputs = model(img)

            score = torch.sigmoid(outputs[:, 0, :, :])
            outputs = (torch.sign(outputs - args.binary_th) + 1) / 2

            text = outputs[:, 0, :, :]
            kernels = outputs[:, 0:args.kernel_num, :, :] * text

            score = score.data.cpu().numpy()[0].astype(np.float32)
            text = text.data.cpu().numpy()[0].astype(np.uint8)
            kernels = kernels.data.cpu().numpy()[0].astype(np.uint8)

            # c++ version pse
            pred = pse(kernels, args.min_kernel_area / (args.scale * args.scale))

            scale = (org_img.shape[1] * 1.0 / pred.shape[1], org_img.shape[0] * 1.0 / pred.shape[0])
            label = pred
            label_num = np.max(label) + 1
            bboxes = []
            for i in range(1, label_num):
                points = np.array(np.where(label == i)).transpose((1, 0))[:, ::-1]

                if points.shape[0] < args.min_area / (args.scale * args.scale):
                    continue

                score_i = np.mean(score[label == i])
                if score_i < args.min_score:
                    continue

                rect = cv2.minAreaRect(points)
                bbox = cv2.boxPoints(rect) * scale
                bbox = bbox.astype('int32')
                bboxes.append(bbox.reshape(-1))

            torch.cuda.synchronize()
            end = time.time()
            total_frame += 1
            total_time += (end - start)
            print(('fps: %.2f' % (total_frame / total_time)))
            sys.stdout.flush()

            for bbox in bboxes:
                cv2.drawContours(text_box, [bbox.reshape(4, 2)], -1, (0, 255, 0), 10)

            image_name = data_loader.img_paths[idx].split('/')[-1].split('.')[0]
            write_result_as_txt(image_name, bboxes, 'outputs/demo/')

            text_box = cv2.resize(text_box, (text.shape[1], text.shape[0]))
            debug(idx, data_loader.img_paths, [[text_box]], 'outputs/demo/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='resnet50')
    parser.add_argument('--resume', nargs='?', type=str, default=None,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--input_dir', nargs='?', type=str, default='../data/demo/',
                        help='Path to input directory')
    parser.add_argument('--binary_th', nargs='?', type=float, default=1.0,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--kernel_num', nargs='?', type=int, default=7,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--scale', nargs='?', type=int, default=1,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--long_size', nargs='?', type=int, default=2240,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--min_kernel_area', nargs='?', type=float, default=5.0,
                        help='min kernel area')
    parser.add_argument('--min_area', nargs='?', type=float, default=800.0,
                        help='min area')
    parser.add_argument('--min_score', nargs='?', type=float, default=0.93,
                        help='min score')

    args = parser.parse_args()
    test(args)
