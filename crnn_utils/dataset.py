import torch
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
import json

from PIL import Image

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from glob import glob


class CustomDatasetWithBoundingBox(Dataset):

    def __init__(self, json_path='printed_data_info.json', origin_data_path='./test', result_data_path='./result',
                 transform=None):

        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        # self.types = []
        self.labels = []
        for i in json_data['annotations']:
            # self.types.append((i['attributes']['type']))
            self.labels.append(i['text'])

        self.poly_files = glob(result_data_path + '/*.txt')
        self.polys = []

        for i in self.poly_files:
            with open(i, 'r') as f:
                lines = f.readlines()
                squares = []
                for poly in lines[0::2]:
                    x1, y1, x2, y2, x3, y3, x4, y4 = map(float, poly.split(',')[:8])
                    squares.append(np.array([x1, y1, x2, y2, x3, y3, x4, y4]).reshape(4, 2))
                self.polys.append(squares)

        self.image_files = glob(origin_data_path + '/*.png')

        self.images = [cv2.imread(i) for i in self.image_files]
        self.data = list(zip(self.images, self.polys, self.labels))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = self.data[idx][0]
        image = Image.fromarray(image)
        boxes = self.data[idx][1]
        labels = self.data[idx][2]

        if self.transform:
            image = self.transform(image)

        return image, labels, boxes

    def __plotting__(self, idx):

        fig, ax = plt.subplots()
        ax.imshow(self.images[idx])
        for coord in self.polys[idx][:]:
            polygon = Polygon(coord, closed=True, edgecolor='r', fill=False)
            ax.add_patch(polygon)
        plt.show()


def collate_fn(batch):
    images, labels, bounding_boxes = zip(*batch)

    images = torch.stack(images, dim=0)

    # 레이블을 정수 시퀀스로 변환
    start = 44032  # '가'의 유니코드 값
    char_set_korean = list(chr(start + i) for i in range(11172))

    # 레이블을 정수 시퀀스로 변환
    label_lengths = torch.tensor([len(label) for label in labels])
    targets = torch.nn.utils.rnn.pad_sequence([
        torch.tensor([char_set_korean.index(char) for char in label], dtype=torch.long) for label in labels
    ], batch_first=True)

    # 바운딩 박스 정보를 텐서로 변환 (좌표를 정규화하여 표현)
    normalized_bounding_boxes = []
    for boxes in bounding_boxes:
        # 각 좌표를 이미지의 너비와 높이에 대한 상대적인 비율로 정규화
        normalized_boxes = torch.FloatTensor([[coord[0] / images.size(3), coord[1] / images.size(2)] for coord in boxes])
        normalized_bounding_boxes.append(normalized_boxes)

    bounding_box_lengths = torch.tensor([len(boxes) for boxes in normalized_bounding_boxes])
    bounding_boxes = torch.nn.utils.rnn.pad_sequence(normalized_bounding_boxes, batch_first=True)

    return images, targets, bounding_boxes, label_lengths, bounding_box_lengths


def get_dataloader(json_path='printed_data_info.json', origin_data_path='./test', result_data_path='./result',
                   batch_size=8, num_workers=0):
    transform = transforms.Compose([
        transforms.Resize((32, 100)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    dataset = CustomDatasetWithBoundingBox(json_path=json_path, origin_data_path=origin_data_path,
                                           result_data_path=result_data_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                             collate_fn=collate_fn)

    return dataloader
