from torch.utils.data import Dataset
import glob
import random
from PIL import Image
import torch
from augment import strong, normalize, text_augment_pool, weak
import numpy as np
import cv2


class Batch_Balanced_Dataset(object):
    def __init__(self, opt):
        label_file_path = opt.data_path + "/" + opt.dataset_name + '/label/' + opt.label_file
        with open(label_file_path, 'r') as f:
            lines = f.read().strip().split("\n")
            label_dict = {line.split(" ")[0]: line.split(" ")[1] for line in lines}
        labeled_train_filenames = glob.glob(opt.data_path + "/" + opt.dataset_name + "/train/*.*")
        labeled_train_filenames = [train_filename for train_filename in labeled_train_filenames if
                                   train_filename.split("/")[-1] in label_dict]

        nolabeled_train_filenames = glob.glob(opt.data_path + "/" + opt.dataset_name + "/buchong/*.*")
        nolabeled_train_filenames = sorted(nolabeled_train_filenames)
        nolabeled_train_filenames = random.sample(nolabeled_train_filenames, opt.unlabeled_number) \
                                    + labeled_train_filenames

        test_filenames = glob.glob(opt.data_path + "/" + opt.dataset_name + "/test/*.*")

        nolabeled_train_filenames = sorted(nolabeled_train_filenames)
        test_filenames = sorted(test_filenames)

        print("label and nolabel: %d" %len(nolabeled_train_filenames))
        print("test: %d" % len(test_filenames))

        labeled_dataset = CaptchaDataset(labeled_train_filenames, opt.cut)
        unlabeled_dataset = CaptchaDataset(nolabeled_train_filenames, opt.cut)
        test_dataset = CaptchaDataset(test_filenames, opt.cut)

        Labeled_AlignCollate = LabelAlignCollate()
        Unlabeled_AlignCollate = AlignCollate()
        Test_AlignCollate = TestAlignCollate()

        self.label_data_loader = torch.utils.data.DataLoader(
            labeled_dataset, batch_size=opt.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=opt.workers,
            collate_fn=Labeled_AlignCollate)

        self.unlabel_data_loader = torch.utils.data.DataLoader(
            unlabeled_dataset, batch_size=opt.batch_size*2,
            shuffle=True,
            drop_last=True,
            num_workers=opt.workers,
            collate_fn=Unlabeled_AlignCollate)

        self.test_data_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=opt.test_batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=opt.workers,
            collate_fn=Test_AlignCollate)

        self.label_data_loader_iter = iter(self.label_data_loader)
        self.unlabel_data_loader_iter = iter(self.unlabel_data_loader)
        self.test_data_loader_iter = iter(self.test_data_loader)

    def get_batch(self):
        try:
            images, text = self.label_data_loader_iter.next()
        except StopIteration:
            self.label_data_loader_iter = iter(self.label_data_loader)
            images, text = self.label_data_loader_iter.next()

        try:
            images_1, images_2 = self.unlabel_data_loader_iter.next()
        except StopIteration:
            self.unlabel_data_loader_iter = iter(self.unlabel_data_loader)
            images_1, images_2 = self.unlabel_data_loader_iter.next()

        return images, text, images_1, images_2

    def get_batch_test(self):
        try:
            images, labels = self.test_data_loader_iter.next()
        except StopIteration:
            self.test_data_loader_iter = iter(self.test_data_loader)
            images, labels = self.test_data_loader_iter.next()

        return images, labels


class LabelAlignCollate(object):
    def __init__(self):
        self.transforms = weak
        self.pools = text_augment_pool()

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        images = [image.resize((200, 70), Image.ANTIALIAS) for image in images]

        image_tensors = [self.transforms(image) for image in images]
        image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
        return image_tensors, labels


class AlignCollate(object):
    def __init__(self):
        self.weak = weak
        self.transforms = strong
        self.pools = text_augment_pool()

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        images = [image.resize((200, 70), Image.ANTIALIAS) for image in images]

        image_tensors_1 = [self.weak(image) for image in images]
        image_tensors_1 = torch.cat([t.unsqueeze(0) for t in image_tensors_1], 0)

        images = [np.array(image) for image in images]
        new_images = []
        for image in images:
            op, v = random.choice(self.pools)
            new_images.append(op(image, v))
        images = [Image.fromarray(image) for image in new_images]

        image_tensors_2 = [self.transforms(image) for image in images]
        image_tensors_2 = torch.cat([t.unsqueeze(0) for t in image_tensors_2], 0)
        return image_tensors_1, image_tensors_2


def get_vvList(list_data):
    vv_list=list()
    for index,i in enumerate(list_data):
        if i>0:
            vv_list.append(index)
    return vv_list


def extract_fig(filename):
    img_bgr = cv2.imread(filename)
    if img_bgr is None:
        img_bgr = Image.open(filename).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(img_bgr),cv2.COLOR_RGB2BGR)

    img = img_bgr.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    t, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    rows, cols = binary.shape

    hor_list = [0] * rows
    for i in range(rows):
        for j in range(cols):
            if binary.item(i, j) == 0:
                hor_list[i] = hor_list[i] + 1

    ver_list = [0] * cols
    for i in range(cols):
        for j in range(rows):
            if binary.item(j, i) == 0:
                ver_list[i] = ver_list[i] + 1

    vv_list = get_vvList(hor_list)
    hh_list = get_vvList(ver_list)

    shift = 0
    img_hor = img_bgr[vv_list[0] - shift:vv_list[-1] + shift, hh_list[0] - shift:hh_list[-1] + shift, :]
    img_hor = cv2.cvtColor(img_hor, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_hor)
    img = img.resize((200, 70), Image.ANTIALIAS)
    return img


class CaptchaDataset(Dataset):
    def __init__(self, filenames, cut):
        self.filenames = filenames
        self.cut = cut

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        if self.cut:
            img = extract_fig(filename)
        else:
            img = Image.open(filename).convert('RGB')
        label = filename.split("/")[-1].split(".")[0]

        return img, label


class TestAlignCollate(object):
    def __init__(self):
        self.transforms = normalize

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        images = [image.resize((200, 70), Image.ANTIALIAS) for image in images]
        image_tensors = [self.transforms(image) for image in images]
        image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
        return image_tensors, labels
