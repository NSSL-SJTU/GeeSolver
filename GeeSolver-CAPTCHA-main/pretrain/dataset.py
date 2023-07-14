from torch.utils.data import Dataset
import glob
import random
from PIL import Image
import torch
from augment import strong, normalize, text_augment_pool
import numpy as np


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

        unlabeled_dataset = CaptchaDataset(nolabeled_train_filenames)
        test_dataset = CaptchaDataset(test_filenames)
        Unlabeled_AlignCollate = AlignCollate(is_training=True)
        Test_AlignCollate = AlignCollate(is_training=False)

        self.unlabel_data_loader = torch.utils.data.DataLoader(
            unlabeled_dataset, batch_size=opt.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=opt.workers,
            collate_fn=Unlabeled_AlignCollate)
        self.test_data_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=opt.test_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=opt.workers,
            collate_fn=Test_AlignCollate)

        self.unlabel_data_loader_iter = iter(self.unlabel_data_loader)
        self.test_data_loader_iter = iter(self.test_data_loader)

    def get_batch(self):
        try:
            images, labels = self.unlabel_data_loader_iter.next()
        except StopIteration:
            self.unlabel_data_loader_iter = iter(self.unlabel_data_loader)
            images, labels = self.unlabel_data_loader_iter.next()

        return images, labels

    def get_batch_test(self):
        try:
            images, labels = self.test_data_loader_iter.next()
        except StopIteration:
            self.test_data_loader_iter = iter(self.test_data_loader)
            images, labels = self.test_data_loader_iter.next()

        return images, labels


class AlignCollate(object):
    def __init__(self, is_training=False):
        self.is_training = is_training
        if is_training:
            self.transforms = strong
            self.pools = text_augment_pool()
        else:
            self.transforms = normalize

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        if not self.is_training:
            images = [image.resize((200, 70), Image.ANTIALIAS) for image in images]
        else:
            images = [np.array(image) for image in images]
            new_images = []
            for image in images:
                op, v = random.choice(self.pools)
                new_images.append(op(image, v))
            images = [Image.fromarray(image) for image in new_images]

        image_tensors = [self.transforms(image) for image in images]
        image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)
        return image_tensors, labels


class CaptchaDataset(Dataset):
    def __init__(self, filenames):
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        img = Image.open(filename).convert('RGB')
        label = filename.split("/")[-1].split(".")[0]

        return img, label
