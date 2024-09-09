import logging
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from utils.read_data import read_split_data_mydata
from utils.my_dataset import MyDataSet
logger = logging.getLogger(__name__)


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if args.dataset == 'mydata':
        train_images_path, train_images_label, val_images_path, val_images_label = read_split_data_mydata(
            args.data_path)
        # 实例化训练数据集
        trainset = MyDataSet(images_path=train_images_path,
                             images_class=train_images_label,
                             transform=transforms.Compose([
                                 transforms.ToTensor()])
                             )
        # 实例化验证数据集
        testset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=transforms.Compose([
                                transforms.ToTensor()])
                            )
    else:
        print('数据集异常')

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset)
    test_sampler = SequentialSampler(testset)

    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=args.num_workers,
                              drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=args.num_workers,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader