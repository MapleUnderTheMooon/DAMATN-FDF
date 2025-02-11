import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.vnet import VNet
from dataloaders import utils
from utils import ramps, losses
from dataloaders.la_heart import LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
from networks.net_factory import net_factory
from utils import test_patch

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str,  default='LA', help='dataset_name')
parser.add_argument('--root_path', type=str, default='E:/PyCharmProjects/UA-MT-master/data/2018LA_Seg_Training Set/', help='Name of Experiment')
# parser.add_argument('--exp', type=str,  default='Test_0.06_1-3', help='model_name')
parser.add_argument('--exp', type=str,  default='Test_11111111111111111111111111111111111111', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
# parser.add_argument('--max_iterations', type=int,  default=15000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
# parser.add_argument('--consistency', type=float,  default=0.4, help='consistency')
# parser.add_argument('--consistency', type=float,  default=0.6, help='consistency')
parser.add_argument('--consistency', type=float,  default=0.8, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')
parser.add_argument('--labelnum', type=int,  default=8, help='number of labeled data')
parser.add_argument('--max_samples', type=int,  default=80, help='maximum samples to train')
parser.add_argument('--lambda_tri_sup', type=float,  default=0.06, help='')
args = parser.parse_args()

train_data_path = args.root_path
# snapshot_path = "../model/" + args.exp + "/"

snapshot_path = "../model/" + args.exp + "_" + str(args.consistency) + "_" + str(args.max_iterations) + "_" + args.dataset_name + \
    "_{}labels/".format(args.labelnum)


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:  # 这个参数用于指示是否设置训练过程为确定性模式。
    cudnn.benchmark = False  # 取消将 cuDNN 的自动调整功能。
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (112, 112, 80)
def sharpening(P):
    T = 1 / 0.1
    P_sharpen = P ** T / (P ** T + (1 - P) ** T)
    return P_sharpen
def get_lambda_c(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)
def get_not_inf_not_nan_tensor(tensor):
    tensor = torch.where(
        torch.isnan(tensor),
        torch.full_like(tensor, 0),
        tensor)
    tensor = torch.where(
        torch.isinf(tensor),
        torch.full_like(tensor, 0),
        tensor)
    return tensor


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        # ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(ema=False):
        # Network definition
        net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    # model = create_model()
    # ema_model = create_model(ema=True)
    # 备用模型
    from networks.vnet import VNet, CAML3d_v2_MTNet, CAML3d_v2_MTNet_DN, CAML3d_v2_MTNet_CDMA_DN
    model = CAML3d_v2_MTNet_CDMA_DN(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True).cuda()
    # model = CAML3d_v2_MTNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True).cuda()
    # model = net_factory(net_type='caml3d_v1', in_chns=1, class_num=num_classes, mode="train")

    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    db_test = LAHeart(base_dir=train_data_path,
                       split='test',
                       transform = transforms.Compose([
                           CenterCrop(patch_size),
                           ToTensor()
                       ]))
    labeled_nums = args.labelnum
    # labeled_nums = 8
    labeled_idxs = list(range(labeled_nums))
    unlabeled_idxs = list(range(labeled_nums, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    # trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)

    # model.train()  # 学生模型
    # ema_model.train()  # 老师模型
    model.train()  # 备用模型
    # 优化器优化 学生模型和老师模型的参数
    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer_aux = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    if args.consistency_type == 'mse':  # 一致性类型
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    best_dice = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    # model.train()
    model.train()  # 备用模型
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[labeled_bs:]
            # outputs shape is  torch.Size([4, 2, 112, 112, 80])
            for num in range(3):
                if num == 0:
                    outputs, outputs_2, outputs_3, masks = model(volume_batch, [])
                else:
                    outputs, outputs_2, outputs_3, masks = model(volume_batch, en)

                en = []
                for idx in range(len(masks[0])):
                    mask1 = masks[0][idx].detach()
                    mask2 = masks[1][idx].detach()
                    # mask1 = sharpening(mask1)
                    # mask2 = sharpening(mask2)
                    en.append(1e-3 * (mask1 - mask2))  # 1e-3

                outputs = get_not_inf_not_nan_tensor(outputs)
                outputs_2 = get_not_inf_not_nan_tensor(outputs_2)
                outputs_3 = get_not_inf_not_nan_tensor(outputs_3)
                #
                outputs_4 = get_not_inf_not_nan_tensor((outputs + outputs_2 + outputs_3) / 3)

                # calculate the loss
                # 备用模型有标签数据的分割监督 loss
                loss_seg_aux = F.cross_entropy(outputs[:labeled_bs], label_batch[:labeled_bs])
                outputs_soft_aux = F.softmax(outputs, dim=1)  # (batch, 2, 112,112,80) 在 2 这个维度上进行softmax
                loss_seg_dice_aux = losses.dice_loss(outputs_soft_aux[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)  # label_batch[:labeled_bs] == 1 将返回一个布尔类型（Boolean）的张量
                # 第二个encoder的监督loss
                loss_seg_2 = F.cross_entropy(outputs_2[:labeled_bs], label_batch[:labeled_bs])
                outputs_soft_2 = F.softmax(outputs_2, dim=1)  # (batch, 2, 112,112,80) 在 2 这个维度上进行softmax
                loss_seg_dice_2 = losses.dice_loss(outputs_soft_2[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)  # label_batch[:labeled_bs] == 1 将返回一个布尔类型（Boolean）的张量
                # 第三个encoder的监督loss
                loss_seg_3 = F.cross_entropy(outputs_3[:labeled_bs], label_batch[:labeled_bs])
                outputs_soft_3 = F.softmax(outputs_3, dim=1)  # (batch, 2, 112,112,80) 在 2 这个维度上进行softmax
                loss_seg_dice_3 = losses.dice_loss(outputs_soft_3[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)  # label_batch[:labeled_bs] == 1 将返回一个布尔类型（Boolean）的张量
                # 三个decoder的监督loss
                loss_seg_4 = F.cross_entropy(outputs_4[:labeled_bs], label_batch[:labeled_bs])
                outputs_soft_4 = F.softmax(outputs_4, dim=1)  # (batch, 2, 112,112,80) 在 2 这个维度上进行softmax
                loss_seg_dice_4 = losses.dice_loss(outputs_soft_4[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)  # label_batch[:labeled_bs] == 1 将返回一个布尔类型（Boolean）的张量
                # 修改了一致性损失
                consistency_loss_seg_dice_1_2 = losses.mse_loss(outputs_soft_aux, sharpening(outputs_soft_2))  # label_batch[:labeled_bs] == 1 将返回一个布尔类型（Boolean）的张量
                consistency_loss_seg_dice_2_3 = losses.mse_loss(outputs_soft_2, sharpening(outputs_soft_3))  # label_batch[:labeled_bs] == 1 将返回一个布尔类型（Boolean）的张量
                consistency_loss_seg_dice_1_3 = losses.mse_loss(outputs_soft_aux, sharpening(outputs_soft_3))  # label_batch[:labeled_bs] == 1 将返回一个布尔类型（Boolean）的张量

                lambda_c = get_lambda_c((iter_num + 1) // 150)
                # 修改一致性 loss
                lambda_c = lambda_c * 0.5
                # lambda_c = lambda_c * 1.5
                consistency_loss_seg_dice_2_1 = losses.mse_loss(outputs_soft_2, sharpening(outputs_soft_aux))  # label_batch[:labeled_bs] == 1 将返回一个布尔类型（Boolean）的张量
                consistency_loss_seg_dice_3_2 = losses.mse_loss(outputs_soft_3, sharpening(outputs_soft_2))  # label_batch[:labeled_bs] == 1 将返回一个布尔类型（Boolean）的张量
                consistency_loss_seg_dice_3_1 = losses.mse_loss(outputs_soft_3, sharpening(outputs_soft_aux))  # label_batch[:labeled_bs] == 1 将返回一个布尔类型（Boolean）的张量
                consistency_loss_seg_dice_else_3 = lambda_c * (
                        consistency_loss_seg_dice_2_1 + consistency_loss_seg_dice_3_2 + consistency_loss_seg_dice_3_1
                )

                # consistency_loss_seg_dice_aux = lambda_c*consistency_loss_seg_dice_1_2 \
                #                               + lambda_c*consistency_loss_seg_dice_2_3 \
                #                               + lambda_c*consistency_loss_seg_dice_1_3 \
                #                               + consistency_loss_seg_dice_else_3
                # lambda_tri_sup = 0.66
                lambda_tri_sup = args.lambda_tri_sup
                # 监督loss
                supervised_loss = 0.5*(loss_seg_aux+loss_seg_dice_aux) \
                                  + 0.5*(loss_seg_2+loss_seg_dice_2) \
                                  + 0.5*(loss_seg_3+loss_seg_dice_3) \
                                  + lambda_tri_sup*(loss_seg_4+loss_seg_dice_4) \
                                  # + consistency_loss_seg_dice_aux
                # 仿照 CDMA
                supervised_loss = supervised_loss / 3

                consistency_weight = get_current_consistency_weight(iter_num//150)
                # 计算不确定性 loss
                outputs_avg_soft = (outputs_soft_aux + outputs_soft_2 + outputs_soft_3) / 3
                entropy_loss = losses.entropy_loss(outputs_avg_soft, C=2)
                # 最小化不确定性
                # consistency_weight_ = get_current_consistency_weight((1.5 * iter_num) // 150)
                consistency_weight_ = get_current_consistency_weight(iter_num // 150)
                # 降低权重值
                consistency_loss = consistency_weight_ * entropy_loss

                """计算有标签数据的分布 loss"""
                x1_label = outputs_soft_aux[:labeled_bs, 1, :, :, :]
                x2_label = outputs_soft_2[:labeled_bs, 1, :, :, :]
                x3_label = outputs_soft_3[:labeled_bs, 1, :, :, :]
                B, H, W, D = x1_label.shape

                model.dual_batch_norm(x1_label.view(B, 1, H, W, D), 1)
                model.dual_batch_norm_2(x2_label.view(B, 1, H, W, D), 1)
                model.dual_batch_norm_3(x3_label.view(B, 1, H, W, D), 1)

                x1_unlabel = outputs_soft_aux[labeled_bs:, 1, :, :, :]
                x2_unlabel = outputs_soft_2[labeled_bs:, 1, :, :, :]
                x3_unlabel = outputs_soft_3[labeled_bs:, 1, :, :, :]
                B, H, W, D = x1_unlabel.shape

                model.dual_batch_norm(x1_unlabel.view(B, 1, H, W, D), 0)
                model.dual_batch_norm_2(x2_unlabel.view(B, 1, H, W, D), 0)
                model.dual_batch_norm_3(x3_unlabel.view(B, 1, H, W, D), 0)

                dual_batch_norms = [model.dual_batch_norm, model.dual_batch_norm_2, model.dual_batch_norm_3]
                distribution_loss = 0
                for i in range(len(dual_batch_norms)):
                    bns_i_0 = dual_batch_norms[i].bns[0]
                    bns_i_1 = dual_batch_norms[i].bns[1]
                    for j in range(len(dual_batch_norms)):
                        if i != j:
                            bns_j_0 = dual_batch_norms[i].bns[0]
                            bns_j_1 = dual_batch_norms[i].bns[1]
                            mean_loss = losses.mse_loss(bns_i_1.running_mean, bns_j_1.running_mean)
                            variance_loss = losses.mse_loss(bns_i_1.running_var, bns_j_1.running_var)
                            distribution_loss += (mean_loss + variance_loss) * consistency_weight * 45

                            mean_loss = losses.mse_loss(bns_i_0.running_mean, bns_j_0.running_mean)
                            variance_loss = losses.mse_loss(bns_i_0.running_var, bns_j_0.running_var)
                            distribution_loss += (mean_loss + variance_loss) * consistency_weight * 45

                            mean_loss = losses.mse_loss(bns_i_1.running_mean, bns_j_0.running_mean)
                            variance_loss = losses.mse_loss(bns_i_1.running_var, bns_j_0.running_var)
                            distribution_loss += (mean_loss + variance_loss) * consistency_weight * 45

                        mean_loss = losses.mse_loss(bns_i_1.running_mean, bns_i_0.running_mean)
                        variance_loss = losses.mse_loss(bns_i_1.running_var, bns_i_0.running_var)
                        distribution_loss += (mean_loss + variance_loss) * consistency_weight * 45

                distribution_loss /= 3

                # distribution_loss = (0.01 if distribution_loss > 0.01 else distribution_loss)
                # distribution_loss = (mean_loss + variance_loss) * consistency_weight * 1
                # distribution_loss = (mean_loss + variance_loss) * consistency_weight

                # tri_sup_sharping_loss = losses.mse_loss(outputs_soft_4, sharpening(outputs_soft_4))
                # tri_sup_sharping_loss = get_current_consistency_weight(iter_num // 150) * tri_sup_sharping_loss * 10
                loss = supervised_loss + consistency_loss \
                       + distribution_loss \
                       # + tri_sup_sharping_loss

                # optimizer.zero_grad()
                optimizer_aux.zero_grad()
                loss.backward()
                # optimizer.step()
                optimizer_aux.step()
                # EMA 更新 DN
                for i in range(len(dual_batch_norms)):
                    bn_1 = dual_batch_norms[i].bns[1]
                    bn_0 = dual_batch_norms[i].bns[0]

                    update_ema_variables(bn_1, bn_0, args.ema_decay, iter_num)

                # # EMA 更新 DN UNLABEL2LABEL
                # update_ema_variables(bn_0, bn_1, args.ema_decay, iter_num)

            iter_num = iter_num + 1

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            # writer.add_scalar('loss/loss_seg', supervised_loss - consistency_loss_seg_dice_aux, iter_num)
            writer.add_scalar('loss/loss_seg', supervised_loss, iter_num)
            writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
            # writer.add_scalar('train/tri_sup_sharping_loss', tri_sup_sharping_loss, iter_num)
            writer.add_scalar('train/distribution_loss', distribution_loss, iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            # writer.add_scalar('train/consistency_loss_seg_dice_aux', consistency_loss_seg_dice_aux, iter_num)
            writer.add_scalar('train/tri_sup_loss', lambda_tri_sup * (loss_seg_4+loss_seg_dice_4) / 3, iter_num)

            # logging.info('iteration %d : loss : %f  loss_weight: %f, supervised_loss:%f, consistency_loss:%f, distribution_loss:%f'
            logging.info('iteration %d : loss : %f  loss_weight: %f, supervised_loss:%f, consistency_loss:%f'
                         # ', consistency_loss_seg_dice_aux:%f'
                         ', distribution_loss:%f'
                         ', tri_sup_loss:%f'
                         # ', sharping_loss:%f'
                          %
#                          (iter_num, loss.item(), consistency_weight, supervised_loss, consistency_loss, distribution_loss))
                         (iter_num, loss.item(), consistency_weight, supervised_loss, consistency_loss
#                           , consistency_loss_seg_dice_aux
                          , distribution_loss
                          , lambda_tri_sup * (loss_seg_4+loss_seg_dice_4) / 3
#                           , tri_sup_sharping_loss
                          ))

            # if iter_num % 50 == 0:
            if iter_num % 200 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                # image = outputs_soft[0, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                image = torch.max(outputs_soft_4[0, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label_1_2_3', grid_image, iter_num)

                # image = outputs_soft[0, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                image = torch.max(outputs_soft_aux[0, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label_1', grid_image, iter_num)

                # image = outputs_soft[0, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                image = torch.max(outputs_soft_2[0, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label_2', grid_image, iter_num)

                # image = outputs_soft[0, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                image = torch.max(outputs_soft_3[0, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label_3', grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].permute(2, 0, 1)
                grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                writer.add_image('train/Groundtruth_label', grid_image, iter_num)

                #####
                image = volume_batch[-1, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('unlabel/Image', grid_image, iter_num)

                # image = outputs_soft[-1, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                image = torch.max(outputs_soft_4[-1, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('unlabel/Predicted_label_1_2_3', grid_image, iter_num)

                # image = outputs_soft[-1, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                image = torch.max(outputs_soft_aux[-1, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('unlabel/Predicted_label_1', grid_image, iter_num)

                # image = outputs_soft[-1, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                image = torch.max(outputs_soft_2[-1, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('unlabel/Predicted_label_2', grid_image, iter_num)

                # image = outputs_soft[-1, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                image = torch.max(outputs_soft_3[-1, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('unlabel/Predicted_label_3', grid_image, iter_num)

                image = label_batch[-1, :, :, 20:61:10].permute(2, 0, 1)
                grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                writer.add_image('unlabel/Groundtruth_label', grid_image, iter_num)

            if iter_num >= 800 and iter_num % 200 == 0:
                model.eval()
                dice_sample = 0
                if args.dataset_name == "LA":
                    dice_sample = test_patch.var_all_case(model, num_classes=num_classes, patch_size=patch_size,
                                                          stride_xy=18, stride_z=4, dataset_name='LA')
                    logging.info('curr_dice: %f, best_dice: %f' % (dice_sample, best_dice))
                if dice_sample > best_dice:
                    best_dice = dice_sample
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.exp))
                    torch.save(model.state_dict(), save_best_path)
                    writer.add_scalar('Var_dice/Best_dice', best_dice, iter_num)
                writer.add_scalar('Var_dice/Dice', dice_sample, iter_num)
                model.train()

            ## change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                # for param_group in optimizer.param_groups:
                #     param_group['lr'] = lr_
                # 更新备用模型的 lr
                for param_group in optimizer_aux.param_groups:
                    param_group['lr'] = lr_
            # if iter_num % 1000 == 0:
            # # if iter_num % 8000 == 0:
            #     save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
            #     torch.save(model.state_dict(), save_mode_path)
            #     logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break
    # save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations)+'.pth')
    # torch.save(model.state_dict(), save_mode_path)
    # logging.info("save model to {}".format(save_mode_path))
    logging.info("End model at{}".format(snapshot_path))
    writer.close()
