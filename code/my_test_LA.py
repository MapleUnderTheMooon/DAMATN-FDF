import os
import argparse
import torch
from networks.vnet import VNet, CAML3d_v2_MTNet_DN, CAML3d_v2_MTNet_CDMA_DN
from test_util import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str,  default='LA', help='dataset_name')
parser.add_argument('--root_path', type=str, default='E:/PyCharmProjects/UA-MT-master/data/2018LA_Seg_Training Set/', help='Name of Experiment')
# parser.add_argument('--model', type=str,  default='vnet_supervisedonly_dp', help='model_name')
# parser.add_argument('--model', type=str,  default='UAMT_unlabel', help='model_name')
parser.add_argument('--model', type=str,  default='Ours_8', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
# test_type = 'model'
test_type = 't_ablation'
snapshot_path = "../" + test_type + "/"+FLAGS.model+"/"
test_save_path = "../" + test_type + "/prediction/"+FLAGS.model+"_post/"
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = 2

with open(FLAGS.root_path + '/../test.list', 'r') as f:
    image_list = f.readlines()
image_list = [FLAGS.root_path +item.replace('\n', '')+"/mri_norm2.h5" for item in image_list]


def test_calculate_metric(epoch_num):
    # net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False).cuda()
    # has_dropout 作为测试模式 的参数
    net = CAML3d_v2_MTNet_CDMA_DN(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False).cuda()
    # save_mode_path = os.path.join(snapshot_path, FLAGS.model.split('_'+FLAGS.dataset_name)[0] + '_best_model.pth')
    # save_mode_path = os.path.join(snapshot_path, 'Test_0.06_1-3_best_model.pth')
    # save_mode_path = os.path.join(snapshot_path, 'Test_0.4_0.06_1-3_15000_best_model.pth')
    save_mode_path = os.path.join(snapshot_path, FLAGS.model.split("0")[0] + 'best_model.pth')
    # save_mode_path = os.path.join(snapshot_path, 'iter_' + str(epoch_num) + '.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    avg_metric = test_all_case(net, image_list, num_classes=num_classes,
                               patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                               save_result=True, test_save_path=test_save_path)

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric(6000)
    # metric = test_calculate_metric(8000)
    # metric = test_calculate_metric(15000)
    print(metric)