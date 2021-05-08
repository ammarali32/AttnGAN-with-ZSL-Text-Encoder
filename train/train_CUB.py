##ref https://github.com/EthanZhu90/ZSL_GAN/blob/master/train_CUB.py

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.init as init

from sklearn.metrics.pairwise import cosine_similarity
import scipy.integrate as integrate
from termcolor import cprint
from time import gmtime, strftime
import numpy as np
import os
import random
import glob
import copy


from datasets.dataset import FeatDataLayer, LoadDataset
from models.modelFromZSL import _netD, _netG, _param
from cfg import configs as configs

os.environ['CUDA_VISIBLE_DEVICES'] = configs.gpu


if configs.manualSeed is None:
    configs.manualSeed = random.randint(1, 10000)
print("Random Seed: ", configs.manualSeed)
random.seed(configs.manualSeed)
torch.manual_seed(configs.manualSeed)
torch.cuda.manual_seed_all(configs.manualSeed)

def train():
    print("start training ....")
    param = _param()
    dataset = LoadDataset(configs)
    param.X_dim = dataset.feature_dim

    data_layer = FeatDataLayer(dataset.labels_train, dataset.pfc_feat_data_train, configs)
    result = Result()
    result_gzsl = Result()
    netG = _netG(dataset.text_dim, dataset.feature_dim).to(configs.device)  # .cuda()
    netG.apply(weights_init)
    # print(netG)
    netD = _netD(dataset.train_cls_num, dataset.feature_dim).to(configs.device)  # .cuda()
    netD.apply(weights_init)
    # print(netD)

    exp_info = 'CUB_EASY' if configs.splitmode == 'easy' else 'CUB_HARD'
    exp_params = 'Eu{}_Rls{}_RWz{}'.format(configs.CENT_LAMBDA, configs.REG_W_LAMBDA, configs.REG_Wz_LAMBDA)

    out_dir = 'out/{:s}'.format(exp_info)
    out_subdir = 'out/{:s}/{:s}'.format(exp_info, exp_params)
    if not os.path.exists('out'):
        os.mkdir('out')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(out_subdir):
        os.mkdir(out_subdir)

    cprint(" The output dictionary is {}".format(out_subdir), 'red')
    log_dir = out_subdir + '/log_{:s}.txt'.format(exp_info)
    with open(log_dir, 'a') as f:
        f.write('Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')

    start_step = 0

    if configs.resume:
        if os.path.isfile(configs.resume):
            print("=> loading checkpoint '{}'".format(configs.resume))
            checkpoint = torch.load(configs.resume)
            netG.load_state_dict(checkpoint['state_dict_G'])
            netD.load_state_dict(checkpoint['state_dict_D'])
            start_step = checkpoint['it']
            print(checkpoint['log'])
        else:
            print("=> no checkpoint found at '{}'".format(configs.resume))

    nets = [netG, netD]

    tr_cls_centroid = Variable(torch.from_numpy(dataset.tr_cls_centroid.astype('float32'))).to(configs.device)  # .cuda()
    optimizerD = optim.Adam(netD.parameters(), lr=configs.lr, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=configs.lr, betas=(0.5, 0.9))

    for it in range(start_step, 3000 + 1):
        """ Discriminator """
        for _ in range(5):
            blobs = data_layer.forward()
            feat_data = blobs['data']  # image data
            labels = blobs['labels'].astype(int)  # class labels
            text_feat = np.array([dataset.train_text_feature[i, :] for i in labels])
            text_feat = Variable(torch.from_numpy(text_feat.astype('float32'))).to(configs.device)  # .cuda()
            X = Variable(torch.from_numpy(feat_data)).to(configs.device)  # .cuda()
            y_true = Variable(torch.from_numpy(labels.astype('int'))).to(configs.device)  # .cuda()
            z = Variable(torch.randn(configs.batchsize, param.z_dim)).to(configs.device)  # .cuda()

            # GAN's D loss
            D_real, C_real = netD(X)
            D_loss_real = torch.mean(D_real)
            C_loss_real = F.cross_entropy(C_real, y_true)
            DC_loss = -D_loss_real + C_loss_real
            DC_loss.backward()

            # GAN's D loss
            G_sample = netG(z, text_feat).detach()
            D_fake, C_fake = netD(G_sample)
            D_loss_fake = torch.mean(D_fake)
            C_loss_fake = F.cross_entropy(C_fake, y_true)
            DC_loss = D_loss_fake + C_loss_fake
            DC_loss.backward()

            # train with gradient penalty (WGAN_GP)
            grad_penalty = calc_gradient_penalty(netD, X.data, G_sample.data)
            grad_penalty.backward()

            Wasserstein_D = D_loss_real - D_loss_fake
            optimizerD.step()
            reset_grad(nets)

        """ Generator """
        for _ in range(1):
            blobs = data_layer.forward()
            feat_data = blobs['data']  # image data
            labels = blobs['labels'].astype(int)  # class labels
            text_feat = np.array([dataset.train_text_feature[i, :] for i in labels])
            text_feat = Variable(torch.from_numpy(text_feat.astype('float32'))).to(configs.device)  # .cuda()

            X = Variable(torch.from_numpy(feat_data)).to(configs.device)  # .cuda()
            y_true = Variable(torch.from_numpy(labels.astype('int'))).to(configs.device)  # .cuda()
            z = Variable(torch.randn(configs.batchsize, param.z_dim)).to(configs.device)  # .cuda()

            G_sample = netG(z, text_feat)
            D_fake, C_fake = netD(G_sample)
            _, C_real = netD(X)

            # GAN's G loss
            G_loss = torch.mean(D_fake)
            # Auxiliary classification loss
            C_loss = (F.cross_entropy(C_real, y_true) + F.cross_entropy(C_fake, y_true)) / 2

            GC_loss = -G_loss + C_loss

            # Centroid loss
            Euclidean_loss = Variable(torch.Tensor([0.0])).to(configs.device)  # .cuda()
            if configs.CENT_LAMBDA != 0:
                for i in range(dataset.train_cls_num):
                    sample_idx = (y_true == i).data.nonzero().squeeze()
                    if sample_idx.numel() == 0:
                        Euclidean_loss += 0.0
                    else:
                        G_sample_cls = G_sample[sample_idx, :]
                        Euclidean_loss += (G_sample_cls.mean(dim=0) - tr_cls_centroid[i]).pow(2).sum().sqrt()
                Euclidean_loss *= 1.0 / dataset.train_cls_num * configs.CENT_LAMBDA

            # ||W||_2 regularization
            reg_loss = Variable(torch.Tensor([0.0])).to(configs.device)  # .cuda()
            if configs.REG_W_LAMBDA != 0:
                for name, p in netG.named_parameters():
                    if 'weight' in name:
                        reg_loss += p.pow(2).sum()
                reg_loss.mul_(configs.REG_W_LAMBDA)

            # ||W_z||21 regularization, make W_z sparse
            reg_Wz_loss = Variable(torch.Tensor([0.0])).to(configs.device)  # .cuda()
            if configs.REG_Wz_LAMBDA != 0:
                Wz = netG.rdc_text.weight
                reg_Wz_loss = Wz.pow(2).sum(dim=0).sqrt().sum().mul(configs.REG_Wz_LAMBDA)

            all_loss = GC_loss + Euclidean_loss + reg_loss + reg_Wz_loss
            all_loss.backward()
            optimizerG.step()
            reset_grad(nets)

        if it % configs.disp_interval == 0 and it:
            acc_real = (np.argmax(C_real.data.cpu().numpy(), axis=1) == y_true.data.cpu().numpy()).sum() / float(
                y_true.data.size()[0])
            acc_fake = (np.argmax(C_fake.data.cpu().numpy(), axis=1) == y_true.data.cpu().numpy()).sum() / float(
                y_true.data.size()[0])

            log_text = 'Iter-{}; Was_D: {:.4}; Euc_ls: {:.4}; reg_ls: {:.4}; Wz_ls: {:.4}; G_loss: {:.4}; D_loss_real: {:.4};' \
                       ' D_loss_fake: {:.4}; rl: {:.4}%; fk: {:.4}%' \
                .format(it, Wasserstein_D.item(), Euclidean_loss.item(), reg_loss.item(), reg_Wz_loss.item(),
                        G_loss.item(), D_loss_real.item(), D_loss_fake.item(), acc_real * 100, acc_fake * 100)
            print(log_text)
            with open(log_dir, 'a') as f:
                f.write(log_text + '\n')

        if it % configs.evl_interval == 0 and it >= 100:
            netG.eval()
            eval_fakefeat_test(it, netG, dataset, param, result)
            eval_fakefeat_GZSL(it, netG, dataset, param, result_gzsl)
            if result.save_model:
                files2remove = glob.glob(out_subdir + '/Best_model*')
                for _i in files2remove:
                    os.remove(_i)
                torch.save({
                    'it': it + 1,
                    'state_dict_G': netG.state_dict(),
                    'state_dict_D': netD.state_dict(),
                    'random_seed': configs.manualSeed,
                    'log': log_text,
                }, out_subdir + '/Best_model_Acc_{:.2f}.tar'.format(result.acc_list[-1]))
            netG.train()

        if it % configs.save_interval == 0 and it:
            torch.save({
                'it': it + 1,
                'state_dict_G': netG.state_dict(),
                'state_dict_D': netD.state_dict(),
                'random_seed': configs.manualSeed,
                'log': log_text,
            }, out_subdir + '/Iter_{:d}.tar'.format(it))
            cprint('Save model to ' + out_subdir + '/Iter_{:d}.tar'.format(it), 'red')


def eval_fakefeat_test(it, netG, dataset, param, result):
    gen_feat = np.zeros([0, param.X_dim])
    for i in range(dataset.test_cls_num):
        text_feat = np.tile(dataset.test_text_feature[i].astype('float32'), (configs.nSample, 1))
        text_feat = Variable(torch.from_numpy(text_feat)).to(configs.device)  # .cuda()
        z = Variable(torch.randn(configs.nSample, param.z_dim)).to(configs.device)  # .cuda()
        G_sample = netG(z, text_feat)
        gen_feat = np.vstack((gen_feat, G_sample.data.cpu().numpy()))

    # cosince predict K-nearest Neighbor
    sim = cosine_similarity(dataset.pfc_feat_data_test, gen_feat)
    idx_mat = np.argsort(-1 * sim, axis=1)
    label_mat = (idx_mat[:, 0:configs.Knn] / configs.nSample).astype(int)
    preds = np.zeros(label_mat.shape[0])
    for i in range(label_mat.shape[0]):
        (values, counts) = np.unique(label_mat[i], return_counts=True)
        preds[i] = values[np.argmax(counts)]

    # produce acc
    label_T = np.asarray(dataset.labels_test)
    acc = (preds == label_T).mean() * 100

    result.acc_list += [acc]
    result.iter_list += [it]
    result.save_model = False
    if acc > result.best_acc:
        result.best_acc = acc
        result.best_iter = it
        result.save_model = True
    print("{}nn Classifier: ".format(configs.Knn))
    print("Accuracy is {:.4}%".format(acc))


""" Generalized ZSL"""


def eval_fakefeat_GZSL(it, netG, dataset, param, result):
    gen_feat = np.zeros([0, param.X_dim])
    for i in range(dataset.train_cls_num):
        text_feat = np.tile(dataset.train_text_feature[i].astype('float32'), (configs.nSample, 1))
        text_feat = Variable(torch.from_numpy(text_feat)).to(configs.device)  # .cuda()
        z = Variable(torch.randn(configs.nSample, param.z_dim)).to(configs.device)  # .cuda()
        G_sample = netG(z, text_feat)
        gen_feat = np.vstack((gen_feat, G_sample.data.cpu().numpy()))

    for i in range(dataset.test_cls_num):
        text_feat = np.tile(dataset.test_text_feature[i].astype('float32'), (configs.nSample, 1))
        text_feat = Variable(torch.from_numpy(text_feat)).to(configs.device)  # .cuda()
        z = Variable(torch.randn(configs.nSample, param.z_dim)).to(configs.device)  # .cuda()
        G_sample = netG(z, text_feat)
        gen_feat = np.vstack((gen_feat, G_sample.data.cpu().numpy()))

    visual_pivots = [gen_feat[i * configs.nSample:(i + 1) * configs.nSample].mean(0) \
                     for i in range(dataset.train_cls_num + dataset.test_cls_num)]
    visual_pivots = np.vstack(visual_pivots)

    """collect points for gzsl curve"""

    acc_S_T_list, acc_U_T_list = list(), list()
    seen_sim = cosine_similarity(dataset.pfc_feat_data_train, visual_pivots)
    unseen_sim = cosine_similarity(dataset.pfc_feat_data_test, visual_pivots)
    for GZSL_lambda in np.arange(-2, 2, 0.01):
        tmp_seen_sim = copy.deepcopy(seen_sim)
        tmp_seen_sim[:, dataset.train_cls_num:] += GZSL_lambda
        pred_lbl = np.argmax(tmp_seen_sim, axis=1)
        acc_S_T_list.append((pred_lbl == np.asarray(dataset.labels_train)).mean())

        tmp_unseen_sim = copy.deepcopy(unseen_sim)
        tmp_unseen_sim[:, dataset.train_cls_num:] += GZSL_lambda
        pred_lbl = np.argmax(tmp_unseen_sim, axis=1)
        acc_U_T_list.append((pred_lbl == (np.asarray(dataset.labels_test) + dataset.train_cls_num)).mean())

    auc_score = integrate.trapz(y=acc_S_T_list, x=acc_U_T_list)

    result.acc_list += [auc_score]
    result.iter_list += [it]
    result.save_model = False
    if auc_score > result.best_acc:
        result.best_acc = auc_score
        result.best_iter = it
        result.save_model = True
    print("AUC Score is {:.4}".format(auc_score))


class Result(object):
    def __init__(self):
        self.best_acc = 0.0
        self.best_iter = 0.0
        self.acc_list = []
        self.iter_list = []
        self.save_model = False


def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        init.xavier_normal(m.weight.data)
        init.constant(m.bias, 0.0)


def reset_grad(nets):
    for net in nets:
        net.zero_grad()


def label2mat(labels, y_dim):
    c = np.zeros([labels.shape[0], y_dim])
    for idx, d in enumerate(labels):
        c[idx, d] = 1
    return c


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(configs.batchsize, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(configs.device)  # .cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(configs.device)  # .cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates, _ = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(configs.device),  # .cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * configs.GP_LAMBDA
    return gradient_penalty


if __name__ == "__main__":
    train()

