import argparse
import os
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from pytorch_transformers import *
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from read_coarse_fine import *

import copy
import pickle
import time

from vat import VATLoss
from vat_fine import Fine_vat
import torch.optim as optim
from mixtext import MixText

parser = argparse.ArgumentParser(description='PyTorch Base Models')

parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=4, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--batch-size-u', default=8, type=int, metavar='N',
                    help='train batchsize')

parser.add_argument('--lrmain', '--learning-rate-bert', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate for bert')
parser.add_argument('--lrlast', '--learning-rate-model', default=0.001, type=float,
                    metavar='LR', help='initial learning rate for models')

parser.add_argument('--gpu', default='0,1,2,3,4,5,6,7', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--n-labeled', type=int, default=10,
                    help='Number of labeled data')
parser.add_argument('--val-iteration', type=int, default=2000)


parser.add_argument('--mix-option', default=True, type=bool, metavar='N',
                    help='mix option')
parser.add_argument('--train_aug', default=False, type=bool, metavar='N',
                    help='aug for training data')


parser.add_argument('--model', type=str, default='bert-base-uncased',
                    help='pretrained model')

parser.add_argument('--data-path', type=str, default='ag_news_csv/',
                    help='path to data folders')


parser.add_argument('--cycle', type=int, default=5,
                    help='AL cycle')

parser.add_argument('--num_train', type=int, default=112000,
                    help='train_samples')

parser.add_argument('--addnum', type=int, default=200,
                    help='initial num')


parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")

parser.add_argument('--xi', type=float, default=1, metavar='XI',
            help='hyperparameter of VAT (default: 0.1)')
parser.add_argument('--eps', type=float, default=0.01, metavar='EPS',
        help='hyperparameter of VAT (default: 1.0)')
parser.add_argument('--ip', type=int, default=1, metavar='IP',
        help='hyperparameter of VAT (default: 1)')

parser.add_argument('--temp-change', default=1000000, type=int)

parser.add_argument('--T', default=0.5, type=float,
                    help='temperature for sharpen function')

parser.add_argument('--co', default=False, type=bool, metavar='N',
                    help='set a random choice between mix and unmix during training')

parser.add_argument('--alpha', default=16, type=float,
                    help='alpha for beta distribution')

parser.add_argument('--separate-mix', default=False, type=bool, metavar='N',
                    help='mix separate from labeled data and unlabeled data')

parser.add_argument('--mix-layers-set', nargs='+',
                    default=[7,9,12], type=int, help='define mix layer set')

parser.add_argument('--mix-method', default=0, type=int, metavar='N',
                    help='mix method, set different mix method')

parser.add_argument('--margin', default=0.7, type=float, metavar='N',
                    help='margin for hinge loss')

parser.add_argument('--lambda-u', default=1, type=float,
                    help='weight for consistency loss term of unlabeled data')

parser.add_argument('--lambda-u-hinge', default=0, type=float,
                    help='weight for hinge loss term of unlabeled data')

parser.add_argument('--gamma', default=0.5, type=float,
                    help='weight to balance two criteria')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("gpu num: ", n_gpu)

best_acc = 0
total_steps = 0

def main():

    for trial in range(5):

        fine_loss = Fine_vat(xi=args.xi, eps=0.1, ip=args.ip)

        global best_acc
        train_labeled_set, train_unlabeled_set, coarse_fine, val_set, test_set, n_labels, initial_idxs = get_data(
            args.data_path, args.n_labeled, train_aug=args.train_aug)

        all_indices = set(np.arange(args.num_train))

        initial_indices = initial_idxs

        current_indices = list(initial_indices)

        train_loader = Data.DataLoader(train_labeled_set, batch_size=args.batch_size, 
                                    sampler=SubsetRandomSampler(initial_indices), 
                                    pin_memory=True)

        unlabeled_trainloader = Data.DataLoader(
            dataset=train_unlabeled_set, batch_size=args.batch_size_u, shuffle=True)

        test_loader  = Data.DataLoader(test_set, batch_size=512, pin_memory=True)

        val_loader = Data.DataLoader(val_set, batch_size=512)

        accuracies = []
        for cycle in range(args.cycle):

            unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
   
            unlabeled_loader = Data.DataLoader(coarse_fine, batch_size=512, 
                                                sampler=SubsetRandomSampler(unlabeled_indices), 
                                                pin_memory=True, num_workers=16)

 
            model = MixText(n_labels, args.mix_option).cuda()
            model = nn.DataParallel(model)
            optimizer = AdamW(
                [
                    {"params": model.module.bert.parameters(), "lr": args.lrmain},
                    {"params": model.module.linear.parameters(), "lr": args.lrlast},
                ])

            num_warmup_steps = math.floor(50)
            num_total_steps = args.val_iteration

            scheduler = None


            train_criterion = SemiLoss()
            criterion = nn.CrossEntropyLoss()

            test_accs = []
            best_acc = 0
            # Start training
            
            for epoch in range(args.epochs):

                train(train_loader, unlabeled_trainloader, model, optimizer,
                    scheduler, train_criterion, fine_loss, epoch, n_labels, args.train_aug)


                val_loss, val_acc = validate(
                    val_loader, model, criterion, epoch, mode='Valid Stats')

                print("epoch {}, val acc {}, val_loss {}".format(
                    epoch, val_acc, val_loss))


                test_loss, test_acc = validate(
                        test_loader, model, criterion, epoch, mode='Test Stats ')


            model.eval()          
            all_un_indice = []
            fine_uo = torch.tensor([]).cuda()
            fine_u1 = torch.tensor([]).cuda()
            fine_u2 = torch.tensor([]).cuda()
            var_out = torch.tensor([]).cuda()
            excout = torch.tensor([]).cuda()
            
            for ((inputs_u, inputs_u2, inputs_ori), (length_u, length_u2, length_ori), idx) in unlabeled_loader:
                inputs_u, inputs_u2, inputs_ori = inputs_u.cuda(), inputs_u2.cuda(), inputs_ori.cuda()     

                with torch.no_grad():
                    outputs_u, feature_adv = model(inputs_u)
                    outputs_u2, feature_adv2 = model(inputs_u2)
                    outputs_ori, feature_ori = model(inputs_ori)

                    outputs_u = F.softmax(outputs_u, dim=1)
                    outputs_u2 = F.softmax(outputs_u2, dim=1)
                    outputs_ori = F.softmax(outputs_ori, dim=1)

                    outputs_u = torch.unsqueeze(outputs_u, 1)
                    outputs_u2 = torch.unsqueeze(outputs_u2, 1)
                    outputs_ori = torch.unsqueeze(outputs_ori, 1)

                    output_var = torch.cat((outputs_u,outputs_u2,outputs_ori),1)
                    output_var = torch.var(output_var, 1)
                    output_var = torch.sum(output_var, -1)

                    var_out = torch.cat((var_out, output_var),0)

                    fine_u1 = torch.cat((fine_u1, feature_adv),0)
                    fine_u2 = torch.cat((fine_u2, feature_adv2),0)
                    excout = torch.cat((excout, feature_ori),0)

                all_un_indice.extend(idx)

            vat_loss = VATLoss(xi=args.xi, eps=args.eps, ip=args.ip)

            _, lds_each1 = vat_loss(model.module.linear, fine_u1)
            _, lds_each2 = vat_loss(model.module.linear, fine_u2)

            lds_each1 = lds_each1.view(-1)
            lds_each2 = lds_each2.view(-1)

            lds_each = lds_each1 + lds_each2

            ldsperc = lds_each.argsort().argsort().float()/len(lds_each)

            varperc = var_out.argsort().argsort().float()/len(var_out)

            gamma = args.gamma

            finalperc = (1 - gamma) * ldsperc + gamma * varperc

            _, querry_indices = torch.topk(finalperc, int(1500))
            querry_indices = querry_indices.cpu()
            querry_pool_indices = np.asarray(all_un_indice)[querry_indices]

            sim_excout = excout[querry_indices]
            input1 = torch.nn.functional.normalize(sim_excout)
            input2 = input1.T
            sim = torch.mm(input1, input2)
            sim = torch.sum(sim, 1)
            sim = sim.cpu()
            excout = model.module.linear(excout)
            excout = F.softmax(excout, dim=1)
            excout = excout.cpu()
            excout = excout.detach().numpy()
            entropy_list = -excout * np.log(excout)
            entropy_list = np.add.reduce(entropy_list, axis=1)
            entropy_list = torch.FloatTensor(entropy_list)
            entropy_list = entropy_list[querry_indices]
            entropy_list = entropy_list * sim
            _, querry_indices_en = torch.topk(entropy_list, int(190))
            sampled_indices = querry_pool_indices[querry_indices_en]
            
                
            current_indices = list(current_indices) + list(sampled_indices)


            train_loader = Data.DataLoader(train_labeled_set, batch_size=args.batch_size, 
                                    sampler=SubsetRandomSampler(current_indices), 
                                    pin_memory=True)


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, scheduler, criterion, fine_loss, epoch, n_labels, train_aug=False):
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    model.train()

    global total_steps
    global flag
    if flag == 0 and total_steps > args.temp_change:
        print('Change T!')
        args.T = 0.9
        flag = 1

    for batch_idx in range(args.val_iteration):

        total_steps += 1

        if not train_aug:
            try:
                inputs_x, targets_x, inputs_x_length, idx = labeled_train_iter.next()
            except:
                labeled_train_iter = iter(labeled_trainloader)
                inputs_x, targets_x, inputs_x_length, idx = labeled_train_iter.next()
        else:
            try:
                (inputs_x, inputs_x_aug), (targets_x, _), (inputs_x_length,
                                                           inputs_x_length_aug) = labeled_train_iter.next()
            except:
                labeled_train_iter = iter(labeled_trainloader)
                (inputs_x, inputs_x_aug), (targets_x, _), (inputs_x_length,
                                                           inputs_x_length_aug) = labeled_train_iter.next()
        try:
            (inputs_u, inputs_u2,  inputs_ori), (length_u,
                                                 length_u2,  length_ori) = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2, inputs_ori), (length_u,
                                                length_u2, length_ori) = unlabeled_train_iter.next()

        batch_size = inputs_x.size(0)
        batch_size_2 = inputs_ori.size(0)
        targets_x = torch.zeros(batch_size, n_labels).scatter_(
            1, targets_x.view(-1, 1), 1)

        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
        inputs_u = inputs_u.cuda()
        inputs_u2 = inputs_u2.cuda()
        inputs_ori = inputs_ori.cuda()

        with torch.no_grad(): 
            _, feature_adv = model(inputs_u)
            _, feature_adv2 = model(inputs_u2)
            # _, feature_ori = model(inputs_ori)

        adv1 = fine_loss(model.module.linear, feature_adv) 
        adv2 = fine_loss(model.module.linear, feature_adv2) 
        feature_adv = feature_adv+ adv1
        feature_adv2 = feature_adv2 + adv2

        with torch.no_grad(): 
            outputs_u = model.module.linear(feature_adv)
            outputs_u2 = model.module.linear(feature_adv2)

        mask = []

        with torch.no_grad():
            # Predict labels for unlabeled data.
            outputs_ori, _ = model(inputs_ori)

            # Based on translation qualities, choose different weights here.
            p = (0.5 * torch.softmax(outputs_u, dim=1) + 0.5 * torch.softmax(outputs_u2,
                                                                         dim=1) + 1 * torch.softmax(outputs_ori, dim=1)) / (2)
            # Do a sharpen here.
            pt = p**(1/args.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        mixed = 1

        if args.co:
            mix_ = np.random.choice([0, 1], 1)[0]
        else:
            mix_ = 1

        if mix_ == 1:
            l = np.random.beta(args.alpha, args.alpha)
            if args.separate_mix:
                l = l
            else:
                l = max(l, 1-l)
        else:
            l = 1

        mix_layer = np.random.choice(args.mix_layers_set, 1)[0]
        mix_layer = mix_layer - 1

        if not train_aug:
            all_inputs = torch.cat(
                [inputs_x, inputs_u, inputs_u2, inputs_ori, inputs_ori], dim=0)

            all_lengths = torch.cat(
                [inputs_x_length, length_u, length_u2, length_ori, length_ori], dim=0)

            all_targets = torch.cat(
                [targets_x, targets_u, targets_u, targets_u, targets_u], dim=0)

        else:
            all_inputs = torch.cat(
                [inputs_x, inputs_x_aug, inputs_u, inputs_u2, inputs_ori], dim=0)
            all_lengths = torch.cat(
                [inputs_x_length, inputs_x_length, length_u, length_u2, length_ori], dim=0)
            all_targets = torch.cat(
                [targets_x, targets_x, targets_u, targets_u, targets_u], dim=0)

        if args.separate_mix:
            idx1 = torch.randperm(batch_size)
            idx2 = torch.randperm(all_inputs.size(0) - batch_size) + batch_size
            idx = torch.cat([idx1, idx2], dim=0)

        else:
            idx1 = torch.randperm(all_inputs.size(0) - batch_size_2)
            idx2 = torch.arange(batch_size_2) + \
                all_inputs.size(0) - batch_size_2
            idx = torch.cat([idx1, idx2], dim=0)

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        length_a, length_b = all_lengths, all_lengths[idx]

        if args.mix_method == 0:
            # Mix sentences' hidden representations
            logits, _ = model(input_a, input_b, l, mix_layer)
            mixed_target = l * target_a + (1 - l) * target_b

        elif args.mix_method == 1:
            # Concat snippet of two training sentences, the snippets are selected based on l
            # For example: "I lova you so much" and "He likes NLP" could be mixed as "He likes NLP so much".
            # The corresponding labels are mixed with coefficient as well
            mixed_input = []
            if l != 1:
                for i in range(input_a.size(0)):
                    length1 = math.floor(int(length_a[i]) * l)
                    idx1 = torch.randperm(int(length_a[i]) - length1 + 1)[0]
                    length2 = math.ceil(int(length_b[i]) * (1-l))
                    if length1 + length2 > 256:
                        length2 = 256-length1 - 1
                    idx2 = torch.randperm(int(length_b[i]) - length2 + 1)[0]
                    try:
                        mixed_input.append(
                            torch.cat((input_a[i][idx1: idx1 + length1], torch.tensor([102]).cuda(), input_b[i][idx2:idx2 + length2], torch.tensor([0]*(256-1-length1-length2)).cuda()), dim=0).unsqueeze(0))
                    except:
                        print(256 - 1 - length1 - length2,
                              idx2, length2, idx1, length1)

                mixed_input = torch.cat(mixed_input, dim=0)

            else:
                mixed_input = input_a

            logits, _ = model(mixed_input)
            mixed_target = l * target_a + (1 - l) * target_b

        elif args.mix_method == 2:
            # Concat two training sentences
            # The corresponding labels are averaged
            if l == 1:
                mixed_input = []
                for i in range(input_a.size(0)):
                    mixed_input.append(
                        torch.cat((input_a[i][:length_a[i]], torch.tensor([102]).cuda(), input_b[i][:length_b[i]], torch.tensor([0]*(512-1-int(length_a[i])-int(length_b[i]))).cuda()), dim=0).unsqueeze(0))

                mixed_input = torch.cat(mixed_input, dim=0)
                logits, _ = model(mixed_input, sent_size=512)

                mixed = 0
                mixed_target = (target_a + target_b)/2
            else:
                mixed_input = input_a
                mixed_target = target_a
                logits, _ = model(mixed_input, sent_size=256)
                mixed = 1

        Lx, Lu, w, Lu2, w2 = criterion(logits[:batch_size], mixed_target[:batch_size], logits[batch_size:-batch_size_2],
                                       mixed_target[batch_size:-batch_size_2], logits[-batch_size_2:], epoch+batch_idx/args.val_iteration, mixed)

        if mix_ == 1:
            loss = Lx + w * Lu
        else:
            loss = Lx + w * Lu + w2 * Lu2


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        if batch_idx % 1000 == 0:
            print("epoch {}, step {}, loss {}, Lx {}, Lu {}, Lu2 {}".format(
                epoch, batch_idx, loss.item(), Lx.item(), Lu.item(), Lu2.item()))


def validate(valloader, model, criterion, epoch, mode):
    model.eval()
    with torch.no_grad():
        loss_total = 0
        total_sample = 0
        acc_total = 0
        correct = 0

        for batch_idx, (inputs, targets, length, idx) in enumerate(valloader):
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)

            _, predicted = torch.max(outputs.data, 1)

            # if batch_idx == 0:
            #     print("Sample some true labeles and predicted labels")
            #     print(predicted[:20])
            #     print(targets[:20])

            correct += (np.array(predicted.cpu()) ==
                        np.array(targets.cpu())).sum()
            loss_total += loss.item() * inputs.shape[0]
            total_sample += inputs.shape[0]

        acc_total = correct/total_sample
        loss_total = loss_total/total_sample

    return loss_total, acc_total


def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, outputs_u_2, epoch, mixed=1):

        if args.mix_method == 0 or args.mix_method == 1:

            Lx = - torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))

            probs_u = torch.softmax(outputs_u, dim=1)

            Lu = F.kl_div(probs_u.log(), targets_u, None, None, 'batchmean')

            Lu2 = torch.mean(torch.clamp(torch.sum(-F.softmax(outputs_u, dim=1)
                                                   * F.log_softmax(outputs_u, dim=1), dim=1) - args.margin, min=0))

        elif args.mix_method == 2:
            if mixed == 0:
                Lx = - \
                    torch.mean(torch.sum(F.logsigmoid(
                        outputs_x) * targets_x, dim=1))

                probs_u = torch.softmax(outputs_u, dim=1)

                Lu = F.kl_div(probs_u.log(), targets_u,
                              None, None, 'batchmean')

                Lu2 = torch.mean(torch.clamp(args.margin - torch.sum(
                    F.softmax(outputs_u_2, dim=1) * F.softmax(outputs_u_2, dim=1), dim=1), min=0))
            else:
                Lx = - \
                    torch.mean(torch.sum(F.log_softmax(
                        outputs_x, dim=1) * targets_x, dim=1))

                probs_u = torch.softmax(outputs_u, dim=1)
                Lu = F.kl_div(probs_u.log(), targets_u,
                              None, None, 'batchmean')

                Lu2 = torch.mean(torch.clamp(args.margin - torch.sum(
                    F.softmax(outputs_u, dim=1) * F.softmax(outputs_u, dim=1), dim=1), min=0))

        return Lx, Lu, args.lambda_u * linear_rampup(epoch), Lu2, args.lambda_u_hinge * linear_rampup(epoch)


if __name__ == '__main__':
    main()
