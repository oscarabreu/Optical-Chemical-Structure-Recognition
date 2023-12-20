# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import sys
sys.path.append("..")

import os
import time
import argparse
import datetime
import numpy as np
from pre_transformer import Transformer
from data import SMILES_build_loader_125wan
from timm.utils import AverageMeter
from transformer import make_std_mask, subsequent_mask
from config import get_config
from models import build_model
from data import *
from lr_scheduler import build_scheduler
from nltk.translate.bleu_score import corpus_bleu
from optimizer import build_optimizer
from logger import create_logger
from model_utils import *
import argparse

import deepsmiles
from eval import Swin_Evaluate

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch import nn
from torch.autograd import Variable
from torch.cuda import amp
from utils import throughput

decoder_dim = 256  # dimension of decoder RNN
dropout = 0.1
ff_dim = 2048
num_head = 8
encoder_num_layer = 6
decoder_num_layer = 6
max_len = 277
decoder_lr = 5e-4
best_acc = 0.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        targets_onehot = class_mask.scatter_(1, ids.data, 1.)
        positive_label_mask = torch.eq(targets_onehot, torch.Tensor([1]).cuda())
        sigmoid_cross_entropy = self.loss(inputs, targets_onehot)
        probs = torch.sigmoid(inputs)
        probs_gt = torch.where(positive_label_mask, probs, 1.0 - probs)
        # With small gamma, the implementation could produce NaN during back prop.
        modulator = torch.pow(1.0 - probs_gt, self.gamma)
        loss = modulator * sigmoid_cross_entropy
        weighted_loss = torch.where(positive_label_mask, self.alpha * loss,
                                 (1.0 - self.alpha) * loss)

        return weighted_loss.sum()


def acc(pred_data, test_data):
    error_ids = []
    sum_count = len(pred_data)
    count = 0
    notin_count = 0
    for i in range(sum_count):
        if pred_data[i] == test_data[i][0]:
                count += 1
        else:
            error_ids.append(i)
            notin_count+=1
    return count/sum_count*1.0,error_ids

def greedy_decode(decoder, memory, max_len, start_symbol, end_symbol):
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long)
    logits = None
    for i in range(max_len-1):
        logit = decoder(Variable(ys).to(device), memory,
                           Variable(subsequent_mask(ys.size(1)).type(torch.long)).to(device))
        prob = nn.Softmax(dim=-1)(logit[0][i])
        _, next_word = torch.max(prob, dim=-1)
        next_word = next_word.data
        ys = torch.cat([ys,
                        torch.ones(1, 1).fill_(next_word).type(torch.long)], dim=1)
        if i == max_len - 2:
            logits = logit
    return ys, logits

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', default='Model/configs/swin_large_patch4_window7_224.yaml', type=str, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    parser.add_argument("--test_dir", default='../../Data/125wan/125wan_shuffle_DeepSMILES_test.pkl', type=str,
                        help='direction for eval_dataset')

    args, _ = parser.parse_known_args()

    config = get_config(args)

    return args, config

def main(config) :
    dir = '../../Data/125wan/125wan_shuffle_DeepSMILES'
    global best_acc, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, word_map
    word_map_file = '../../Data/125wan/125wan_shuffle_DeepSMILES_word_map'
    word_map = torch.load(word_map_file)
    data_loader_train, data_loader_val, _, mixup_fn = SMILES_build_loader_125wan(config, dir)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    if config.EVAL_MODE:
        tag = False
    else:
        tag = True
    decoder = Transformer(dim=decoder_dim, ff_dim=ff_dim, num_head=num_head,
                          encoder_num_layer=encoder_num_layer,
                          decoder_num_layer=decoder_num_layer,
                          vocab_size=len(word_map), max_len=max_len,
                          drop_rate=dropout, tag=tag)

    decoder = decoder.to(device)
    decoder_optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                          lr=decoder_lr, weight_decay=5e-6, eps=2e-7)
    encoder = build_model(config, tag=tag)
    encoder.to(device)
    encoder_optimizer = build_optimizer(config, encoder)

    encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    decoder = torch.nn.parallel.DistributedDataParallel(decoder, device_ids=[config.LOCAL_RANK],
                                                      broadcast_buffers=False)
    encoder_without_ddp = encoder.module
    decoder_without_ddp = decoder.module
    encoder_parameters = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    decoder_parameters = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    logger.info(f"encoder number of params: {encoder_parameters}")
    logger.info(f"decoder number of params: {decoder_parameters}")

    if hasattr(encoder_without_ddp, 'flops'):
        flops = encoder_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    encoder_lr_scheduler = build_scheduler(config, encoder_optimizer, len(data_loader_train))
    decoder_lr_scheduler = torch.optim.lr_scheduler.StepLR(decoder_optimizer,10)

    criterion = FocalLoss()
    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy, best_acc= load_checkpoint(config, encoder_without_ddp, encoder_optimizer, decoder_without_ddp, decoder_optimizer, encoder_lr_scheduler, decoder_lr_scheduler,logger)

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, encoder, logger)
        return
    if config.EVAL_MODE:
        Swin_Evaluate(config.MODEL.RESUME.split('/')[-1], 1, encoder_without_ddp, decoder_without_ddp,config.TEST_DIR)
        return
    logger.info("Start training")
    start_time = time.time()
    scaler = amp.GradScaler()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)
        train_one_epoch(config, encoder, encoder_optimizer, decoder, decoder_optimizer, criterion, data_loader_train, epoch, mixup_fn, encoder_lr_scheduler,scaler)
        decoder_lr_scheduler.step()
        recent_acc = validate_with_gold(val_loader=data_loader_val,
                                          encoder=encoder,
                                          decoder=decoder,
                                          criterion=criterion)

        is_best = recent_acc > best_acc
        best_acc = max(recent_acc, best_acc)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, encoder_without_ddp, encoder_optimizer, decoder_without_ddp, decoder_optimizer, max_accuracy, encoder_lr_scheduler,decoder_lr_scheduler, logger, best_acc, is_best)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))



def train_one_epoch(config, encoder, encoder_optimizer, decoder, decoder_optimizer, criterion, train_loader, epoch, mixup_fn, encoder_lr_scheduler,scaler):
    encoder.train()
    decoder.train()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    num_steps = len(train_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    encoder_norm_meter = AverageMeter()
    decoder_norm_meter = AverageMeter()
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    top1accs = AverageMeter()  # top5 accuracy
    top2accs = AverageMeter()
    top3accs = AverageMeter()
    top4accs = AverageMeter()
    top5accs = AverageMeter()  # top5 accuracy


    start_batch = time.time()
    start = time.time()
    end = time.time()
    for idx, (imgs, caps, caplens) in enumerate(train_loader):
        imgs = imgs.cuda(non_blocking=True)
        caps = caps.cuda(non_blocking=True)
        caplens = caplens.cuda(non_blocking=True)
        caps_np = caps.cpu().tolist()
        caplens_np = caplens.squeeze().cpu().tolist()
        caps_input, caps_target, decode_lengths, tgt_pad_mask = [], [], [], []
        for index, caplen in enumerate(caplens_np):
            caps_input.append(caps_np[index][:caplen - 1] + caps_np[index][caplen:])  # remove <end>
            caps_target += caps_np[index][
                           1:caplen]  # Removed <start> and extra pad after it
            decode_lengths.append(caplen - 1)  #  This is exactly the normal length of the sequence, with <start> and <end> removed

        caps_input = torch.tensor(caps_input).cuda()
        targets = torch.tensor(caps_target, dtype=torch.long).cuda()  # Level to one dimension
        decode_lengths = np.array(decode_lengths)
        tgt_pad_mask = make_std_mask(caps_input, pad=0)  # Generate mask
        with amp.autocast(True):
            imgs = encoder(imgs)
            scores = decoder(caps_input, imgs, tgt_pad_mask)  # scores:[16,81,256]  _:[16,81]
            scores_packed = None
            for index, decode_len in enumerate(decode_lengths):  # Removing the model predicted excess relative to correct labeling
                if scores_packed is None:
                    scores_packed = scores[index, :decode_len, :]  # [..,256]
                else:
                    scores_packed = torch.cat([scores_packed, scores[index, :decode_len, :]], dim=0)
            scores = scores_packed  # [635, 256]
            loss = criterion(scores, targets)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        scaler.scale(loss).backward()
        if config.TRAIN.CLIP_GRAD:
            encoder_grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.TRAIN.CLIP_GRAD)
            decoder_grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), config.TRAIN.CLIP_GRAD)
        else:
            encoder_grad_norm = get_grad_norm(encoder.parameters())
            decoder_grad_norm = get_grad_norm(decoder.parameters())
        scaler.step(encoder_optimizer)
        scaler.step(decoder_optimizer)
        scaler.update()
        encoder_lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        encoder_norm_meter.update(encoder_grad_norm)
        decoder_norm_meter.update(decoder_grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()
        # Keep track of metrics
        acc1 = accuracy(scores, targets, 1)
        acc2 = accuracy(scores, targets, 2)
        acc3 = accuracy(scores, targets, 3)
        acc4 = accuracy(scores, targets, 4)
        acc5 = accuracy(scores, targets, 5)
        top1accs.update(acc1, sum(decode_lengths))
        top2accs.update(acc2, sum(decode_lengths))
        top3accs.update(acc3, sum(decode_lengths))
        top4accs.update(acc4, sum(decode_lengths))
        top5accs.update(acc5, sum(decode_lengths))
        loss_meter.update(loss.item(), 1)
        batch_time.update(time.time() - start_batch)
        start_batch = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = encoder_optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                           'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                           'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                           'Batch_Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                           'Top-1 Acc {top1.val:.3f} ({top1.avg:.3f})\t'
                           'Top-2 Acc {top2.val:.3f} ({top2.avg:.3f})\t'
                           'Top-3 Acc {top3.val:.3f} ({top3.avg:.3f})\t'
                           'Top-4 Acc {top4.val:.3f} ({top4.avg:.3f})\t'
                           'Top-5 Acc {top5.val:.3f} ({top5.avg:.3f})\n'.format(epoch, idx, num_steps,
                                                                                batch_time=batch_time,
                                                                                data_time=data_time, loss=loss_meter,
                                                                                top1=top1accs,
                                                                                top2=top2accs,
                                                                                top3=top3accs,
                                                                                top4=top4accs,
                                                                                top5=top5accs))
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate_with_gold(encoder, decoder,  criterion, val_loader):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    encoder.eval()
    decoder.eval()  # eval mode (no dropout or batchnorm)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1accs = AverageMeter()  # top5 accuracy
    top2accs = AverageMeter()
    top3accs = AverageMeter()
    top4accs = AverageMeter()
    top5accs = AverageMeter()  # top5 accuracy

    batch_start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens) in enumerate(val_loader):
            if i > 10000:
                break
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)


            caps_np = caps.cpu().tolist()
            caplens_np = caplens.squeeze().cpu().tolist()
            caps_input, caps_target, decode_lengths, tgt_pad_mask = [], [], [], []
            for index, caplen in enumerate(caplens_np):
                caps_input.append(
                    caps_np[index][:caplen - 1] + caps_np[index][caplen:])
                caps_target += caps_np[index][
                               1:caplen]
                decode_lengths.append(caplen - 1)

            caps_input = torch.tensor(caps_input).to(
                device)
            targets = torch.tensor(caps_target, dtype=torch.long).to(device)
            decode_lengths = np.array(decode_lengths)
            tgt_pad_mask = make_std_mask(caps_input,
                                         pad=0)
            with amp.autocast(True):
                imgs = encoder(imgs)
                logits = decoder(caps_input, imgs, tgt_pad_mask)
                scores_packed = None
                for index, decode_len in enumerate(decode_lengths):
                    if scores_packed is None:
                        scores_packed = logits[index, :decode_len, :]
                    else:
                        scores_packed = torch.cat([scores_packed, logits[index, :decode_len, :]], dim=0)
                scores = scores_packed
                loss = criterion(scores, targets)


            losses.update(loss.item(), 1)
            acc1 = accuracy(scores_packed, targets, 1)
            acc2 = accuracy(scores_packed, targets, 2)
            acc3 = accuracy(scores_packed, targets, 3)
            acc4 = accuracy(scores_packed, targets, 4)
            acc5 = accuracy(scores_packed, targets, 5)
            top1accs.update(acc1, sum(decode_lengths))
            top2accs.update(acc2, sum(decode_lengths))
            top3accs.update(acc3, sum(decode_lengths))
            top4accs.update(acc4, sum(decode_lengths))
            top5accs.update(acc5, sum(decode_lengths))
            batch_time.update(time.time() - batch_start)
            batch_start=time.time()

            if i % 100 == 0:  # print_freq=15
                logger.info('Validation with gold: [{0}/{1}]\t'
                               'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                               'Batch_ValLoss {loss.val:.4f} ({loss.avg:.4f})\t'
                               'Top-1 Acc {top1.val:.3f} ({top1.avg:.3f})\t'
                               'Top-2 Acc {top2.val:.3f} ({top2.avg:.3f})\t'
                               'Top-3 Acc {top3.val:.3f} ({top3.avg:.3f})\t'
                               'Top-4 Acc {top4.val:.3f} ({top4.avg:.3f})\t'
                               'Top-5 Acc {top5.val:.3f} ({top5.avg:.3f})\t\n'.format(i, len(val_loader),
                                                                                      batch_time=batch_time,
                                                                                      loss=losses,
                                                                                      top1=top1accs,
                                                                                      top2=top2accs,
                                                                                      top3=top3accs,
                                                                                      top4=top4accs,
                                                                                      top5=top5accs))

                print('Validation with gold: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Batch_ValLoss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-1 Acc {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Top-2 Acc {top2.val:.3f} ({top2.avg:.3f})\t'
                      'Top-3 Acc {top3.val:.3f} ({top3.avg:.3f})\t'
                      'Top-4 Acc {top4.val:.3f} ({top4.avg:.3f})\t'
                      'Top-5 Acc {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
                                                                           batch_time=batch_time,
                                                                           loss=losses,
                                                                           top1=top1accs,
                                                                           top2=top2accs,
                                                                           top3=top3accs,
                                                                           top4=top4accs,
                                                                           top5=top5accs))

            for index, decode_len in enumerate(decode_lengths):
                _, preds = torch.max(logits[index, :decode_len, :], dim=-1)
                hypotheses.append(preds.tolist())

                # References

            img_captions = list(
                map(lambda c: [[w for w in c if w not in {word_map['<start>'], word_map['<pad>']}]],
                    caps_np))  # remove <start> and pads
            references.extend(img_captions)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
        bleu2 = corpus_bleu(references, hypotheses, weights=(0, 1, 0, 0))
        bleu3 = corpus_bleu(references, hypotheses, weights=(0, 0, 1, 0))
        bleu4 = corpus_bleu(references, hypotheses, weights=(0, 0, 0, 1))
        bleu_avg = corpus_bleu(references, hypotheses)
        acc_score, error_ids = acc(hypotheses, references)
        logger.info(
            '\n Validation with gold * LOSS - {loss.avg:.3f}, TOP-1 ACCURACY - {top1.avg:.3f},TOP-2 ACCURACY - {top2.avg:.3f},TOP-3 ACCURACY - {top3.avg:.3f},TOP-4 ACCURACY - {top4.avg:.3f},TOP-5 ACCURACY - {top5.avg:.3f}, BLEU - {bleu1}-{bleu2}-{bleu3}-{bleu4}-{bleu_avg}, ACC - {acc}\n'.format(
                loss=losses,
                top1=top1accs,
                top2=top2accs,
                top3=top3accs,
                top4=top4accs,
                top5=top5accs,
                bleu1=bleu1, bleu2=bleu2, bleu3=bleu3, bleu4=bleu4, bleu_avg=bleu_avg, acc=acc_score))

        print(
            '\n Validation with gold * LOSS - {loss.avg:.3f}, TOP-1 ACCURACY - {top1.avg:.3f},TOP-2 ACCURACY - {top2.avg:.3f},TOP-3 ACCURACY - {top3.avg:.3f},TOP-4 ACCURACY - {top4.avg:.3f},TOP-5 ACCURACY - {top5.avg:.3f}, BLEU - {bleu1}-{bleu2}-{bleu3}-{bleu4}-{bleu_avg}, ACC - {acc}\n'.format(
                loss=losses,
                top1=top1accs,
                top2=top2accs,
                top3=top3accs,
                top4=top4accs,
                top5=top5accs,
                bleu1=bleu1, bleu2=bleu2, bleu3=bleu3, bleu4=bleu4, bleu_avg=bleu_avg, acc=acc_score))

    return acc_score


@torch.no_grad()
def test(encoder,  decoder, criterion, test_loader):
    """
    Performs one epoch's validation.

    :param test_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1accs = AverageMeter()  # top5 accuracy
    top2accs = AverageMeter()
    top3accs = AverageMeter()
    top4accs = AverageMeter()
    top5accs = AverageMeter()  # top5 accuracy

    batch_start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(test_loader):
            if i > 500:
                break
            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.

            imgs = encoder(imgs)

            y_preds, logits = greedy_decode(decoder, imgs, max_len=caps.shape[1],
                                            start_symbol=word_map['<start>'], end_symbol=word_map['<end>'])

            cap_list = caps.squeeze().cpu().tolist()
            caplen = caplens.squeeze().cpu().tolist()
            decode_lengths = [caplen - 1]
            targets = torch.tensor(cap_list[1:caplen], dtype=torch.long).to(device)

            # scores_packed = pack_padded_sequence(probs, decode_lengths, batch_first=True, enforce_sorted=False)
            scores_packed = logits[0, :caplen - 1, :]
            loss = criterion(scores_packed, targets)

            losses.update(loss.item(), sum(decode_lengths))
            acc1 = accuracy(scores_packed, targets, 1)
            acc2 = accuracy(scores_packed, targets, 2)
            acc3 = accuracy(scores_packed, targets, 3)
            acc4 = accuracy(scores_packed, targets, 4)
            acc5 = accuracy(scores_packed, targets, 5)
            top1accs.update(acc1, sum(decode_lengths))
            top2accs.update(acc2, sum(decode_lengths))
            top3accs.update(acc3, sum(decode_lengths))
            top4accs.update(acc4, sum(decode_lengths))
            top5accs.update(acc5, sum(decode_lengths))
            batch_time.update(time.time() -batch_start)
            batch_start = time.time()

            if i % 100 == 0:  # print_freq=15
                logger.info('Test: [{0}/{1}]\t'
                               'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                               'TestLoss {loss.val:.4f} ({loss.avg:.4f})\t'
                               'Top-1 Acc {top1.val:.3f} ({top1.avg:.3f})\t'
                               'Top-2 Acc {top2.val:.3f} ({top2.avg:.3f})\t'
                               'Top-3 Acc {top3.val:.3f} ({top3.avg:.3f})\t'
                               'Top-4 Acc {top4.val:.3f} ({top4.avg:.3f})\t'
                               'Top-5 Acc {top5.val:.3f} ({top5.avg:.3f})\t\n'.format(i, len(test_loader),
                                                                                      batch_time=batch_time,
                                                                                      loss=losses, top1=top1accs,
                                                                                      top2=top2accs, top3=top3accs,
                                                                                      top4=top4accs, top5=top5accs))
                # if acc1<70.0:
                #     print(caps.cpu())
                print('Test: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'TestLoss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-1 Acc {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Top-2 Acc {top2.val:.3f} ({top2.avg:.3f})\t'
                      'Top-3 Acc {top3.val:.3f} ({top3.avg:.3f})\t'
                      'Top-4 Acc {top4.val:.3f} ({top4.avg:.3f})\t'
                      'Top-5 Acc {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(test_loader), batch_time=batch_time,
                                                                           loss=losses, top1=top1accs, top2=top2accs,
                                                                           top3=top3accs, top4=top4accs, top5=top5accs))

            # References
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_packed, dim=-1)

            preds = preds.tolist()
            temp_preds = list()
            start = 0
            for j in range(len(decode_lengths)):
                end = start + int(decode_lengths[j])
                temp_preds.append(preds[start:end])
                start = end
                # temp_preds.append(preds[j][1:decode_lengths[j]]) # remove start
            # preds = temp_preds
            hypotheses.extend(temp_preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
        bleu2 = corpus_bleu(references, hypotheses, weights=(0, 1, 0, 0))
        bleu3 = corpus_bleu(references, hypotheses, weights=(0, 0, 1, 0))
        bleu4 = corpus_bleu(references, hypotheses, weights=(0, 0, 0, 1))
        bleu_avg = corpus_bleu(references, hypotheses)
        acc_score, error_ids = acc(hypotheses, references)
        logger.info(
            '\n Test * LOSS - {loss.avg:.3f}, TOP-1 ACCURACY - {top1.avg:.3f},TOP-2 ACCURACY - {top2.avg:.3f},TOP-3 ACCURACY - {top3.avg:.3f},TOP-4 ACCURACY - {top4.avg:.3f},TOP-5 ACCURACY - {top5.avg:.3f}, BLEU - {bleu1}-{bleu2}-{bleu3}-{bleu4}-{bleu_avg}, ACC - {acc}\n'.format(
                loss=losses,
                top1=top1accs,
                top2=top2accs,
                top3=top3accs,
                top4=top4accs,
                top5=top5accs,
                bleu1=bleu1, bleu2=bleu2, bleu3=bleu3, bleu4=bleu4, bleu_avg=bleu_avg, acc=acc_score))

        print(
            '\n Test * LOSS - {loss.avg:.3f}, TOP-1 ACCURACY - {top1.avg:.3f},TOP-2 ACCURACY - {top2.avg:.3f},TOP-3 ACCURACY - {top3.avg:.3f},TOP-4 ACCURACY - {top4.avg:.3f},TOP-5 ACCURACY - {top5.avg:.3f}, BLEU - {bleu1}-{bleu2}-{bleu3}-{bleu4}-{bleu_avg}, ACC - {acc}\n'.format(
                loss=losses,
                top1=top1accs,
                top2=top2accs,
                top3=top3accs,
                top4=top4accs,
                top5=top5accs,
                bleu1=bleu1, bleu2=bleu2, bleu3=bleu3, bleu4=bleu4, bleu_avg=bleu_avg, acc=acc_score))

    return bleu4

if __name__ == '__main__':
    _, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    config.defrost()
    config.MODEL.NAME = 'Swinfocalloss'
    config.OUTPUT = os.path.join('output', config.MODEL.NAME, config.TAG)
    config.freeze()
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()
    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())

    main(config)
