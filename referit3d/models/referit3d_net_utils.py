"""
Utilities to analyze, train, test an 3d_listener.
"""

import torch
import numpy as np
import pandas as pd
import tqdm
import math
import sys
current_module = sys.modules[__name__]
import torch.nn.functional as F

from ..utils.evaluation import AverageMeter


def make_batch_keys(args, extras=None):
    """depending on the args, different data are used by the listener."""
    batch_keys = ['objects', 'tokens', 'target_pos', 'token_inds', 'token_num', 'context_size', 'obj_offset']  # all models use these
    batch_keys.extend(['context_objects','closest_context_objects','farthest_context_objects','rand_context_objects',\
        'context_offset','closest_context_offset','farthest_context_offset','rand_context_offset',\
        'feat_2d','coords_2d','word_split_inds'])
    if extras is not None:
        batch_keys += extras

    if args.obj_cls_alpha > 0:
        batch_keys.append('class_labels')

    if args.lang_cls_alpha > 0:
        batch_keys.append('target_class')

    return batch_keys


def single_epoch_train(model, data_loader, criteria, optimizer, device, pad_idx, args, epoch=None):
    """
    :param model:
    :param data_loader:
    :param criteria: (dict) holding all modules for computing the losses.
    :param optimizer:
    :param device:
    :param pad_idx: (int)
    :param args:
    :return:
    """
    metrics = dict()  # holding the losses/accuracies
    total_loss_mtr = AverageMeter()
    referential_loss_mtr = AverageMeter()
    obj_loss_mtr = AverageMeter()
    ref_acc_mtr = AverageMeter()
    cls_acc_mtr = AverageMeter()
    txt_acc_mtr = AverageMeter()

    # Set the model in training mode
    model.train()
    np.random.seed(args.random_seed+epoch)  # call this to change the sampling of the point-clouds, in a time-invariant way
    batch_keys = make_batch_keys(args)
    for batch in tqdm.tqdm(data_loader):
        # Move data to gpu
        for k in batch_keys:
            if k in batch:
                batch[k] = batch[k].to(device)

        if args.object_encoder == 'pnet':
            batch['objects'] = batch['objects'].permute(0, 1, 3, 2)

        # Forward pass
        res = model(batch)

        # Backward
        optimizer.zero_grad()
        all_losses = compute_losses(batch, res, criteria, args)
        total_loss = all_losses['total_loss']
        total_loss.backward()
        optimizer.step()

        # Update the loss and accuracy meters
        target = batch['target_pos']
        batch_size = target.size(0)  # B x N_Objects
        total_loss_mtr.update(total_loss.item(), batch_size)

        # referential_loss_mtr.update(all_losses['referential_loss'].item(), batch_size)
        # TODO copy the ref-loss to homogeneize the code
        referential_loss_mtr.update(all_losses['referential_loss'], batch_size)

        predictions = torch.argmax(res['logits'], dim=1)
        guessed_correctly = torch.mean((predictions == target).double()).item()
        ref_acc_mtr.update(guessed_correctly, batch_size)

        if args.obj_cls_alpha > 0:
            cls_b_acc, _ = cls_pred_stats(res['class_logits'], batch['class_labels'], ignore_label=pad_idx)
            cls_acc_mtr.update(cls_b_acc, batch_size)
            obj_loss_mtr.update(all_losses['obj_clf_loss'].item(), batch_size)

        if args.lang_cls_alpha > 0:
            batch_guess = torch.argmax(res['lang_logits'], -1)
            cls_b_acc = torch.mean((batch_guess == batch['target_class']).double())
            txt_acc_mtr.update(cls_b_acc, batch_size)

    metrics['train_total_loss'] = total_loss_mtr.avg
    metrics['train_referential_loss'] = referential_loss_mtr.avg
    metrics['train_obj_clf_loss'] = obj_loss_mtr.avg
    metrics['train_referential_acc'] = ref_acc_mtr.avg
    metrics['train_object_cls_acc'] = cls_acc_mtr.avg
    metrics['train_txt_cls_acc'] = txt_acc_mtr.avg
    return metrics

def contrastive_loss(feat1, feat2, obj_count, margin=0.1, max_margin=True, weight=10., reduction=True):
    sim_losses = 0. if reduction else []
    feat1 = F.normalize(feat1, p=2, dim=-1)
    feat2 = F.normalize(feat2, p=2, dim=-1)
    for b_i in range(feat1.shape[0]):
        feat_2d, feat_3d, num_obj = feat1[b_i,:,:], feat2[b_i,:,:], obj_count[b_i]
        feat_2d, feat_3d = feat_2d[:num_obj,:], feat_3d[:num_obj,:]
        cos_scores = feat_2d.mm(feat_3d.t())
        diagonal = cos_scores.diag().view(feat_2d.size(0), 1)
        d1 = diagonal.expand_as(cos_scores)
        d2 = diagonal.t().expand_as(cos_scores)
        # feat_3d retrieval
        cost_3d = (margin + cos_scores - d1).clamp(min=0)
        # feat2d retrieval
        cost_2d = (margin + cos_scores - d2).clamp(min=0)
        # clear diagonals
        I = (torch.eye(cos_scores.size(0), device=torch.device('cuda')) > .5)
        cost_3d = cost_3d.masked_fill_(I, 0)
        cost_2d = cost_2d.masked_fill_(I, 0)
        topk = min(3,int(cost_3d.shape[0]))
        cost_3d = (torch.topk(cost_3d, topk, dim=1)[0])
        cost_2d = (torch.topk(cost_2d, topk, dim=0)[0])
        if reduction: 
            batch_loss = torch.sum(cost_3d) + torch.sum(cost_2d)
            sim_losses = sim_losses + batch_loss
        else: 
            batch_loss = torch.mean(cost_3d) + torch.mean(cost_2d)
            sim_losses.append(batch_loss)
    if reduction: 
        return weight * sim_losses/(torch.sum(obj_count))
    else:
        return weight * torch.tensor(sim_losses, device=torch.device('cuda'))

def compute_losses(batch, res, criterion_dict, args):
    """Calculate the loss given the model logits and the criterion
    :param batch:
    :param res: dict of logits
    :param criterion_dict: dict of the criterion should have key names same as the logits
    :param args, argparse.Namespace
    :return: scalar loss value
    """
    # Get the object language classification loss and the object classification loss
    criterion = criterion_dict['logits']
    logits = res['logits']

    if args.s_vs_n_weight is not None:
        # total_loss = criterion(logits, batch)
        total_loss = criterion_dict['logits_nondec'](logits, batch['target_pos'])
    else:
        total_loss = criterion(logits, batch['target_pos'])

    sim_loss_type = 'contrastive'
    if sim_loss_type == 'contrastive':
        simloss = getattr(current_module, 'contrastive_loss')

    if args.context_2d=='unaligned':
        ## 2D-lang align loss
        if args.s_vs_n_weight is not None:
            # vg2d_loss = criterion(res['logits_2D'], batch)
            vg2d_loss = criterion_dict['logits_nondec'](logits, batch['target_pos'])
        else:
            vg2d_loss = criterion(res['logits_2D'], batch['target_pos'])
        ## 2D-3D align loss; contra loss, cos
        feat_2d = res['mmt_obj_output_2D']
        feat_3d = res['mmt_obj_output']
        feat_texttoken_2d = res['mmt_texttoken2d_output']
        feat_texttoken_3d = res['mmt_texttoken3d_output']
        sim2d3d_loss = simloss(feat_2d, feat_3d, batch['context_size'], reduction=(args.s_vs_n_weight is None))
        total_loss = total_loss + vg2d_loss + sim2d3d_loss
    if args.s_vs_n_weight is not None:
        weights = torch.ones(total_loss.shape).to(total_loss.device)
        weights[batch['is_nr3d']] = 1./args.s_vs_n_weight
        total_loss = total_loss * weights
        total_loss = total_loss.sum() / len(total_loss)

    referential_loss = total_loss.item()
    obj_clf_loss = lang_clf_loss = 0

    if args.obj_cls_alpha > 0:
        criterion = criterion_dict['class_logits']
        obj_clf_loss = criterion(res['class_logits'].transpose(2, 1), batch['class_labels'])
        if args.s_vs_n_weight is not None:
            obj_clf_loss = torch.mean(obj_clf_loss,dim=1) * weights
            obj_clf_loss = obj_clf_loss.sum() / len(obj_clf_loss)
        total_loss += obj_clf_loss * args.obj_cls_alpha

    if args.lang_cls_alpha > 0:
        criterion = criterion_dict['lang_logits']
        lang_clf_loss = criterion(res['lang_logits'], batch['target_class'])
        if args.s_vs_n_weight is not None:
            lang_clf_loss = lang_clf_loss * weights
            lang_clf_loss = lang_clf_loss.sum() / len(lang_clf_loss)
        total_loss += lang_clf_loss * args.lang_cls_alpha

    return {'total_loss': total_loss, 'referential_loss': referential_loss,
            'obj_clf_loss': obj_clf_loss, 'lang_clf_loss': lang_clf_loss}


@torch.no_grad()
def evaluate_on_dataset(model, data_loader, criteria, device, pad_idx, args, randomize=False):
    # TODO post-deadline, can we replace this func with the train + a 'phase==eval' parameter?
    metrics = dict()  # holding the losses/accuracies
    total_loss_mtr = AverageMeter()
    referential_loss_mtr = AverageMeter()
    obj_loss_mtr = AverageMeter()
    ref_acc_mtr = AverageMeter()
    cls_acc_mtr = AverageMeter()
    txt_acc_mtr = AverageMeter()

    # Set the model in training mode
    model.eval()

    assert(randomize==False)
    if randomize:
        np.random.seed()  # call this to change the sampling of the point-clouds #TODO-A talk about it.
    else:
        np.random.seed(args.random_seed)

    batch_keys = make_batch_keys(args)

    for batch in data_loader:
        # Move data to gpu
        for k in batch_keys:
            if k in batch:
                batch[k] = batch[k].to(device)

        if args.object_encoder == 'pnet':
            batch['objects'] = batch['objects'].permute(0, 1, 3, 2)

        # Forward pass
        res = model(batch)

        all_losses = compute_losses(batch, res, criteria, args)

        # Update the loss and accuracy meters
        target = batch['target_pos']
        batch_size = target.size(0)  # B x N_Objects
        total_loss_mtr.update(all_losses['total_loss'].item(), batch_size)

        # referential_loss_mtr.update(all_losses['referential_loss'].item(), batch_size)
        referential_loss_mtr.update(all_losses['referential_loss'], batch_size)

        predictions = torch.argmax(res['logits'], dim=1)
        guessed_correctly = torch.mean((predictions == target).double()).item()
        ref_acc_mtr.update(guessed_correctly, batch_size)

        if args.obj_cls_alpha > 0:
            cls_b_acc, _ = cls_pred_stats(res['class_logits'], batch['class_labels'], ignore_label=pad_idx)
            cls_acc_mtr.update(cls_b_acc, batch_size)
            obj_loss_mtr.update(all_losses['obj_clf_loss'].item(), batch_size)

        if args.lang_cls_alpha > 0:
            batch_guess = torch.argmax(res['lang_logits'], -1)
            cls_b_acc = torch.mean((batch_guess == batch['target_class']).double())
            txt_acc_mtr.update(cls_b_acc, batch_size)

    metrics['test_total_loss'] = total_loss_mtr.avg
    metrics['test_referential_loss'] = referential_loss_mtr.avg
    metrics['test_obj_clf_loss'] = obj_loss_mtr.avg
    metrics['test_referential_acc'] = ref_acc_mtr.avg
    metrics['test_object_cls_acc'] = cls_acc_mtr.avg
    metrics['test_txt_cls_acc'] = txt_acc_mtr.avg
    return metrics


@torch.no_grad()
def detailed_predictions_on_dataset(model, data_loader, args, device, FOR_VISUALIZATION=True):
    model.eval()

    res = dict()
    res['guessed_correctly'] = list()
    res['confidences_probs'] = list()
    res['contrasted_objects'] = list()
    res['target_pos'] = list()
    res['context_size'] = list()
    res['guessed_correctly_among_true_class'] = list()

    batch_keys = make_batch_keys(args, extras=['context_size', 'target_class_mask'])

    if FOR_VISUALIZATION:
        res['utterance'] = list()
        res['stimulus_id'] = list()
        res['object_ids'] = list()
        res['target_object_id'] = list()
        res['distrators_pos'] = list()

    for batch in tqdm.tqdm(data_loader):
        # Move data to gpu
        for k in batch_keys:
            if k in batch:
                batch[k] = batch[k].to(device)

        if args.object_encoder == 'pnet':
            batch['objects'] = batch['objects'].permute(0, 1, 3, 2)

        # Forward pass
        out = model(batch)

        if FOR_VISUALIZATION:
            n_ex = len(out['logits'])
            c = batch['context_size']
            n_obj = out['logits'].shape[1]
            for i in range(n_ex):
                if c[i] < n_obj:
                    out['logits'][i][c[i]:] = -10e6

        predictions = torch.argmax(out['logits'], dim=1)
        res['guessed_correctly'].append((predictions == batch['target_pos']).cpu().numpy())
        res['confidences_probs'].append(F.softmax(out['logits'], dim=1).cpu().numpy())
        res['contrasted_objects'].append(batch['class_labels'].cpu().numpy())
        res['target_pos'].append(batch['target_pos'].cpu().numpy())
        res['context_size'].append(batch['context_size'].cpu().numpy())

        if FOR_VISUALIZATION:
            res['utterance'].append(batch['utterance'])
            res['stimulus_id'].append(batch['stimulus_id'])
            res['object_ids'].append(batch['object_ids'])
            res['target_object_id'].append(batch['target_object_id'])
            res['distrators_pos'].append(batch['distrators_pos'])

        # also see what would happen if you where to constraint to the target's class.
        cancellation = -1e6
        mask = batch['target_class_mask']
        out['logits'] = out['logits'].float() * mask.float() + (~mask).float() * cancellation
        predictions = torch.argmax(out['logits'], dim=1)
        res['guessed_correctly_among_true_class'].append((predictions == batch['target_pos']).cpu().numpy())

    res['guessed_correctly'] = np.hstack(res['guessed_correctly'])
    res['confidences_probs'] = np.vstack(res['confidences_probs'])
    res['contrasted_objects'] = np.vstack(res['contrasted_objects'])
    res['target_pos'] = np.hstack(res['target_pos'])
    res['context_size'] = np.hstack(res['context_size'])
    res['guessed_correctly_among_true_class'] = np.hstack(res['guessed_correctly_among_true_class'])

    return res


@torch.no_grad()
def save_predictions_for_visualization(model, data_loader, device, channel_last, seed=13):
    """
    Return the predictions along with the scan data for further visualization
    """
    batch_keys = ['objects', 'tokens', 'class_labels', 'target_pos', 'scan', 'bboxes']

    # Set the model in eval mode
    model.eval()

    # Create table
    res_list = []

    # Fix the test random seed
    np.random.seed(seed)

    for batch in data_loader:
        # Move the batch to gpu
        for k in batch_keys:
            if len(batch[k]) > 0:
                batch[k] = batch[k].to(device)

        if not channel_last:
            batch['objects'] = batch['objects'].permute(0, 1, 3, 2)

        # Forward Pass
        res = model(batch)

        batch_size = batch['target_pos'].size(0)
        for i in range(batch_size):
            res_list.append({
                'scan_id': batch['scan_id'][i],
                'utterance': batch['utterance'][i],
                'target_pos': batch['target_pos'][i].cpu(),
                'confidences': res['logits'][i].cpu().numpy(),
                'bboxes': batch['objects_bboxes'][i].cpu().numpy(),
                'predicted_classes': res['class_logits'][i].argmax(dim=-1).cpu(),
                'predicted_target_pos': res['logits'][i].argmax(-1).cpu(),
                'object_ids': batch['object_ids'][i],
                'context_size': batch['context_size'][i],
                'is_easy': batch['is_easy'][i]
            })

    return res_list


def prediction_stats(logits, gt_labels):
    """ Get the prediction statistics: accuracy, correctly/wrongly predicted test examples
    :param logits: The output of the model (predictions) of size: B x N_Objects
    :param gt_labels: The ground truth labels of size: B x 1
    :param ignore_label: The label of the padding class (to be ignored)
    :return: The mean accuracy and lists of correct and wrong predictions
    """
    predictions = logits.argmax(dim=1)
    correct_guessed = gt_labels == predictions
    assert (type(correct_guessed) == torch.Tensor)
    mean_accuracy = torch.mean(correct_guessed.double()).item()
    return mean_accuracy


@torch.no_grad()
def cls_pred_stats(logits, gt_labels, ignore_label):
    """ Get the prediction statistics: accuracy, correctly/wrongly predicted test examples
    :param logits: The output of the model (predictions) of size: B x N_Objects x N_Classes
    :param gt_labels: The ground truth labels of size: B x N_Objects
    :param ignore_label: The label of the padding class (to be ignored)
    :return: The mean accuracy and lists of correct and wrong predictions
    """
    predictions = logits.argmax(dim=-1)  # B x N_Objects x N_Classes --> B x N_Objects
    valid_indices = gt_labels != ignore_label

    predictions = predictions[valid_indices]
    gt_labels = gt_labels[valid_indices]

    correct_guessed = gt_labels == predictions
    assert (type(correct_guessed) == torch.Tensor)

    found_samples = gt_labels[correct_guessed]
    # missed_samples = gt_labels[torch.logical_not(correct_guessed)] # TODO  - why?
    mean_accuracy = torch.mean(correct_guessed.double()).item()
    return mean_accuracy, found_samples
