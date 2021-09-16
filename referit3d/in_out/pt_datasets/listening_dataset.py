import random
import torch
import time
import os
import numpy as np
from torch.utils.data import Dataset
from functools import partial
from .utils import dataset_to_dataloader, max_io_workers

from pytorch_transformers.tokenization_bert import BertTokenizer

# the following will be shared on other datasets too if not, they should become part of the ListeningDataset
# maybe make SegmentedScanDataset with only static functions and then inherit.
from .utils import check_segmented_object_order, sample_scan_object, pad_samples, objects_bboxes
from .utils import instance_labels_of_context, mean_rgb_unit_norm_transform
from ...data_generation.nr3d import decode_stimulus_string


class ListeningDataset(Dataset):
    def __init__(self, references, scans, vocab, max_seq_len, points_per_object, max_distractors,
                 class_to_idx=None, object_transformation=None,
                 visualization=False, feat2dtype=None,
                 num_class_dim=525, evalmode=False):

        self.references = references
        self.scans = scans
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.points_per_object = points_per_object
        self.max_distractors = max_distractors
        self.max_context_size = self.max_distractors + 1  # to account for the target.
        self.class_to_idx = class_to_idx
        self.visualization = visualization
        self.object_transformation = object_transformation
        self.feat2dtype = feat2dtype
        self.max_2d_view = 5
        self.num_class_dim = num_class_dim
        self.evalmode = evalmode

        self.bert_tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased')
        assert self.bert_tokenizer.encode(self.bert_tokenizer.pad_token) == [0]

        if not check_segmented_object_order(scans):
            raise ValueError

    def __len__(self):
        return len(self.references)

    def get_reference_data(self, index):
        ref = self.references.loc[index]
        scan = self.scans[ref['scan_id']]
        target = scan.three_d_objects[ref['target_id']]
        tokens = np.array(self.vocab.encode(ref['tokens'], self.max_seq_len), dtype=np.long)
        is_nr3d = ref['dataset'] == 'nr3d'

        return scan, target, tokens, ref['tokens'], is_nr3d

    def prepare_distractors(self, scan, target):
        target_label = target.instance_label

        # First add all objects with the same instance-label as the target
        distractors = [o for o in scan.three_d_objects if
                       (o.instance_label == target_label and (o != target))]

        # Then all more objects up to max-number of distractors
        already_included = {target_label}
        clutter = [o for o in scan.three_d_objects if o.instance_label not in already_included]
        np.random.shuffle(clutter)

        distractors.extend(clutter)
        distractors = distractors[:self.max_distractors]
        np.random.shuffle(distractors)

        return distractors

    def __getitem__(self, index):
        res = dict()
        scan, target, tokens, text_tokens, is_nr3d = self.get_reference_data(index)
        ## BERT tokenize
        token_inds = torch.zeros(self.max_seq_len, dtype=torch.long)
        indices = self.bert_tokenizer.encode(
            ' '.join(text_tokens), add_special_tokens=True)
        indices = indices[:self.max_seq_len]
        token_inds[:len(indices)] = torch.tensor(indices)
        token_num = torch.tensor(len(indices), dtype=torch.long)

        # Make a context of distractors
        context = self.prepare_distractors(scan, target)

        # Add target object in 'context' list
        target_pos = np.random.randint(len(context) + 1)
        context.insert(target_pos, target)

        # sample point/color for them
        samples = np.array([sample_scan_object(o, self.points_per_object) for o in context])

        # mark their classes
        res['class_labels'] = instance_labels_of_context(context, self.max_context_size, self.class_to_idx)

        if self.object_transformation is not None:
            samples, offset = self.object_transformation(samples)
            res['obj_offset'] = np.zeros((self.max_context_size, offset.shape[1])).astype(np.float32)
            res['obj_offset'][:len(offset),:] = offset.astype(np.float32)

        res['context_size'] = len(samples)

        # take care of padding, so that a batch has same number of N-objects across scans.
        res['objects'] = pad_samples(samples, self.max_context_size)

        # Get a mask indicating which objects have the same instance-class as the target.
        target_class_mask = np.zeros(self.max_context_size, dtype=np.bool)
        target_class_mask[:len(context)] = [target.instance_label == o.instance_label for o in context]

        res['target_class'] = self.class_to_idx[target.instance_label]
        res['target_pos'] = target_pos
        res['target_class_mask'] = target_class_mask
        res['tokens'] = tokens
        res['token_inds'] = token_inds.numpy().astype(np.int64)
        res['token_num'] = token_num.numpy().astype(np.int64)
        res['is_nr3d'] = is_nr3d

        if self.visualization:
            distrators_pos = np.zeros((6))  # 6 is the maximum context size we used in dataset collection
            object_ids = np.zeros((self.max_context_size))
            j = 0
            for k, o in enumerate(context):
                if o.instance_label == target.instance_label and o.object_id != target.object_id:
                    distrators_pos[j] = k
                    j += 1
            for k, o in enumerate(context):
                object_ids[k] = o.object_id
            res['utterance'] = self.references.loc[index]['utterance']
            res['stimulus_id'] = self.references.loc[index]['stimulus_id']
            res['distrators_pos'] = distrators_pos
            res['object_ids'] = object_ids
            res['target_object_id'] = target.object_id
        if self.evalmode:
            return res

        # load cached 2D context information
        if os.path.isfile('../data/scannet_frames_25k_gtobjfeat_aggregate/%s.npy'%scan.scan_id):
            context_2d = np.load('../data/scannet_frames_25k_gtobjfeat_aggregate/%s.npy'%scan.scan_id,allow_pickle=True,encoding='latin1')    ## TODO: update relative path
            objfeat_2d = context_2d.item()['obj_feat']
            bbox_2d = context_2d.item()['obj_coord']
            bboxsize_2d = context_2d.item()['obj_size']
            obj_depth = context_2d.item()['obj_depth']
            campose_2d = context_2d.item()['camera_pose']
            ins_id_2d = context_2d.item()['instance_id']
            if (self.feat2dtype.replace('3D',''))=='ROI': featdim = 2048
            elif (self.feat2dtype.replace('3D',''))=='clsvec': featdim = self.num_class_dim
            elif (self.feat2dtype.replace('3D',''))=='clsvecROI': featdim = 2048+self.num_class_dim
            feat_2d = np.zeros((self.max_context_size, featdim)).astype(np.float32)
            coords_2d = np.zeros((self.max_context_size, 4+12)).astype(np.float32)

            selected_2d_idx = 0
            selected_context_id = [o.object_id+1 for o in context] ## backbround included in cache, so +1
            ## only for creating tensor of the correct size
            selected_objfeat_2d = objfeat_2d[selected_context_id,selected_2d_idx,:]
            selected_bbox_2d = bbox_2d[selected_context_id,selected_2d_idx,:]
            selected_bboxsize_2d = bboxsize_2d[selected_context_id,selected_2d_idx]
            selected_obj_depth = obj_depth[selected_context_id,selected_2d_idx]
            selected_campose_2d = campose_2d[selected_context_id,selected_2d_idx,:]
            selected_ins_id_2d = ins_id_2d[selected_context_id,selected_2d_idx]
            ## Fill in randomly selected view of 2D features
            for ii in range(len(selected_context_id)):
                cxt_id = selected_context_id[ii]
                view_id = random.randint(0, max(0,int((ins_id_2d[cxt_id,:]!=0).astype(np.float32).sum())-1))
                selected_objfeat_2d[ii,:] = objfeat_2d[cxt_id,view_id,:]
                selected_bbox_2d[ii,:] = bbox_2d[cxt_id,view_id,:]
                selected_bboxsize_2d[ii] = bboxsize_2d[cxt_id,view_id]
                selected_obj_depth[ii] = obj_depth[cxt_id,view_id]
                selected_campose_2d[ii,:] = campose_2d[cxt_id,view_id,:]

            if self.feat2dtype!='clsvec':
                feat_2d[:len(selected_context_id),:2048] = selected_objfeat_2d
            for ii in range(len(res['class_labels'])):
                if self.feat2dtype=='clsvec':
                    feat_2d[ii,res['class_labels'][ii]] = 1.
                if self.feat2dtype=='clsvecROI':
                    feat_2d[ii,2048+res['class_labels'][ii]] = 1.
            coords_2d[:len(selected_context_id),:] = np.concatenate([selected_bbox_2d, selected_campose_2d[:,:12]],axis=-1)
            coords_2d[:,0], coords_2d[:,2] = coords_2d[:,0]/1296., coords_2d[:,2]/1296. ## norm by image size
            coords_2d[:,1], coords_2d[:,3] = coords_2d[:,1]/968., coords_2d[:,3]/968.
        else:
            print('please prepare the cached 2d feature')
            exit(0)
        res['feat_2d'] = feat_2d
        res['coords_2d'] = coords_2d

        return res

def make_data_loaders(args, referit_data, vocab, class_to_idx, scans, mean_rgb, seed=None):
    n_workers = args.n_workers
    if n_workers == -1:
        n_workers = max_io_workers()

    data_loaders = dict()
    is_train = referit_data['is_train']
    splits = ['train', 'test']

    object_transformation = partial(mean_rgb_unit_norm_transform, mean_rgb=mean_rgb,
                                    unit_norm=args.unit_sphere_norm)
    for split in splits:
        mask = is_train if split == 'train' else ~is_train
        d_set = referit_data[mask]
        d_set.reset_index(drop=True, inplace=True)

        max_distractors = args.max_distractors if split == 'train' else args.max_test_objects - 1
        ## this is a silly small bug -> not the minus-1.

        # if split == test remove the utterances of unique targets
        if split == 'test':
            def multiple_targets_utterance(x):
                _, _, _, _, distractors_ids = decode_stimulus_string(x.stimulus_id)
                return len(distractors_ids) > 0

            multiple_targets_mask = d_set.apply(multiple_targets_utterance, axis=1)
            d_set = d_set[multiple_targets_mask]
            d_set.reset_index(drop=True, inplace=True)
            print("length of dataset before removing non multiple test utterances {}".format(len(d_set)))
            print("removed {} utterances from the test set that don't have multiple distractors".format(
                np.sum(~multiple_targets_mask)))
            print("length of dataset after removing non multiple test utterances {}".format(len(d_set)))

            assert np.sum(~d_set.apply(multiple_targets_utterance, axis=1)) == 0

        dataset = ListeningDataset(references=d_set,
                                   scans=scans,
                                   vocab=vocab,
                                   max_seq_len=args.max_seq_len,
                                   points_per_object=args.points_per_object,
                                   max_distractors=max_distractors,
                                   class_to_idx=class_to_idx,
                                   object_transformation=object_transformation,
                                   visualization=args.mode == 'evaluate',
                                   feat2dtype=args.feat2d,
                                   num_class_dim = 525 if '00' in args.scannet_file else 608,
                                   evalmode=(args.mode=='evaluate'))

        seed = seed
        if split == 'test':
            seed = args.random_seed

        data_loaders[split] = dataset_to_dataloader(dataset, split, args.batch_size, n_workers, pin_memory=True, seed=seed)

    return data_loaders
