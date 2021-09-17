import torch
import random
import argparse
from torch import nn
from collections import defaultdict


from pytorch_transformers.modeling_bert import (
    BertLayerNorm, BertEmbeddings, BertEncoder, BertConfig,
    BertPreTrainedModel
)
from .mmt_module import *

from . import DGCNN
from .default_blocks import *
from .utils import get_siamese_features
from ..in_out.vocabulary import Vocabulary

try:
    from . import PointNetPP
except ImportError:
    PointNetPP = None


class MMT_ReferIt3DNet(nn.Module):
    def __init__(self,
                 args,
                 object_encoder,
                 num_class,
                 object_language_clf=None,
                 object_clf=None,
                 language_clf=None,
                 visudim=128,
                 MMT_HIDDEN_SIZE=192,
                 TEXT_BERT_HIDDEN_SIZE=768,
                 context_2d=None,
                 feat2dtype=None,
                 mmt_mask=None):

        super().__init__()

        self.args_mode = args.mode
        self.text_length = args.max_seq_len
        self.context_2d = context_2d
        self.feat2dtype = feat2dtype
        self.mmt_mask = mmt_mask
        # Encoders for single object
        self.object_encoder = object_encoder
        self.linear_obj_feat_to_mmt_in = nn.Linear(visudim, MMT_HIDDEN_SIZE)
        self.linear_obj_bbox_to_mmt_in = nn.Linear(4, MMT_HIDDEN_SIZE)
        self.obj_feat_layer_norm = BertLayerNorm(MMT_HIDDEN_SIZE)
        self.obj_bbox_layer_norm = BertLayerNorm(MMT_HIDDEN_SIZE)
        self.obj_drop = nn.Dropout(0.1)
        # Encoders for visual 2D objects
        num_class_dim = 525 if '00' in args.scannet_file else 608
        if (args.feat2d.replace('3D',''))=='ROI': featdim = 2048
        elif (args.feat2d.replace('3D',''))=='clsvec': featdim = num_class_dim
        elif (args.feat2d.replace('3D',''))=='clsvecROI': featdim = 2048+num_class_dim
        self.linear_2d_feat_to_mmt_in = nn.Linear(featdim, MMT_HIDDEN_SIZE)
        self.linear_2d_bbox_to_mmt_in = nn.Linear(16, MMT_HIDDEN_SIZE)
        self.obj2d_feat_layer_norm = BertLayerNorm(MMT_HIDDEN_SIZE)
        self.obj2d_bbox_layer_norm = BertLayerNorm(MMT_HIDDEN_SIZE)

        ## encoder for context object
        self.cnt_object_encoder = single_object_encoder(768)
        self.cnt_linear_obj_feat_to_mmt_in = nn.Linear(visudim, MMT_HIDDEN_SIZE)
        self.cnt_linear_obj_bbox_to_mmt_in = nn.Linear(4, MMT_HIDDEN_SIZE)
        self.cnt_feat_layer_norm = BertLayerNorm(MMT_HIDDEN_SIZE)
        self.cnt_bbox_layer_norm = BertLayerNorm(MMT_HIDDEN_SIZE)
        self.context_drop = nn.Dropout(0.1)

        # Encoders for text
        self.text_bert_config = BertConfig(
                 hidden_size=TEXT_BERT_HIDDEN_SIZE,
                 num_hidden_layers=3,
                 num_attention_heads=12,
                 type_vocab_size=2)
        self.text_bert = TextBert.from_pretrained(
            'bert-base-uncased', config=self.text_bert_config,\
            mmt_mask=self.mmt_mask)
        if TEXT_BERT_HIDDEN_SIZE!=MMT_HIDDEN_SIZE:
            self.text_bert_out_linear = nn.Linear(TEXT_BERT_HIDDEN_SIZE, MMT_HIDDEN_SIZE)
        else:
            self.text_bert_out_linear = nn.Identity()

        # Classifier heads
        self.object_clf = object_clf
        self.language_clf = language_clf
        self.object_language_clf = object_language_clf

        self.mmt_config = BertConfig(
                 hidden_size=MMT_HIDDEN_SIZE,
                 num_hidden_layers=4,
                 num_attention_heads=12,
                 type_vocab_size=2)
        self.mmt = MMT(self.mmt_config,context_2d=self.context_2d,mmt_mask=self.mmt_mask)
        self.matching_cls = MatchingLinear(input_size=MMT_HIDDEN_SIZE)
        if self.context_2d=='unaligned':
            self.matching_cls_2D = MatchingLinear(input_size=MMT_HIDDEN_SIZE)
        self.mlm_cls = BertLMPredictionHead(self.text_bert.embeddings.word_embeddings.weight, input_size=MMT_HIDDEN_SIZE)
        self.contra_cls = PolluteLinear()

    def __call__(self, batch: dict) -> dict:
        result = defaultdict(lambda: None)

        # Get features for each segmented scan object based on color and point-cloud
        objects_features = get_siamese_features(self.object_encoder, batch['objects'],
                                                aggregator=torch.stack)  # B X N_Objects x object-latent-dim

        obj_mmt_in = self.obj_feat_layer_norm(self.linear_obj_feat_to_mmt_in(objects_features)) + \
            self.obj_bbox_layer_norm(self.linear_obj_bbox_to_mmt_in(batch['obj_offset'])) 
        if self.context_2d=='aligned':
            obj_mmt_in = obj_mmt_in + \
                self.obj2d_feat_layer_norm(self.linear_2d_feat_to_mmt_in(batch['feat_2d'])) + \
                self.obj2d_bbox_layer_norm(self.linear_2d_bbox_to_mmt_in(batch['coords_2d']))

        obj_mmt_in = self.obj_drop(obj_mmt_in)
        obj_num = obj_mmt_in.size(1)
        obj_mask = _get_mask(batch['context_size'].to(obj_mmt_in.device), obj_num)    ## all proposals are non-empty

        # Classify the segmented objects
        if self.object_clf is not None:
            objects_classifier_features = obj_mmt_in
            result['class_logits'] = get_siamese_features(self.object_clf, objects_classifier_features, torch.stack)

        if self.context_2d=='unaligned':
            context_obj_mmt_in = self.obj2d_feat_layer_norm(self.linear_2d_feat_to_mmt_in(batch['feat_2d'])) + \
                self.obj2d_bbox_layer_norm(self.linear_2d_bbox_to_mmt_in(batch['coords_2d']))
            context_obj_mmt_in = self.context_drop(context_obj_mmt_in)
            context_obj_mask = _get_mask(batch['context_size'].to(context_obj_mmt_in.device), obj_num)    ## all proposals are non-empty
            obj_mmt_in = torch.cat([obj_mmt_in, context_obj_mmt_in],dim=1)
            obj_mask = torch.cat([obj_mask, context_obj_mask],dim=1)

        # Get feature for utterance
        txt_inds = batch["token_inds"] # batch_size, lang_size
        txt_type_mask = torch.ones(txt_inds.shape, device=torch.device('cuda')) * 1.
        txt_mask = _get_mask(batch['token_num'].to(txt_inds.device), txt_inds.size(1))  ## all proposals are non-empty
        txt_type_mask = txt_type_mask.long()

        text_bert_out = self.text_bert(
            txt_inds=txt_inds,
            txt_mask=txt_mask,
            txt_type_mask=txt_type_mask
        )
        txt_emb = self.text_bert_out_linear(text_bert_out)
        # Classify the target instance label based on the text
        if self.language_clf is not None:
            result['lang_logits'] = self.language_clf(text_bert_out[:,0,:])

        mmt_results = self.mmt(
            txt_emb=txt_emb,
            txt_mask=txt_mask,
            obj_emb=obj_mmt_in,
            obj_mask=obj_mask,
            obj_num=obj_num
        )
        if self.args_mode == 'evaluate':
            assert(mmt_results['mmt_seq_output'].shape[1]==(self.text_length+obj_num))
        if self.args_mode != 'evaluate' and self.context_2d=='unaligned':
            assert(mmt_results['mmt_seq_output'].shape[1]==(self.text_length+obj_num*2))
        result['logits'] = self.matching_cls(mmt_results['mmt_obj_output'])

        result['mmt_obj_output'] = mmt_results['mmt_obj_output']
        if self.context_2d=='unaligned':
            result['logits_2D'] = self.matching_cls_2D(mmt_results['mmt_obj_output_2D'])
            result['mmt_obj_output_2D'] = mmt_results['mmt_obj_output_2D']
        return result

def instantiate_referit3d_net(args: argparse.Namespace, vocab: Vocabulary, n_obj_classes: int) -> nn.Module:
    """
    Creates a neural listener by utilizing the parameters described in the args
    but also some "default" choices we chose to fix in this paper.

    @param args:
    @param vocab:
    @param n_obj_classes: (int)
    """

    # convenience
    geo_out_dim = args.object_latent_dim
    lang_out_dim = args.language_latent_dim
    mmt_out_dim = args.mmt_latent_dim

    # make an object (segment) encoder for point-clouds with color
    if args.object_encoder == 'pnet_pp':
        object_encoder = single_object_encoder(geo_out_dim)
    else:
        raise ValueError('Unknown object point cloud encoder!')

    # Optional, make a bbox encoder
    object_clf = None
    if args.obj_cls_alpha > 0:
        print('Adding an object-classification loss.')
        object_clf = object_decoder_for_clf(geo_out_dim, n_obj_classes)

    if args.model.startswith('mmt') and args.transformer:
        lang_out_dim = 768
        language_clf = None
        if args.lang_cls_alpha > 0:
            print('Adding a text-classification loss.')
            language_clf = text_decoder_for_clf(lang_out_dim, n_obj_classes)

        model = MMT_ReferIt3DNet(            
            args=args,
            num_class=n_obj_classes,
            object_encoder=object_encoder,
            object_clf=object_clf,
            language_clf=language_clf,
            visudim=geo_out_dim,
            TEXT_BERT_HIDDEN_SIZE=lang_out_dim,
            MMT_HIDDEN_SIZE=mmt_out_dim,
            context_2d=args.context_2d,
            feat2dtype=args.feat2d,
            mmt_mask=args.mmt_mask)
    else:
        raise NotImplementedError('Unknown listener model is requested.')

    return model


## pad at the end; used anyway by obj, ocr mmt encode
def _get_mask(nums, max_num):
    # non_pad_mask: b x lq, torch.float32, 0. on PAD
    batch_size = nums.size(0)
    arange = torch.arange(0, max_num).unsqueeze(0).expand(batch_size, -1)
    non_pad_mask = arange.to(nums.device).lt(nums.unsqueeze(-1))
    non_pad_mask = non_pad_mask.type(torch.float32)
    return non_pad_mask