import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_transformers.modeling_bert import (
    BertLayerNorm, BertEmbeddings, BertEncoder, BertConfig,
    BertPreTrainedModel
)

class TextBert(BertPreTrainedModel):
    def __init__(self, config, mmt_mask=None):
        super().__init__(config)

        self.mmt_mask = mmt_mask
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.init_weights()

    def forward(self, txt_inds, txt_mask, txt_type_mask=None):
        ## https://huggingface.co/transformers/v1.2.0/_modules/pytorch_transformers/modeling_bert.html
        encoder_inputs = self.embeddings(txt_inds, token_type_ids=txt_type_mask)
        attention_mask = txt_mask

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        if self.mmt_mask=='train2dmasklabel':
            to_seq_length = attention_mask.size(1)
            from_seq_length = to_seq_length
            extended_attention_mask = extended_attention_mask.repeat(
                1, 1, from_seq_length, 1
            )
            num_query = 24
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )
        seq_output = encoder_outputs[0]
        return seq_output

class MMT(BertPreTrainedModel):
    def __init__(self, config, context_2d=None, mmt_mask=None):
        super().__init__(config)

        self.context_2d = context_2d
        self.mmt_mask = mmt_mask
        self.encoder = BertEncoder(config)
        self.init_weights()

    def forward(self, txt_emb, txt_mask, obj_emb, obj_mask, obj_num):
        encoder_inputs = torch.cat([txt_emb,obj_emb],dim=1)
        attention_mask = torch.cat([txt_mask,obj_mask],dim=1)

        encoder_inputs = encoder_inputs

        txt_max_num = txt_mask.size(-1)
        obj_max_num = obj_mask.size(-1)
        txt_begin = 0
        obj_begin = txt_max_num

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        if self.mmt_mask=='train2d' or self.mmt_mask=='train2dmasklabel':
            # [batch_size, from_seq_length, to_seq_length]
            # mask type 1: 3d, lang can't see 2d
            to_seq_length = attention_mask.size(1)
            from_seq_length = to_seq_length
            extended_attention_mask = extended_attention_mask.repeat(
                1, 1, from_seq_length, 1
            )
            # decoding step elements can attend to themselves in a causal manner
            num_2d = obj_max_num//2
            extended_attention_mask[:, :, :-num_2d, -num_2d:] = 0.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )

        mmt_seq_output = encoder_outputs[0]
        mmt_txt_output = mmt_seq_output[:, txt_begin:txt_max_num]
        mmt_obj_output = mmt_seq_output[:, txt_max_num:]
        results = {
            'mmt_seq_output': mmt_seq_output,
            'mmt_txt_output': mmt_txt_output,
            'mmt_obj_output': mmt_obj_output[:,:obj_num,:],
        }
        if self.context_2d=='unaligned':
            results['mmt_obj_output_2D'] = mmt_obj_output[:,obj_num:,:]
        return results

class MatchingLinear(nn.Module):
    def __init__(self, input_size=192, hidden_size=128,outputdim=1):
        super(MatchingLinear, self).__init__()
        hidden_size = input_size*2//3
        self.dense = nn.Linear(input_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.decoder = nn.Linear(hidden_size, outputdim)

    def forward(self,x):
        hidden_state = self.LayerNorm(gelu(self.dense(x)))
        return self.decoder(hidden_state).squeeze(2)

"""
From VilBert, vilbert/vilbert
"""
class BertLMPredictionHead(nn.Module):
    def __init__(self, bert_model_embedding_weights, input_size=None):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(input_size=input_size)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            bert_model_embedding_weights.size(1),
            bert_model_embedding_weights.size(0),
            bias=False,
        )
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, input_size=None):
        super(BertPredictionHeadTransform, self).__init__()
        hidden_act = "gelu"
        hidden_size = 768
        if input_size is None:
            input_size = hidden_size
        ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}
        self.dense = nn.Linear(input_size, hidden_size)
        if isinstance(hidden_act, str) or (
            sys.version_info[0] == 2 and isinstance(hidden_act, unicode)
        ):
            self.transform_act_fn = ACT2FN[hidden_act]
        else:
            self.transform_act_fn = hidden_act
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class PolluteLinear(nn.Module):
    def __init__(self, input_size=768, hidden_size=512):
        super(PolluteLinear, self).__init__()
        self.dense = nn.Linear(input_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.decoder = nn.Linear(hidden_size, 1)

    def forward(self,x):
        hidden_state = self.LayerNorm(gelu(self.dense(x)))
        return self.decoder(hidden_state)

## pad at the end; used anyway by obj, ocr mmt encode
def _get_mask(nums, max_num):
    # non_pad_mask: b x lq, torch.float32, 0. on PAD
    batch_size = nums.size(0)
    arange = torch.arange(0, max_num).unsqueeze(0).expand(batch_size, -1)
    non_pad_mask = arange.to(nums.device).lt(nums.unsqueeze(-1))
    non_pad_mask = non_pad_mask.type(torch.float32)
    return non_pad_mask

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)