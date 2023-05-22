from math import ceil
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
import UDLib as UD
from bert_wrapper import apply_bert_model_to_pretokinsed


def get_tokens_with_labels(gold_tokens, labels, tokeniser):
    assert len(gold_tokens) == len(labels)
    out_tokens = [tokeniser.cls_token_id]
    out_labels = [None]
    for t, l in zip(gold_tokens, labels):
        if len(out_tokens) == 511:
            break
        tok_ids = tokeniser.encode(t, add_special_tokens=False)
        out_labels.append(l)
        out_tokens.append(tok_ids[0])
        if len(out_tokens) == 511:
            break
        for tok_id in tok_ids[1:]:
            out_tokens.append(tok_id)
            out_labels.append(None)
            if len(out_tokens) == 511:
                break
    out_tokens.append(tokeniser.sep_token_id)
    out_labels.append(None)
    assert len(out_tokens) == len(out_labels)
    seq_length = len(out_tokens)
    padding_length = 512 - seq_length
    tokeniser_output = {
        'input_ids': torch.tensor(out_tokens + [0] * padding_length),
        'token_type_ids': torch.tensor([0] * seq_length +
        [tokeniser._pad_token_type_id] * padding_length),
        'attention_mask': torch.tensor([1] * seq_length + [0] * padding_length)
    }
    return tokeniser_output, out_labels


model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
# bert_model = nn.DataParallel(bert_model)
bert_model.cuda()

trees_train = UD.conllu2trees(
    '/mount/arbeitsdaten33/projekte/tcl/Users/nikolady/ud-treebanks-v2.8/UD_English-GUM/en_gum-ud-train.conllu')
trees_dev = UD.conllu2trees(
    '/mount/arbeitsdaten33/projekte/tcl/Users/nikolady/ud-treebanks-v2.8/UD_English-GUM/en_gum-ud-dev.conllu')
trees = trees_train + trees_dev

batch_size = 256
n_steps = ceil(len(trees) / batch_size)

root_output = torch.zeros(len(trees), 768)
sentences = []
for batch_n in tqdm(range(n_steps)):
    lo = batch_size * batch_n
    hi = batch_size * (batch_n + 1)
    batch = trees[lo: hi]

    input_ids = []
    token_type_ids = []
    attention_mask = []
    batch_labels = []
    for tree in batch:
        good_keys = list(
            filter(lambda k: '.' not in k and '-' not in k, tree.keys))
        tokens = [tree.nodes[key].FORM for key in good_keys]
        deprels = [tree.nodes[key].DEPREL for key in good_keys]
        tokenizer_output, labels = get_tokens_with_labels(
            tokens, deprels, tokenizer)
        input_ids.append(tokenizer_output['input_ids'])
        token_type_ids.append(tokenizer_output['token_type_ids'])
        attention_mask.append(tokenizer_output['attention_mask'])
        batch_labels.append(labels)
    hi = lo + len(batch)
    with torch.no_grad():
        mbert_inputs = {
            'input_ids': torch.vstack(input_ids).cuda(),
            'token_type_ids': torch.vstack(token_type_ids).cuda(),
            'attention_mask': torch.vstack(attention_mask).cuda()
        }
        # batch_outputs = bert_model(**mbert_inputs).hidden_states[3]
        hidden_states, ff_outputs = apply_bert_model_to_pretokinsed(
            bert_model, mbert_inputs)
        ff_layer_output = ff_outputs[10]
        for j in range(ff_layer_output.size(0)):
            root_idx = batch_labels[j].index('root')
            root_output[lo + j] = ff_layer_output[j, root_idx]
        # root_output[lo: hi] = bert_model(**mbert_inputs).pooler_output
np.savetxt(f'../csv/mbert_vanilla_GUM_roots_ff_train_dev_11.csv',
           root_output.numpy(), delimiter=',')
