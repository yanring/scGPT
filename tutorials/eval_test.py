# %% [markdown]
# # Fine-tuning Pre-trained Model for Perturbation Prediction

# %%
import json
import os
import sys
import time
import copy
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Union, Optional
import warnings
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

import torch
import numpy as np
import matplotlib
from torch import nn
from torch.nn import functional as F
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from torch_geometric.loader import DataLoader
from gears import PertData, GEARS
from gears.inference import compute_metrics, deeper_analysis, non_dropout_analysis
from gears.utils import create_cell_graph_dataset_for_prediction

sys.path.insert(0, "../")

import scgpt as scg
from scgpt.model import TransformerGenerator
from scgpt.loss import (
    masked_mse_loss,
    criterion_neg_log_bernoulli,
    masked_relative_error,
)
from scgpt.tokenizer import tokenize_batch, pad_batch, tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed, map_raw_id_to_vocab_id

matplotlib.rcParams["savefig.transparent"] = False
warnings.filterwarnings("ignore")

set_seed(42)

# settings for data prcocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
pad_value = 0  # for padding values
pert_pad_id = 2

n_hvg = 0  # number of highly variable genes
include_zero_gene = "all"  # include zero expr genes in training input, "all", "batch-wise", "row-wise", or False
max_seq_len = 1536

# settings for training
MLM = True  # whether to use masked language modeling, currently it is always on.
CLS = True  # celltype classification objective
CCE = False  # Contrastive cell embedding objective
MVC = False  # Masked value prediction for cell embedding
ECS = False  # Elastic cell similarity objective
cell_emb_style = "cls"
mvc_decoder_style = "inner product, detach"
amp = True
load_model = "/home/scratch.zijiey_sw/Algorithm/scGPT/save/dev_perturb_set16_train-Apr15-14-00"
# load_model = "/home/scratch.zijiey_sw/Algorithm/scGPT/save/dev_perturb_set3_train-Apr15-13-59"
load_model_general = "/home/scratch.zijiey_sw/Algorithm/scGPT/scgpt/save/scGPT_human"
load_param_prefixs = [
    "encoder",
    "value_encoder",
    "transformer_encoder",
]

# settings for optimizer
lr = 1e-4  # or 1e-4
batch_size = 1
eval_batch_size = 1
epochs = 15
schedule_interval = 1
early_stop = 5

# settings for the model
embsize = 512  # embedding dimension
d_hid = 512  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 12  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 8  # number of heads in nn.MultiheadAttention
n_layers_cls = 3
dropout = 0.2  # dropout probability
use_fast_transformer = True  # whether to use fast transformer

# logging
log_interval = 100

# dataset and evaluation choices
data_name = "set16_test1"
split = "no_split"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


save_dir = Path(f"/home/scratch.zijiey_sw/Algorithm/scGPT/save/"+data_name+"_result")
print(f"saving to {save_dir}")

import anndata
adata = anndata.read_h5ad('/home/scratch.zijiey_sw/Algorithm/scGPT/scgpt/save/new/gears_data/'+data_name+'_gears.h5ad')

pert_data = PertData("./data")
# pert_data.new_data_process(dataset_name = data_name, adata = adata)
pert_data.load(data_path = './data/'+data_name)
pert_data.prepare_split(split=split, seed=1)
pert_data.get_dataloader(batch_size=batch_size, test_batch_size=eval_batch_size)

if load_model is not None:
    model_dir = Path(load_model)
    model_dir_2 = Path(f"/home/scratch.zijiey_sw/Algorithm/scGPT/scgpt/save/scGPT_human")
    model_config_file = model_dir_2 / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir_2 / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    pert_data.adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in pert_data.adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(pert_data.adata.var["id_in_vocab"])
    genes = pert_data.adata.var["gene_name"].tolist()
    

    # model
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
        
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]
else:
    genes = pert_data.adata.var["gene_name"].tolist()
    vocab = Vocab(
        VocabPybind(genes + special_tokens, None)
    )  # bidirectional lookup [gene <-> int]
vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(
    [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int
)
n_genes = len(genes)


ntokens = len(vocab)  # size of vocabulary
model = TransformerGenerator(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    nlayers_cls=n_layers_cls,
    n_cls=1,
    vocab=vocab,
    dropout=dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    pert_pad_id=pert_pad_id,
    do_mvc=MVC,
    cell_emb_style=cell_emb_style,
    mvc_decoder_style=mvc_decoder_style,
    use_fast_transformer=use_fast_transformer,
)
if load_param_prefixs is not None and load_model is not None:
    # only load params that start with the prefix
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_file)
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if any([k.startswith(prefix) for prefix in load_param_prefixs])
    }
    # for k, v in pretrained_dict.items():
    #     logger.info(f"Loading params {k} with shape {v.shape}")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
elif load_model is not None:
    try:
        model.load_state_dict(torch.load(model_file))
        # logger.info(f"Loading all model params from {model_file}")
    except:
        # only load params that are in the model and match the size
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        # for k, v in pretrained_dict.items():
        #     logger.info(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
model.to(device)



def predict(
    model: TransformerGenerator, pert_list: List[str], pool_size: Optional[int] = None
) -> Dict:
    """
    Predict the gene expression values for the given perturbations.

    Args:
        model (:class:`torch.nn.Module`): The model to use for prediction.
        pert_list (:obj:`List[str]`): The list of perturbations to predict.
        pool_size (:obj:`int`, optional): For each perturbation, use this number
            of cells in the control and predict their perturbation results. Report
            the stats of these predictions. If `None`, use all control cells.
    """
    adata = pert_data.adata
    ctrl_adata = adata[adata.obs["condition"] == "ctrl"]
    if pool_size is None:
        pool_size = len(ctrl_adata.obs)
    gene_list = pert_data.gene_names.values.tolist()
    for pert in pert_list:
        for i in pert:
            if i not in gene_list:
                raise ValueError(
                    "The gene is not in the perturbation graph. Please select from GEARS.gene_list!"
                )

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        results_pred = {}
        for pert in pert_list:
            cell_graphs = create_cell_graph_dataset_for_prediction(
                pert, ctrl_adata, gene_list, device, num_samples=pool_size
            )
            loader = DataLoader(cell_graphs, batch_size=eval_batch_size, shuffle=False)
            preds = []
            for batch_data in loader:
                pred_gene_values = model.pred_perturb(
                    batch_data, include_zero_gene, gene_ids=gene_ids, amp=amp
                )
                preds.append(pred_gene_values)
            preds = torch.cat(preds, dim=0)
            results_pred["_".join(pert)] = np.mean(preds.detach().cpu().numpy(), axis=0)

    return results_pred



criterion = masked_mse_loss
criterion_cls = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_interval, gamma=0.9)
scaler = torch.cuda.amp.GradScaler(enabled=amp)


def evaluate(model: nn.Module, val_loader: torch.utils.data.DataLoader) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    cell_embed_total = []
    cls_total = []

    with torch.no_grad():
        for batch, batch_data in enumerate(val_loader):
            batch_size = len(batch_data.y)
            batch_data.to(device)
            x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
            
            ori_gene_values = x[:, 0].view(batch_size, n_genes)
            
            pert_flags = x[:, 1].long().view(batch_size, n_genes)
            target_gene_values = batch_data.y  # (batch_size, n_genes)

            if include_zero_gene in ["all", "batch-wise"]:
                if include_zero_gene == "all":
                    input_gene_ids = torch.arange(n_genes, device=device)
                else:  # when batch-wise
                    input_gene_ids = (
                        ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                    )

                # sample input_gene_id
                if len(input_gene_ids) > max_seq_len:
                    input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                        :max_seq_len
                    ]
                input_values = ori_gene_values[:, input_gene_ids]
                input_pert_flags = pert_flags[:, input_gene_ids]
                target_values = target_gene_values[:, input_gene_ids]

                mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
                mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

                # src_key_padding_mask = mapped_input_gene_ids.eq(vocab[pad_token])
                src_key_padding_mask = torch.zeros_like(
                    input_values, dtype=torch.bool, device=input_values.device
                )
            with torch.cuda.amp.autocast(enabled=amp):
                output_dict = model(
                    mapped_input_gene_ids,
                    input_values,
                    input_pert_flags,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=CLS,
                    CCE=CCE,
                    MVC=MVC,
                    ECS=ECS,
                    do_sample=True,
                )
            print(output_dict["cls_output"])
            print(output_dict["cell_emb"].shape)
            print(output_dict["cls_output"].shape)
            
            cell_embed_total.extend(output_dict["cell_emb"].cpu())
            cls_total.extend(output_dict["cls_output"].cpu())
            print(cls_total)
            exit()
            # cell_embed_total = torch.stack(cell_embed_total)
            #     output_values = output_dict["mlm_output"]

            #     masked_positions = torch.ones_like(
            #         input_values, dtype=torch.bool, device=input_values.device
            #     )
            #     loss = criterion(output_values, target_values, masked_positions)
            # total_loss += loss.item()
            # total_error += masked_relative_error(
            #     output_values, target_values, masked_positions
            # ).item()
    # return total_loss / len(val_loader), total_error / len(val_loader)
    return cell_embed_total
    



# %%
def eval_perturb(
    loader: DataLoader, model: TransformerGenerator, device: torch.device
) -> Dict:
    """
    Run model in inference mode using a given data loader
    """

    model.eval()
    model.to(device)
    pert_cat = []
    pred = []
    truth = []
    pred_de = []
    truth_de = []
    results = {}
    logvar = []

    for itr, batch in enumerate(loader):
        batch.to(device)
        pert_cat.extend(batch.pert)

        with torch.no_grad():
            p = model.pred_perturb(batch, include_zero_gene, gene_ids=gene_ids)
            t = batch.y
            pred.extend(p.cpu())
            truth.extend(t.cpu())

            # Differentially expressed genes
            for itr, de_idx in enumerate(batch.de_idx):
                pred_de.append(p[itr, de_idx])
                truth_de.append(t[itr, de_idx])

    # all genes
    results["pert_cat"] = np.array(pert_cat)
    pred = torch.stack(pred)
    truth = torch.stack(truth)
    results["pred"] = pred.detach().cpu().numpy().astype(float)
    results["truth"] = truth.detach().cpu().numpy().astype(float)

    pred_de = torch.stack(pred_de)
    truth_de = torch.stack(truth_de)
    results["pred_de"] = pred_de.detach().cpu().numpy().astype(float)
    results["truth_de"] = truth_de.detach().cpu().numpy().astype(float)

    return results



# %%
# test_loader = pert_data["test_loader"]
test_loader = pert_data.get_dataloader(batch_size=batch_size, test_batch_size=eval_batch_size)['test_loader']
test_res = eval_perturb(test_loader, model, device)
# test_emb = evaluate(model, test_loader)

# print(test_emb.shape)
cell_index = []
for p in pert_data.set2conditions['test']:
    if p != 'ctrl':
        cell_index.extend(list(adata.obs.index[adata.obs.condition == p]))
print('cell_index: ', len(cell_index))
print(test_res['pert_cat'])
print(test_res['pred'].shape)
print(test_res['truth'])
print(test_res['truth'].shape)

pred_index = np.argmax(test_res['pred'], axis=1)
truth_index = np.argmax(test_res['truth'], axis=1)
print(truth_index)

print(pert_data.gene_names[truth_index[0]])
print(pert_data.gene_names[pred_index[0]])

accuracy = accuracy_score(truth_index, pred_index)
f1 = f1_score(truth_index, pred_index, average='weighted')
print("acc: {:.4f}".format(accuracy))
print("f1: {:.4f}".format(f1))
exit()
df = pd.DataFrame(test_res['pred'], index = list(cell_index), columns = list(pert_data.gene_names))
df.to_csv(f'{save_dir}/scGPT_test1_setting16.csv')

df = pd.DataFrame(test_emb, index = list(cell_index))
df.to_csv(f'{save_dir}/scGPT_test1_setting16_embed.csv', header=False)

test_metrics, test_pert_res = compute_metrics(test_res)
print(test_metrics)

# save the dicts in json
with open(f"{save_dir}/test_metrics.json", "w") as f:
    json.dump(test_metrics, f)
with open(f"{save_dir}/test_pert_res.json", "w") as f:
    json.dump(test_pert_res, f)

# deeper_res = deeper_analysis(pert_data.adata, test_res)
# non_dropout_res = non_dropout_analysis(pert_data.adata, test_res)

# metrics = ["pearson_delta", "pearson_delta_de"]
# metrics_non_dropout = [
#     "pearson_delta_top20_de_non_dropout",
#     "pearson_top20_de_non_dropout",
# ]
# subgroup_analysis = {}
# for name in pert_data.subgroup["test_subgroup"].keys():
#     subgroup_analysis[name] = {}
#     for m in metrics:
#         subgroup_analysis[name][m] = []

#     for m in metrics_non_dropout:
#         subgroup_analysis[name][m] = []

# for name, pert_list in pert_data.subgroup["test_subgroup"].items():
#     for pert in pert_list:
#         for m in metrics:
#             subgroup_analysis[name][m].append(deeper_res[pert][m])

#         for m in metrics_non_dropout:
#             subgroup_analysis[name][m].append(non_dropout_res[pert][m])

# for name, result in subgroup_analysis.items():
#     for m in result.keys():
#         subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])



