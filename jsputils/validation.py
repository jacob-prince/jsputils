from jsputils import paths, models

import torch as ch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torch.distributed as dist
ch.backends.cudnn.benchmark = True
ch.autograd.profiler.emit_nvtx(False)
ch.autograd.profiler.profile(False)

import torchmetrics
import numpy as np
import pandas as pd
from tqdm import tqdm
from fastprogress import progress_bar

import os
from os.path import exists
import sys
import time
import json
import gc
from glob import glob
from uuid import uuid4
from typing import List
from pathlib import Path
from argparse import ArgumentParser

# unclear if these are needed?
from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder
from ffcv.fields import IntField, RGBImageField

from pdb import set_trace

########

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256

num_classes = 1000
lr_tta = 1
resolution = 224

def get_imagenet_class_accuracies(model, val_dataloader, device, topk = 5, cv = False):
    
    model.eval()
    
    true_labels = []
    pred_labels = []
        
    start = time.time()
    
    all_labels = []
    
    with ch.no_grad():
        with autocast():
            for imgs, target, _, _ in val_dataloader:

                imgs = imgs.to(device)
                with ch.no_grad():
                    out = model(imgs)

                    true_labels.append(target.detach().cpu().numpy())

                    _, preds = out.topk(k=topk, dim=1, largest=True, sorted=True)
                    pred_labels.append(preds.detach().cpu().numpy())
            
    categs = np.sort(np.unique(np.concatenate(true_labels)))
    
    assert(len(categs) == 1000)
    
    all_pred_labels = np.vstack(pred_labels)
    all_true_labels = np.concatenate(true_labels)
    
    if cv:
        
        all_c_accs = np.zeros((len(categs),2))
        
        for p, subset in enumerate([np.arange(25), np.arange(25,50)]):
            
            for c in categs:
                idx = np.argwhere(all_true_labels == c)[subset] # which indices are from this category?

                if topk > 1:
                    all_c_accs[c,p] = np.mean(np.sum(np.squeeze(all_pred_labels[idx] == c),axis=1))
                elif topk == 1:
                    all_c_accs[c,p] = np.mean(all_pred_labels[idx] == c)

    else:
        
        all_c_accs = np.zeros((len(categs),))

        for c in categs:
            idx = np.argwhere(all_true_labels == c) # which indices are from this category?

            if topk > 1:
                all_c_accs[c] = np.mean(np.sum(np.squeeze(all_pred_labels[idx] == c),axis=1))
            elif topk == 1:
                all_c_accs[c] = np.mean(all_pred_labels[idx] == c)
            
    del imgs, out, preds, target
    gc.collect()
    ch.cuda.empty_cache()
               
    return all_c_accs #, all_pred_labels, all_true_labels


def get_selective_unit_acts(model, selective_units, layer_names, val_dataloader, device):
    
    floc_domains = selective_units['floc_domains']
    domain_mean_acts = dict()
    for domain in floc_domains:
        domain_mean_acts[domain] = dict()
        for layer in layer_names:
            domain_mean_acts[domain][layer] = []
            
    start = time.time()
    count = 0
            
    all_labels = []
    
    with ch.no_grad():
        with autocast():
            for imgs, target, _, _ in progress_bar(val_dataloader):

                imgs = imgs.to(device)
                out, acts = model(imgs)

                for layer in layer_names:

                    for domain in floc_domains:

                        ac = acts[layer].detach().clone()
                        
                        ac = ch.reshape(ac, (ac.shape[0], -1))
                        this_mask = selective_units[domain][layer]['mask']
                        try:
                            ac_masked = ac[:,this_mask]
                        except:
                            set_trace()
                        mean_act = ch.mean(ac_masked, axis=1)

                        domain_mean_acts[domain][layer].append(mean_act)
            
                count += imgs.shape[0]
        
                all_labels.append(target.detach().clone())
        
    all_labels = ch.cat(all_labels).cpu().numpy()
    
    all_layer_mean_acts = dict()
    
    for domain in floc_domains:

        all_layer_mean_acts[domain] = dict()

        for lay in layer_names:
            c_acts = np.zeros((1000,))
            
            all_mean_acts = ch.cat(domain_mean_acts[domain][lay]).cpu().numpy()

            for c in range(1000):
                idx = np.argwhere(all_labels == c)
                c_acts[c] = np.nanmean(all_mean_acts[idx])

            all_layer_mean_acts[domain][lay] = c_acts
            
    end = time.time()
    dur = end - start
    print(f"==> dur={dur}s")
    
    del imgs, target, out, acts, ac, ac_masked, mean_act, domain_mean_acts, all_labels
    gc.collect()
    ch.cuda.empty_cache()

    return all_layer_mean_acts

def validate_trained_model(model_dir, dropout_prop, learning_rate, lr_peak,
                           device):
    
    weight_dir = paths.dnn_dropout_weightdir()

    model_list = os.listdir(f'{weight_dir}/{model_dir}')

    dropout_props = [float(mdl.split('_')[1][4:])/100 for mdl in model_list]
    learning_rates = [float(mdl.split('_')[3][3:])/100 for mdl in model_list]
    lr_peaks = [float(mdl.split('_')[4][4:])/100 for mdl in model_list]

    model_str = f'alexnet_drop{int(100*dropout_prop):03}_ep100_lr{int(100*learning_rate):02}_peak{int(100*lr_peak):02}_ramp65-76'
    this_model_dir = f'{weight_dir}/{model_dir}/{model_str}'
    weight_fn = f'{this_model_dir}/{os.listdir(this_model_dir)[0]}/final_weights.pt'
    assert(exists(weight_fn))
    
    model = models.AlexNet(dropout = dropout_prop).eval()

    state_dict = ch.load(weight_fn)
    state_dict = {str.replace(k,'module.',''): v for k,v in state_dict.items()}

    model.load_state_dict(state_dict)

    model.to(device)
    
    val_meters = {'top_1': torchmetrics.Accuracy(task="multiclass", compute_on_step=False, num_classes=num_classes).to(device),
                  'top_5': torchmetrics.Accuracy(task="multiclass", compute_on_step=False, num_classes=num_classes, top_k=5).to(device), 
                  'loss': MeanScalarMetric(compute_on_step=False).to(device)}
    
    val_dataset = paths.ffcv_imagenet1k_valset()

    val_loader = create_val_loader(val_dataset, resolution = resolution)

    loss = ch.nn.CrossEntropyLoss(label_smoothing=0.1)
    
    with ch.no_grad():
        with autocast():
            for images, target, _, _ in progress_bar(val_loader):
                output = model(images)
                if lr_tta:
                    output += model(ch.flip(images, dims=[3]))

                for k in ['top_1', 'top_5']:
                    val_meters[k](output, target)

                loss_val = loss(output, target)
                val_meters['loss'](loss_val)

    stats = {k: m.compute().item() for k, m in val_meters.items()}
    
    del model, val_meters, val_loader, images, target, output
    ch.cuda.empty_cache()
    gc.collect()
    
    return stats, model_str
     
def create_val_loader(val_dataset, indices = None, device = 'cuda:0', num_workers = 64, batch_size = 512, resolution = 256, batches_ahead = 3, distributed = 0, normalize = True):
    
        val_path = Path(val_dataset)
        assert val_path.is_file()
        res_tuple = (resolution, resolution)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        
        if normalize:
            image_pipeline = [
                cropper,
                ToTensor(),
                ToDevice(ch.device(device), non_blocking=True),
                ToTorchImage(),
                NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
            ]
        else:
            image_pipeline = [
                cropper,
                ToTensor(),
                ToDevice(ch.device(device), non_blocking=True),
                ToTorchImage()
            ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(device),
            non_blocking=True)
        ]
        
        loader = Loader(val_dataset,
                        indices=indices,
                        batch_size=batch_size,
                        batches_ahead=batches_ahead,
                        num_workers=num_workers,
                        order=OrderOption.SEQUENTIAL,
                        drop_last=False,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        custom_fields={
                            'image': RGBImageField,
                            'label': IntField,
                        },
                        distributed=distributed)
        return loader
    
class MeanScalarMetric(torchmetrics.Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_state('sum', default=ch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('count', default=ch.tensor(0), dist_reduce_fx='sum')

    def update(self, sample: ch.Tensor):
        self.sum += sample.sum()
        self.count += sample.numel()

    def compute(self):
        return self.sum.float() / self.count
    
