import os
import torch
import torch.nn as nn
import gc
from fastprogress import master_bar,progress_bar
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torchattacks
from torchvision import models
from collections import defaultdict
import pandas as pd
from pdb import set_trace
from jsputils import classes

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        corrects = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
            corrects.append(correct[:k].any(dim=0).reshape(-1).float())
        return pred, *corrects, *res

def validate_attack(model, atk, num_classes = 1000,
                    device=None, print_freq=100, mb=None, store_outputs=False):
    if device is None: device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    ValLoader = classes.DataLoaderFFCV('val')
    
    model = model.float()

    def run_validate(loader):
        results = defaultdict(list)
        count = 0
        for i, batch in enumerate(progress_bar(loader, parent=mb)):
            batch_size = batch[0].shape[0]
            images = batch[0].to(device, non_blocking=True).float()
            target = batch[1].to(device, non_blocking=True)
            
            images = torch.clamp(images, 0, 1)# Ensure images are in [0, 1] range

            adv_images = atk(images, target)
            
            with torch.no_grad():
                output_orig = model(images)
                output_atk = model(adv_images)

            print( (target.cpu()==output_orig.cpu().argmax(dim=1)).float().mean() )
            print( (target.cpu()==output_atk.cpu().argmax(dim=1)).float().mean() )

            loss_orig = criterion(output_orig, target)
            loss_atk = criterion(output_atk, target)

            # measure accuracy and record loss
            preds_orig, correct1_orig, correct5_orig, _, _ = accuracy(output_orig, target, topk=(1, 5))
            preds_atk, correct1_atk, correct5_atk, _, _ = accuracy(output_atk, target, topk=(1, 5))

            results['image_set'] += ['original'] * batch_size
            results['label'] += target.tolist()
            results['loss'] += loss_orig.tolist()
            results['pred_label'] += preds_orig[0].tolist()
            results['correct1'] += correct1_orig.tolist()
            results['correct5'] += correct5_orig.tolist()

            results['image_set'] += ['adversarial'] * batch_size
            results['label'] += target.tolist()
            results['loss'] += loss_atk.tolist()
            results['pred_label'] += preds_atk[0].tolist()
            results['correct1'] += correct1_atk.tolist()
            results['correct5'] += correct5_atk.tolist()

        df = pd.DataFrame(results)

        return df

    # switch to evaluate mode
    model.eval()
    model.to(device)

    df = run_validate(ValLoader.data_loader)

    return df

def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = models.alexnet(pretrained=True)
    model.to(device)

    return model