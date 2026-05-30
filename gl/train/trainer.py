import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
import torch.nn.functional as F
from Mapper.mapper import InfoNCELoss, InfoNCELossWithTopK, gather_tensor
import torch.distributed as dist
import wandb

def train(config, device, model, dataloader, loss_fn, optimizer, scheduler=None, scaler=None, epoch=None):

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        print("==> error: cannot get rank")
        return

    # set model train mode
    model.train()
    losses = AverageMeter()
    # wait before starting progress bar
    time.sleep(0.1)
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
    if rank == 0:
        bar = tqdm(dataloader, total=len(dataloader))
    # for loop over one epoch
    # id == label
    for i, (ground_img, sat_img, ground_view, sat_view, label) in enumerate(dataloader):
        
        if scaler:
            with autocast():
                # data (batches) to device   
                query = ground_img.to(device)
                reference = sat_img.to(device)
                ground_view = ground_view.to(device)
                sat_view = sat_view.to(device)
                # Forward pass
                # features1, features2 = model(query, reference)
                
                # only for inference, will lead to gradient bug
                # z_grd = model(query, ground_view, mode="grd")
                # z_sat = model(reference, sat_view, mode="sat")
                # safe for training
                z_grd = model.module.encode_grds(query, ground_view)
                z_sat = model.module.encode_sats(reference, sat_view)
                if config['training']['fix_logit']:
                    loss, all_grd, all_sat = loss_fn(z_grd, z_sat, 1/0.07)
                else:
                    loss, all_grd, all_sat = loss_fn(z_grd, z_sat, model.module.logit_scale.exp())

                losses.update(loss.item())
                
                  
            scaler.scale(loss).backward()
            
            # Gradient clipping 
            if config['training']['clip_grad']:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), config['training']['clip_grad']) 
            
            # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            # if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
            # using cosine by default
            scheduler.step(epoch)
   
        else:
        
            # data (batches) to device   
            query = query.to(device)
            reference = reference.to(device)

            # Forward pass
            # z_grd = model(query, ground_view, mode="grd")
            # z_sat = model(reference, sat_view, mode="sat")
            z_grd = model.module.encode_grds(query, ground_view)
            z_sat = model.module.encode_sats(reference, sat_view)
            if config['training']['fix_logit']:
                loss, all_grd, all_sat = loss_fn(z_grd, z_sat, 1/0.07)
            else:
                loss, all_grd, all_sat = loss_fn(z_grd, z_sat, model.module.logit_scale.exp())

            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()
            
            # Gradient clipping 
            if config['training']['clip_grad']:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), config['training']['clip_grad'])             
            
            # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            # if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
            # using cosine by default
            scheduler.step(epoch)
        
        
        if rank == 0:
            monitor = {"loss": "{:.4f}".format(loss.item()),
                        "loss_avg": "{:.4f}".format(losses.avg),
                        "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            bar.set_postfix(ordered_dict=monitor)
            bar.update(1)
        
        step += 1

    if rank == 0:
        bar.close()

    return losses.avg


def predict(config, model, dataloader, device, mode="sat"):
    
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        print("==> error: cannot get rank")
        return
    
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if rank == 0:
        bar = tqdm(dataloader, total=len(dataloader))

    img_features_list = []
    
    ids_list = []
    with torch.no_grad():
        
        # for img, view, ids in bar:
        for i, (img, view, ids) in enumerate(dataloader):
        
            ids_list.append(ids)
            
            with autocast():
                img = img.to(device)
                img_feature = model(img, view, mode=mode)
            
                # normalize is calculated in fp32
                # if train_config.normalize_features:
                # normalize by default
                img_feature = F.normalize(img_feature, dim=-1)
            
            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))
      
        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0) 
        ids_list = torch.cat(ids_list, dim=0).to(device)
        
        all_feats = gather_tensor(img_features)
        all_ids = gather_tensor(ids_list)
    
    if rank == 0:
        bar.close()
    return all_feats, all_ids
    # return img_features, ids_list