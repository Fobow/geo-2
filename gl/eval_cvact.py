import os, json, argparse
from datetime import datetime
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from timm.scheduler.cosine_lr import CosineLRScheduler
import signal


# ===== Import model =====
from Mapper.mapper import VGGTLocator, InfoNCELoss, gather_tensor, InfoNCELossWithTopK
from Mapper.geomap import GeoLocator
from dataset.vigor import VigorDatasetEval, VigorDatasetTrain
from dataset.cvact import CVACTDatasetTrain, CVACTDatasetEval, CVACTDatasetTest
from dataset.cvusa import CVUSADatasetEval, CVUSADatasetTrain
from gl.train.transforms import get_transforms_train, get_transforms_val
from gl.evaluate.cvusa_and_cvact import evaluate, calc_sim
from gl.train.trainer import train
import pickle
# ===================================

import wandb

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    print(f"[Rank {rank}] Starting DDP setup", flush=True)
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()


def train_worker(rank, world_size, config, debug=False):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    if rank == 0 and not debug:
        wandb.init(
            project=config['run_settings'].get('project', 'Geo4'),
            name=f"{config['run_settings']['run_name']}",
            config=config,
        )
    else:
        wandb.init(mode="disabled")  # disable logging for other ranks

    # === Output folder ===
    if not debug and rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{config['run_settings']['run_name']}_{timestamp}"
        output_dir = os.path.join("output", run_name)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
    else:
        output_dir = None

    # === Dataset setup ===
    dataset_name = config['dataset']['name'].lower()
    if dataset_name == 'act':
        img_size = config['dataset']['image_size']
        image_size_sat = (img_size, img_size)
        new_width = img_size * 2    
        new_hight = round((224 / 1232) * new_width)
        img_size_ground = (new_hight, new_width)

        sat_transforms_val, ground_transforms_val = get_transforms_val(image_size_sat,
                                img_size_ground
                                )
        reference_dataset_val = CVACTDatasetEval(data_folder=config['dataset']['path'],
                                             split="val",
                                             img_type="reference",
                                             transforms=sat_transforms_val,
                                             )
        query_dataset_val = CVACTDatasetEval(data_folder=config['dataset']['path'],
                                        split="val",
                                        img_type="query",    
                                        transforms=ground_transforms_val,
                                        )
        # avact test
        reference_dataset_test = CVACTDatasetTest(data_folder=config['dataset']['path'],
                                            img_type="reference",
                                            transforms=sat_transforms_val,
                                            )
        query_dataset_test = CVACTDatasetTest(data_folder=config['dataset']['path'],
                                    img_type="query",
                                    transforms=ground_transforms_val,
                                    )
        
        if rank == 0:
            print("Reference Images Val:", len(reference_dataset_val))
            print("Query Images Val:", len(query_dataset_val))
            print("Reference Images Test:", len(reference_dataset_test))
            print("Query Images Test:", len(query_dataset_test))
    else:
        raise NotImplementedError(f"Unsupported dataset {dataset_name}")

    # do not shuffle the train set if we do hard sample mining
    reference_sampler_val = DistributedSampler(reference_dataset_val, num_replicas=world_size, rank=rank, shuffle=False)
    query_sampler_val = DistributedSampler(query_dataset_val, num_replicas=world_size, rank=rank, shuffle=False)
    reference_sampler_test = DistributedSampler(reference_dataset_test, num_replicas=world_size, rank=rank, shuffle=False)
    query_sampler_test = DistributedSampler(query_dataset_test, num_replicas=world_size, rank=rank, shuffle=False)

    reference_dataloader_val  = DataLoader(reference_dataset_val,
                              batch_size=config['training']['batch_size'],
                              num_workers=4, sampler=reference_sampler_val, persistent_workers=True,  prefetch_factor=2,
                              pin_memory=True, drop_last=False)
    query_dataloader_val  = DataLoader(query_dataset_val,
                            batch_size=config['training']['batch_size'],
                            num_workers=4, sampler=query_sampler_val, persistent_workers=True,  prefetch_factor=2,
                            pin_memory=True, drop_last=False)
    reference_dataloader_test  = DataLoader(reference_dataset_test,
                              batch_size=config['training']['batch_size'],
                              num_workers=4, sampler=reference_sampler_test, persistent_workers=True,  prefetch_factor=2,
                              pin_memory=True, drop_last=False)
    query_dataloader_test  = DataLoader(query_dataset_test,
                            batch_size=config['training']['batch_size'],
                            num_workers=4, sampler=query_sampler_test, persistent_workers=True,  prefetch_factor=2,
                            pin_memory=True, drop_last=False)



    # === Model & loss ===
    # old model
    # model = VGGTLocator(embed_dim=config['model']['embed_dim'], head_dim=config['model']['head_dim'], 
    #                     n_query=config['model']['n_query'], n_head=config['model']['n_head'], 
    #                     n_attn=config['model']['n_attn'], device=device, 
    #                     multi_layer_token=config['training']['multi_layer_token'], use_conv=config['training']['use_conv'],
    #                     img_size=config['training']['img_size'],
    #                     model_path=config['training']['model_path']
    #                     )
    # new model
    model = GeoLocator (
        embed_dim=config['model']['embed_dim'],
        model_path=config['training']['model_path'],
        use_vggt = config['model']['use_vggt'],
    )
    if rank == 0:
        print(f"==>embed_dim: {config['model']['embed_dim']}. using multi_layer_token: {config['training']['multi_layer_token']}")
    model = model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    if config['dataset']['eval_val']:
        if rank == 0:
            print("\n{}[{}]{}".format(30*"-", "CVACT_VAL", 30*"-"))   
        r1_test = evaluate(config=config,
                            model=model,
                            device=device,
                            reference_dataloader=reference_dataloader_val,
                            query_dataloader=query_dataloader_val, 
                            ranks=[1, 5, 10],
                            step_size=1000,
                            cleanup=True)
    if config['dataset']['eval_test']:
        if rank == 0:
            print("\n{}[{}]{}".format(30*"-", "CVACT_TEST", 30*"-")) 
        r1_test = evaluate(config=config,
                        model=model,
                        reference_dataloader=reference_dataloader_test,
                        query_dataloader=query_dataloader_test, 
                        ranks=[1, 5, 10],
                        step_size=1000,
                        cleanup=True)
    
    cleanup()

# # === Launcher ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    world_size = torch.cuda.device_count()
    
    try:
        if world_size > 1:
            mp.spawn(train_worker, args=(world_size, config, args.debug), nprocs=world_size, join=True)
        else:
            train_worker(0, 1, config, True)
    except KeyboardInterrupt:
        print("\n[!] Caught Ctrl+C — terminating all DDP workers...\n")
        os.killpg(0, signal.SIGKILL)