import os
import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import SegformerForSemanticSegmentation
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import pandas as pd
import numpy as np

from ml.biomarcers.config import Config
from ml.biomarcers.dataloader import ImageMaskDataset
from ml.biomarcers.utils_loss import TverskyLoss

config = Config()

def dice_score_fast(preds, targets, ignore_index=config.IGNORE_INDEX):
    """Быстрый Dice для мультикласса"""
    preds_labels = preds.argmax(dim=1)
    num_classes = preds.shape[1]
    
    dice_sum = 0.0
    count = 0
    eps = 1e-6
    
    for cls in range(1, num_classes):
        pred_mask = (preds_labels == cls)
        target_mask = (targets == cls)
        valid_mask = (targets != ignore_index)
        pred_mask = pred_mask & valid_mask
        target_mask = target_mask & valid_mask
        
        TP = (pred_mask & target_mask).sum().item()
        FP = (pred_mask & (~target_mask)).sum().item()
        FN = ((~pred_mask) & target_mask).sum().item()
        
        if TP + FP + FN == 0:
            continue
        
        dice_cls = (2 * TP + eps) / (2 * TP + FP + FN + eps)
        dice_sum += dice_cls
        count += 1
    
    if count == 0:
        return 0.0
    return dice_sum / count

def reduce_tensor(tensor, world_size):
    """Синхронизация тензора между всеми GPU"""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt / world_size

def setup_ddp(rank, world_size):
    """Инициализация DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    if rank == 0:
        print(f"[GPU {rank}] Инициализирован, всего GPU: {world_size}")

def cleanup_ddp():
    """Очистка DDP"""
    dist.destroy_process_group()

def train_fold(rank, world_size, train_folds, val_fold, patience=5):
    """Обучение с DDP"""
    
    # Инициализация DDP
    setup_ddp(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    
    # Загрузка данных (только на rank 0 загружаем CSV, остальные получат через broadcast)
    if rank == 0:
        train_dfs = []
        for f in train_folds:
            csv_path = f"/kaggle/input/datasets/andreikarabin/data-filter/aspirantura/PROF/npy_article_fold/train_article_fold_{f}.csv"
            train_dfs.append(pd.read_csv(csv_path))
        df_train = pd.concat(train_dfs).reset_index(drop=True)
        df_val = pd.read_csv(f"/kaggle/input/datasets/andreikarabin/data-filter/aspirantura/PROF/npy_article_fold/train_article_fold_{val_fold}.csv")
        
        # Исправление путей
        BASE_PATH = "/kaggle/input/datasets/andreikarabin/data-filter/aspirantura/PROF/npy_article_fold"
        for df in [df_train, df_val]:
            df["image"] = df["image"].str.replace(r"D:.*npy_article_fold", BASE_PATH, regex=True)
            df["mask"] = df["mask"].str.replace(r"D:.*npy_article_fold", BASE_PATH, regex=True)
            df["image"] = df["image"].str.replace("\\", "/")
            df["mask"] = df["mask"].str.replace("\\", "/")
    else:
        df_train = None
        df_val = None
    
    # Broadcast данных на все GPU
    # Для простоты загружаем датасеты на каждом GPU отдельно (т.к. читаем из файлов)
    # Но если данные общие, можно использовать broadcast
    if rank != 0:
        train_dfs = []
        for f in train_folds:
            csv_path = f"/kaggle/input/datasets/andreikarabin/data-filter/aspirantura/PROF/npy_article_fold/train_article_fold_{f}.csv"
            train_dfs.append(pd.read_csv(csv_path))
        df_train = pd.concat(train_dfs).reset_index(drop=True)
        df_val = pd.read_csv(f"/kaggle/input/datasets/andreikarabin/data-filter/aspirantura/PROF/npy_article_fold/train_article_fold_{val_fold}.csv")
        
        # Исправление путей
        BASE_PATH = "/kaggle/input/datasets/andreikarabin/data-filter/aspirantura/PROF/npy_article_fold"
        for df in [df_train, df_val]:
            df["image"] = df["image"].str.replace(r"D:.*npy_article_fold", BASE_PATH, regex=True)
            df["mask"] = df["mask"].str.replace(r"D:.*npy_article_fold", BASE_PATH, regex=True)
            df["image"] = df["image"].str.replace("\\", "/")
            df["mask"] = df["mask"].str.replace("\\", "/")
    
    # Датасеты
    train_dataset = ImageMaskDataset(df_train, augment_prob=0.5)
    val_dataset = ImageMaskDataset(df_val, augment_prob=0.0)
    
    # Sampler для DDP
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=True,
        drop_last=True  # Важно добавить для DDP
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE,
        sampler=train_sampler,
        shuffle=False,
        num_workers=4, 
        pin_memory=True,
        drop_last=True  # Важно для DDP
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    
    # Модель
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=config.NUM_CLASSES,
        ignore_mismatched_sizes=True
    ).to(device)
    
    # Оборачиваем в DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW([
        {'params': model.module.segformer.encoder.parameters(), 'lr': 1e-5},
        {'params': model.module.decode_head.parameters(), 'lr': 5e-4},
    ])
    
    gradient_accumulation_steps = 2
    num_training_steps = (len(train_loader) // gradient_accumulation_steps) * config.EPOCHS
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=config.IGNORE_INDEX).to(device)
    tversky_loss = TverskyLoss(ignore_index=config.IGNORE_INDEX).to(device)
    
    def combined_loss(logits, targets):
        return ce_loss(logits, targets) + 2.0 * tversky_loss(logits, targets)
    
    scaler = torch.cuda.amp.GradScaler()
    best_dice = 0.0
    epochs_no_improve = 0
    
    # Только главный процесс (rank=0) создает папки и выводит логи
    if rank == 0:
        checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, f"fold_{val_fold}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Train Folds: {train_folds}, Val Fold: {val_fold}")
        print(f"Batch size per GPU: {config.BATCH_SIZE}")
        print(f"Effective batch size: {config.BATCH_SIZE * world_size}")
        print(f"Number of training batches per GPU: {len(train_loader)}")
    
    for epoch in range(config.EPOCHS):
        # Важно: устанавливаем эпоху для sampler
        train_sampler.set_epoch(epoch)
        
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        # Синхронизация времени для всех GPU
        if rank == 0:
            loader_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch")
        else:
            loader_iter = train_loader
        
        optimizer.zero_grad()
        for i, (imgs, masks) in enumerate(loader_iter):
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            with torch.cuda.amp.autocast():
                outputs = model(pixel_values=imgs)
                logits = outputs.logits if not isinstance(outputs, dict) else outputs['logits']
                logits = F.interpolate(logits, masks.shape[-2:], mode="bilinear", align_corners=False)
                loss = combined_loss(logits, masks)
                loss = loss / gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            # Синхронизируем loss для отображения
            running_loss += loss.item() * gradient_accumulation_steps
            
            if rank == 0:
                loader_iter.set_postfix({"avg_loss": running_loss / (i+1)})
        
        # Синхронизируем финальный loss между GPU для точности
        loss_tensor = torch.tensor([running_loss / len(train_loader)]).to(device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        epoch_loss = loss_tensor.item() / world_size
        
        if rank == 0:
            print(f"Epoch {epoch+1} finished, Avg Loss: {epoch_loss:.4f}, Time: {time.time()-start_time:.1f}s")
        
        # Валидация (с синхронизацией метрик между GPU)
        model.eval()
        all_dice = 0.0
        count = 0
        
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            with torch.no_grad():
                outputs = model(pixel_values=imgs)
                logits = outputs.logits if not isinstance(outputs, dict) else outputs['logits']
                logits = F.interpolate(logits, masks.shape[-2:], mode="bilinear", align_corners=False)
                
                # Вычисляем Dice на каждом GPU
                d = dice_score_fast(logits, masks)
                
                # Синхронизируем Dice между всеми GPU
                d_tensor = torch.tensor([d]).to(device)
                dist.all_reduce(d_tensor, op=dist.ReduceOp.SUM)
                d_sync = d_tensor.item() / world_size
                
                all_dice += d_sync
                count += 1
        
        avg_dice = all_dice / count
        
        # Только rank 0 выводит логи и сохраняет модели
        if rank == 0:
            print(f"Validation Dice: {avg_dice:.4f}")
            
            # Сохранение чекпоинта
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss,
                "val_dice": avg_dice
            }, checkpoint_path)
            
            if avg_dice > best_dice:
                best_dice = avg_dice
                epochs_no_improve = 0
                best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save({
                    "epoch": epoch+1,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss,
                    "val_dice": avg_dice
                }, best_model_path)
                print(f"Best model updated! Dice: {best_dice:.4f}")
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Барьер для синхронизации всех процессов перед следующей эпохой
        dist.barrier()
    
    if rank == 0:
        print(f"Training finished. Best Dice: {best_dice:.4f}")
    
    cleanup_ddp()

def main():
    # Параметры фолдов
    FOLD = 1
    
    if FOLD == 1:
        train_folds = [1, 3]
        val_fold = 2
    elif FOLD == 2:
        train_folds = [1, 2]
        val_fold = 3
    elif FOLD == 3:
        train_folds = [2, 3]
        val_fold = 1
    else:
        raise ValueError("FOLD должен быть 1, 2 или 3")
    
    world_size = torch.cuda.device_count()
    print(f"Запуск с {world_size} GPU")
    
    import torch.multiprocessing as mp
    mp.spawn(train_fold, args=(world_size, train_folds, val_fold, 5), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()