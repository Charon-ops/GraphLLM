# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# os.environ['WANDB_DISABLED'] = True
# os.environ['TOKENIZERS_PARALLELISM'] = False

from core.LMs.lm_trainer import LMTrainer
from core.config import cfg, update_cfg
import pandas as pd


def run(cfg):
    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)
    all_acc = []
    for seed in seeds:
        cfg.seed = seed
        trainer = LMTrainer(cfg)
        trainer.train()
        acc = trainer.eval_and_save()
        all_acc.append(acc)

    if len(all_acc) > 1:
        df = pd.DataFrame(all_acc)
        for k, v in df.items():
            print(f"{k}: {v.mean():.4f} ± {v.std():.4f}")


if __name__ == '__main__':
    cfg = update_cfg(cfg)
    run(cfg)

'''
windows debug command:
set WANDB_DISABLED=True&& set TOKENIZERS_PARALLELISM=False&& set CUDA_VISIBLE_DEVICES=0,1,2,3&& python -m debugpy --listen 5678 --wait-for-client core/trainLM.py dataset cora
'''