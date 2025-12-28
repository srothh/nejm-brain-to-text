from omegaconf import OmegaConf
from pretraining import PretrainingPhoneme_Trainer

args = OmegaConf.load('pretrain_args.yaml')
trainer = PretrainingPhoneme_Trainer(args)
metrics = trainer.train()
