from omegaconf import OmegaConf
from e2e_trainer import End2EndModel_Trainer

args = OmegaConf.load('rnn_args.yaml')
trainer = End2EndModel_Trainer(args)
metrics = trainer.train()