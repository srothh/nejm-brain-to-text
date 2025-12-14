from omegaconf import OmegaConf
from e2e_trainer import End2EndModel_Trainer

args = OmegaConf.load('model_training/whisper/e2e_args.yaml')
trainer = End2EndModel_Trainer(args)
metrics = trainer.train()