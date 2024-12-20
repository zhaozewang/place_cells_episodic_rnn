import os
import nn4n
import torch
import hydra
import argparse

from termcolor import colored
from omegaconf import OmegaConf
from os.path import join as pjoin
from trial_manager import TrialManager
from model import PlaceCellsEpisodicRNN
from trainer import Trainer


def load_config(cfg_path_abs):
    """
    Load the configuration files for the model, loss, and experiment.
    
    Parameters:
        - cfg_path_abs (str): The abs path to the configuration folder. 
            It should contains the following files:
                - model.yaml
                - loss.yaml
                - gym.yaml
                - experiment.yaml
    """
    cfg = {}
    with hydra.initialize_config_dir(str(cfg_path_abs), version_base="1.2"):
        cfg["model"] = hydra.compose(config_name="model")
        cfg["loss"] = hydra.compose(config_name="loss")
        cfg["gym"] = hydra.compose(config_name="gym")
        cfg["experiment"] = hydra.compose(config_name="experiment")

    # Add the dimension configuration to the model configuration
    OmegaConf.set_struct(cfg["model"], False)  # Allow dynamic addition of keys
    sensory_num = cfg["gym"]["sensory"]["default_sensory_group"]["n_cells"]
    cfg["model"].update({
        "input_dim": sensory_num,
        "output_dim": sensory_num
    })
    OmegaConf.set_struct(cfg["model"], True)  # Disallow dynamic addition of keys

    return cfg


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Training/Testing script for the RAE of episodic memory that emerges place cells.")
    parser.add_argument("--cfg", type=str, help="Config folder name (under the 'config/' folder)", default="default")
    parser.add_argument("--ckpt", type=str, help="Checkpoint file name (under the 'checkpoint/' folder)", default="final.pth")
    parser.add_argument("-t", "--train", action="store_true", help="Train the model")
    parser.add_argument("-T", "--test", action="store_true", help="Test the model")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    
    args = parser.parse_args()
    
    # Paths
    project_path = os.path.dirname(os.path.realpath(__file__))
    cfg_path = pjoin("place_cell_rae", "configs", args.cfg)
    ckpt_path = pjoin("place_cell_rae", "ckpts", args.ckpt)
    cfg_path_abs = pjoin(project_path, cfg_path)
    ckpt_path_abs = pjoin(project_path, ckpt_path)
    
    # If training, load the configuration files
    if args.train:
        if not os.path.isdir(cfg_path_abs):
            print(f"Error: Config folder './{cfg_path}' does not exist. Please specify a valid config file.")
            exit(1)
        print(colored(f"Training initiated with config folder: ./{cfg_path}", "green"))
        cfg = load_config(cfg_path_abs)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PlaceCellsEpisodicRNN(cfg["model"])
        model.to(device)

        loss = nn4n.criterion.CompositeLoss(cfg["loss"])
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["model"]["lr"])
        
        trial_manager = TrialManager(
            gym_cfg=cfg["gym"], 
            expt_cfg=cfg["experiment"],
            device=device
        )
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss=loss,
            trial_manager=trial_manager
        )
        trainer.train()

    # If testing, load the checkpoint file
    elif args.test:
        if not os.path.isfile(ckpt_path_abs):
            print(colored(f"Error: Checkpoint file './{ckpt_path}' does not exist. Please specify a valid checkpoint file.", "red"))
            exit(1)
        print(colored(f"Testing initiated with checkpoint file: ./{ckpt_path}", "green"))
        # TODO: testing logic here

    # If neither --train nor --test is specified
    else:
        print(colored("Error: You must specify either --train or --test.", "red"))
        exit(1)

if __name__ == "__main__":
    main()
