
import os
import json
import torch
import torch.nn as nn
import argparse
import numpy as np
import matplotlib
from matplotlib import pyplot as plt 
from tensorboardX import SummaryWriter

import src.utils as utils
from src.model_utils import build_model

from src.basketball.dataset import (
    make_basketball_dataset_train, 
    make_basketball_dataset_test__as_list,
)

debug_mode_for_training = False
debug_mode_for_forecasting = False
verbose_logging = True 



def train_step(batch, model, optimizer, step, config, force_breakpoint=False):
    model.train()

    def _set_lr(lr):
        for g in optimizer.param_groups:
            g["lr"] = lr

    switch_temp = utils.get_temperature(step, config, "switch_")
    extra_args = dict()
    dur_temp = 1.0
    if config["model"] == "REDSDS":
        dur_temp = utils.get_temperature(step, config, "dur_")
        extra_args = {"dur_temperature": dur_temp}
    lr = utils.get_learning_rate(step, config)
    xent_coeff = utils.get_cross_entropy_coef(step, config)
    cont_ent_anneal = config["cont_ent_anneal"]
    optimizer.zero_grad()
    result = model(
        batch,
        switch_temperature=switch_temp,
        num_samples=config["num_samples"],
        cont_ent_anneal=cont_ent_anneal,
        force_breakpoint=force_breakpoint,
        **extra_args,
    )
    objective = -1 * (
        result[config["objective"]] + xent_coeff * result["crossent_regularizer"]
    )
    print(
        step,
        f"obj: {objective.item():.4f}",
        f"lr: {lr:.6f}",
        f"s-temp: {switch_temp:.2f}",
        f"cross-ent: {xent_coeff}",
        f"cont ent: {cont_ent_anneal}",
    )
    loss_history[s]=objective.item()
    objective.backward()
    nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
    _set_lr(lr)
    optimizer.step()
    result["objective"] = objective
    result["lr"] = lr
    result["switch_temperature"] = switch_temp
    result["dur_temperature"] = dur_temp
    result["xent_coeff"] = xent_coeff
    return result


def plot_some_reconstructions_and_forecasts(config, model, device, player_idx, step):
    S = config["forecast"]["num_samples"]
    T_pred =config["prediction_length"]
    # Get the preset 78 examples test set examples with hardcoded start/stop indices.    
    test_dataset__list = make_basketball_dataset_test__as_list(n_players=config["n_players"], player_idx=player_idx) 
    
    for e in range(5):
        test_example =test_dataset__list[e]
        test_example = test_example.to(device)
        result_dict = model.predict(
                test_example.to(device),
                num_samples=config["forecast"]["num_samples"],
                basketball=True,
        )
        rec_y_with_forecast = result_dict["rec_y_with_forecast"][:,0]

        for s in range(3):
            context_rec=rec_y_with_forecast[s,:-T_pred]
            context_true=test_example[0,:-T_pred]

            forecast_pred=rec_y_with_forecast[s,-T_pred:]
            forecast_true=test_example[0,-T_pred:]

            T_context=len(context_true)
            T_forecast=len(forecast_true)

            # y (cool): light blue to pink
            # forecast (viridis): purple to yellow

            plt.scatter(context_true[:,0], context_true[:,1], c=np.arange(T_context), cmap='cool', s=30, marker="x")
            plt.scatter(context_rec[:,0],context_rec[:,1], c=np.arange(T_context), cmap='cool', s=30)
            plt.scatter(forecast_true[:,0], forecast_true[:,1], c=np.arange(T_forecast), cmap='viridis', s=30, marker="x")
            plt.scatter(forecast_pred[:,0],forecast_pred[:,1], c=np.arange(T_forecast), cmap='viridis', s=30)
            #plt.xlim(0, 1)
            #plt.ylim(0, 1)
            #plt.show()
            #input("")
            #plt.close("all")

            plot_path=os.path.join(config["log_dir"], f"context_and_reconstruction_{n_train_games}__player_{player_idx}____step_{step}__example_{e}__sample_{s}.pdf")
            plt.savefig(plot_path)
            plt.close("all")

def make_test_forecasts(config, model, device, n_players, player_idx):
    # setup format for return value
    E,D = 78,2
    J = n_players
    S = config["forecast"]["num_samples"]
    T_pred =config["prediction_length"]
    test_forecasts = np.zeros((E, S, T_pred, J, D ))

    # Get the preset 78 examples test set examples with hardcoded start/stop indices.    
    test_dataset__list = make_basketball_dataset_test__as_list(n_players=config["n_players"], player_idx=player_idx) 
    for e, test_example in enumerate(test_dataset__list):
        test_example = test_example.to(device)
        result_dict = model.predict(
                test_example.to(device),
                num_samples=config["forecast"]["num_samples"],
                basketball=True,
        )
        # Rk: I don't think we have any use for result_dict["z_emp_probs"]?
        #       This seems to be the z probs in the forecast range.

        # here we reshape the array.  in the return value
        # one example has dimension (S, batch_dim=1, T_pred, JXD).
        # as implied by the init above, we want the whole thing to be (E, S, T_pred, J, D ).
        # test_forecasts__reversed =result_dict["forecast"][:,0].reshape(S,  T_pred, J,D )
        # test_forecasts[e]=test_forecasts__reversed[:,torch.arange(T_pred - 1, -1, -1),:,:]
        test_forecasts[e]=result_dict["forecast"][:,0].reshape(S,  T_pred, J,D )
    
    return test_forecasts


if __name__ == "__main__":
    matplotlib.use("Agg")

    # COMMAND-LINE ARGS
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=str, help="Path to config file.")
    group.add_argument("--ckpt", type=str, help="Path to checkpoint file.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Which device to use, e.g., cpu, cuda:0, cuda:1, ...",
    )
    
    # Actual CLI input
    # args = parser.parse_args()

    # Simulate CLI input
    args = parser.parse_args(["--config", "configs/basketball.yaml", "--device", "cpu"])

    # Inspect
    print(args)

    # CONFIG
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu")
        config = ckpt["config"]
    else:
        config = utils.get_config_and_setup_dirs(args.config)
    device = torch.device(args.device)
    with open(os.path.join(config["log_dir"], "config.json"), "w") as fp:
        json.dump(config, fp)

    if config["n_players"]!=1:
        raise NotImplementedError(f"This loop now assumes we're running models for one player at a time. "
                                  f"Also the forecasting method with n=10 players seems to be broken. ")

    for n_train_games in [1,5,20]:
        # TODO: put this back in the correct order.
        for player_idx in [0,1,2,3,4,5,6,7,8,9]:
        
            # TODO: make traj_length a parameter... Did we use this in other places?
            train_dataset = make_basketball_dataset_train(n_train_games=n_train_games, traj_length=30, n_players=config["n_players"], player_idx=player_idx) 
            # TODO: check if we can run basketball with more workers.
            num_workers = 0 
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=config["batch_size"],
                num_workers=num_workers,
                shuffle=True,
                pin_memory=True,
            )

            train_gen = iter(train_loader)

            print(f'Running {config["model"]} on {config["dataset"]}.')
            print(f"Train size: {len(train_dataset)}.")

            # MODEL
            model = build_model(config=config)
            start_step = 1
            if args.ckpt:
                model.load_state_dict(ckpt["model"])
                start_step = ckpt["step"] + 1
            model = model.to(device)

            for n, p in model.named_parameters():
                print(n, p.size())

            # TRAIN
            optimizer = torch.optim.Adam(
                model.parameters(), weight_decay=config["weight_decay"]
            )
            if args.ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            summary = SummaryWriter(logdir=config["log_dir"])

            n_steps=(config["num_steps"] + 1)-start_step
            loss_history = np.zeros(n_steps)

            for s,step in enumerate(range(start_step, config["num_steps"] + 1)):
                try:
                    train_batch  = next(train_gen)
                    train_batch = train_batch.to(device)
                except StopIteration:
                    train_gen = iter(train_loader)
                train_result = train_step(train_batch, model, optimizer, step, config)

                if step==start_step:
                    train_batch_fixed=train_batch
                if step % config["save_steps"] == 0 or step == config["num_steps"]:
                    model_path = os.path.join(config["model_dir"], f"n_train_{n_train_games}__model_player_{player_idx}_step_{step}.pt")
                    torch.save(
                        {
                            "step": step,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "config": config,
                        },
                        model_path,
                    )

                if step % config["log_steps"] == 0 or step == config["num_steps"]:
                    summary_items = {
                        "params/learning_rate": train_result["lr"],
                        "params/switch_temperature": train_result["switch_temperature"],
                        "params/dur_temperature": train_result["dur_temperature"],
                        "params/cross_entropy_coef": train_result["xent_coeff"],
                        "elbo/training": train_result[config["objective"]],
                        "xent/training": train_result["crossent_regularizer"],
                    }

                    # sanity check for training
                    if debug_mode_for_training:
                        train_step(train_batch_fixed, model, optimizer, step, config, force_breakpoint=True)
                        breakpoint()
                        # see devel_check_reconstruct.py when force_breakpoint=True above.

                    if debug_mode_for_forecasting:
                        plot_some_reconstructions_and_forecasts(config, model, device, player_idx, step)
                        test_forecasts=make_test_forecasts(config, model, device, n_players=config["n_players"], player_idx=player_idx)
                        # see devel_check_forecasts__one_player.py if we have one player.
                        breakpoint()

                    if verbose_logging:
                        # Save loss history (negative ELBO)
                        loss_history_path=os.path.join(config["log_dir"], f"loss_history__n_train_{n_train_games}__player_{player_idx}__step_{step}.npy")
                        np.save(loss_history_path, loss_history)
                    
                        # Make plot of loss and save to disk.
                        plt.plot((loss_history-min(loss_history))[:step])
                        plt.yscale('log')  # log scale for y-axis
                        plt.xlabel("Step")
                        plt.ylabel("Log (loss - min loss)")
                        loss_plot_path=os.path.join(config["log_dir"], f"loss_history__n_train_{n_train_games}__player_{player_idx}__step_{step}.pdf")
                        plt.savefig(loss_plot_path)
                        plt.close("all")

                        # Save test set forecasts 
                        model_name=config["model"]
                        forecasts_path=os.path.join(config["log_dir"], f"forecasts_test__{model_name}__n_train_{n_train_games}__step_{step}_player_{player_idx}.npy")
                        test_forecasts=make_test_forecasts(config, model, device, n_players=config["n_players"], player_idx=player_idx)
                        np.save(forecasts_path, test_forecasts)

                        # plot some reconstructions and forecasts
                        plot_some_reconstructions_and_forecasts(config, model, device, player_idx, step)
                        
            # Save loss history (negative ELBO)
            loss_history_path=os.path.join(config["train_history_dir"], f"loss_history__n_train_{n_train_games}__player_{player_idx}__step_{step}.npy")
            np.save(loss_history_path, loss_history)
        
            # Make plot of loss and save to disk.
            plt.plot(loss_history-min(loss_history))
            plt.yscale('log')  # log scale for y-axis
            plt.xlabel("Step")
            plt.ylabel("Log (loss - min loss)")
            loss_plot_path=os.path.join(config["train_history_dir"], f"loss_history__n_train_{n_train_games}__player_{player_idx}__step_{step}.pdf")
            plt.savefig(loss_plot_path)
            plt.close("all")

            # To see the plot:
            #import matplotlib
            #matplotlib.use("Qt5Agg") 
            #plt.show()


            # Make and save test set forecasts 
            test_forecasts=make_test_forecasts(config, model, device, n_players=config["n_players"], player_idx=player_idx)
            model_name=config["model"]
            forecasts_path=os.path.join(config["forecasts_dir"], f"forecasts_test__{model_name}__n_train_{n_train_games}__step_{step}__player_{player_idx}.npy")
            np.save(forecasts_path, test_forecasts)



    