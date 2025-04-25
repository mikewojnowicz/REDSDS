# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import os
import json
import torch
import torch.nn as nn
import argparse
import numpy as np
import matplotlib
from tensorboardX import SummaryWriter
import src.utils as utils
from src.model_utils import build_model

from src.basketball.dataset import (
    make_basketball_dataset_train, 
    make_basketball_dataset_test__as_list,
)

available_datasets = {"basketball"}


def train_step(batch, model, optimizer, step, config):
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


# def plot_results(result, prefix=""):
#     original_inputs = torch2numpy(result["inputs"][0])
#     reconstructed_inputs = torch2numpy(result["reconstructions"][0])
#     most_likely_states = torch2numpy(torch.argmax(result["log_gamma"], dim=-1)[0][0])
#     hidden_states = torch2numpy(result["x_samples"][0])
#     discrete_states_lk = torch2numpy(torch.exp(result["log_gamma"][0])[0])
#     true_seg = None
#     if "true_seg" in result:
#         true_seg = torch2numpy(result["true_seg"][0, : config["context_length"]])

#     ylim = 1.3 * np.abs(original_inputs).max()
#     matplotlib_fig = tensorboard_utils.show_time_series(
#         fig_size=(12, 4),
#         inputs=original_inputs,
#         reconstructed_inputs=reconstructed_inputs,
#         segmentation=most_likely_states,
#         true_segmentation=true_seg,
#         fig_title="input_reconstruction",
#         ylim=(-ylim, ylim),
#     )
#     fig_numpy_array = tensorboard_utils.plot_to_image(matplotlib_fig)
#     summary.add_image(
#         f"{prefix}Reconstruction", fig_numpy_array, step, dataformats="HWC"
#     )

#     matplotlib_fig = tensorboard_utils.show_hidden_states(
#         fig_size=(12, 3), zt=hidden_states, segmentation=most_likely_states
#     )
#     fig_numpy_array = tensorboard_utils.plot_to_image(matplotlib_fig)
#     summary.add_image(
#         f"{prefix}Hidden_State_xt", fig_numpy_array, step, dataformats="HWC"
#     )

#     matplotlib_fig = tensorboard_utils.show_discrete_states(
#         fig_size=(12, 3),
#         discrete_states_lk=discrete_states_lk,
#         segmentation=most_likely_states,
#     )
#     fig_numpy_array = tensorboard_utils.plot_to_image(matplotlib_fig)
#     summary.add_image(
#         f"{prefix}Discrete_State_zt", fig_numpy_array, step, dataformats="HWC"
#     )



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
    #args = parser.parse_args(["--config", "configs/bee_duration.yaml", "--device", "cpu"])


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

    # DATA
    # TODO: Add other training lengths besides one game
    # TODO: make traj_length a parameter... Did we use this in other places?
    train_dataset = make_basketball_dataset_train(data_type="train_1", traj_length=30) 
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

    for step in range(start_step, config["num_steps"] + 1):
        try:
            train_batch  = next(train_gen)
            train_batch = train_batch.to(device)
        except StopIteration:
            train_gen = iter(train_loader)
        train_result = train_step(train_batch, model, optimizer, step, config)

        if step % config["save_steps"] == 0 or step == config["num_steps"]:
            model_path = os.path.join(config["model_dir"], f"model_{step}.pt")
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
            #train_result["true_seg"] = train_label
            #plot_results(train_result)

            # # Plot duration models
            # if config["model"] == "REDSDS":
            #     dummy_ctrls = torch.ones(1, 1, 1, device=device)
            #     rho = torch2numpy(
            #         model.ctrl2nstf_network.rho(
            #             dummy_ctrls, temperature=train_result["dur_temperature"]
            #         )
            #     )[0, 0]
            #     matplotlib_fig = tensorboard_utils.show_duration_dists(
            #         fig_size=(15, rho.shape[0] * 2), rho=rho
            #     )
            #     fig_numpy_array = tensorboard_utils.plot_to_image(matplotlib_fig)
            #     summary.add_image("Duration", fig_numpy_array, step, dataformats="HWC")

            # if step == config["num_steps"]:
            #     # Evaluate Forecast
            #     agg_metrics = evaluate_gts_dataset(
            #         test_dataset,
            #         model,
            #         device=device,
            #         num_samples=config["forecast"]["num_samples"],
            #         deterministic_z=config["forecast"]["deterministic_z"],
            #         deterministic_x=config["forecast"]["deterministic_x"],
            #         deterministic_y=config["forecast"]["deterministic_y"],
            #         max_len=np.inf,
            #         batch_size=100,
            #     )
            #     summary_items["metrics/test_mse"] = agg_metrics["MSE"]
            #     summary_items["metrics/CRPS"] = agg_metrics["mean_wQuantileLoss"]
            #     all_metrics["step"].append(step)
            #     all_metrics["CRPS"].append(agg_metrics["mean_wQuantileLoss"])
            #     all_metrics["MSE"].append(agg_metrics["MSE"])

    # Get the preset 78 examples test set examples with hardcoded start/stop indices.    
    test_dataset__list = make_basketball_dataset_test__as_list() 

    # setup format for return value
    E = 78
    J = 10 
    D = 2 
    S = config["forecast"]["num_samples"]
    T_pred =config["prediction_length"]
    test_forecasts = np.zeros((E, S, T_pred, J, D ))
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
        test_forecasts[e] =result_dict["forecast"][:,0].reshape(S,  T_pred, J,D )
 
    
    breakpoint()
    # Here I can compare the forecasts to the reality (might need to train longer)


        # np.shape(result_dict["rec_n_forecast"]) = (20,1,30,20)
        # this seems to be for one example we have (S, discard, T_forecast, JXD)

        # complete_ts = val_batch["past_target"][0:1]
        # matplotlib_fig = tensorboard_utils.show_time_series_forecast(
        #     fig_size=(12, 4),
        #     inputs=complete_ts.data.cpu().numpy()[0],
        #     rec_with_forecast=rec_with_forecast,
        #     context_length=config["context_length"],
        #     prediction_length=config["prediction_length"],
        #     fig_title="forecast",
        # )
        # fig_numpy_array = tensorboard_utils.plot_to_image(matplotlib_fig)
        # summary.add_image("Forecast", fig_numpy_array, step, dataformats="HWC")

        # for k, v in summary_items.items():
        #     summary.add_scalar(k, v, step)
        # summary.flush()

    # with open(os.path.join(config["log_dir"], "metrics.json"), "w") as fp:
    #     json.dump(all_metrics, fp)
