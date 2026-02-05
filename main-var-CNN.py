""" Example script to train a VRRAE with CNN encoder/decoder on images. """
import RRAEsTorch.config # Include this in all your scripts
from RRAEsTorch.AE_classes import *
from RRAEsTorch.training_classes import RRAE_Trainor_class, Trainor_class  # , Trainor_class
from RRAEsTorch.utilities import get_data
import numpy as np
import torch


if __name__ == "__main__":
    # Step 1: Get the data - replace this with your own data of the same shape.
    all_errors = []
    all_stds = []
    data_size = 100
    # Below is the synthetic dataset, choose "mnist", "CelebA", or "CIFAR-10" for 
    # the real world data, you will be asked to download a data folder the first
    # time you use those.
    problem = "2d_gaussian_shift_scale" 
    (
        x_train,
        x_test,
        p_train,
        p_test,
        y_train,
        y_test,
        pre_func_inp,
        pre_func_out,
        args,
    ) = get_data(problem, train_size=data_size, folder="../")


    print(
        f"Shape of data is {x_train.shape} (Ntr x C x H x W) and {x_test.shape} (Nte, C x H x W)"
    )


    # Step 2: Specify the model to use, VAR_Strong_RRAE_CNN is ours (recommended).
    method = "VRRAE"
    model_cls = VRRAE_CNN # or VAR_AE_CNN for VAE

    loss_type = (
        "VRRAE"  # or "VAE" for VAE
    )

    match method:
        case "VRRAE":
            eps_fn = lambda lat, bs: torch.tensor(np.random.normal(0, 1, size=(1, 1, bs, k_max)), dtype=torch.float32, device=device)
        case "VAE":
            eps_fn = lambda lat, bs: torch.tensor(np.random.normal(size=(1, 1, bs, lat)), dtype=torch.float32, device=device)

    # Step 3: Specify the archietectures' parameters:
    latent_size = 200  # latent space dimension
    k_max = (
        2  # number of features in the latent space (after the truncated SVD).
    )


    # Step 4: Define your trainor, with the model, data, and parameters.
    trainor = RRAE_Trainor_class( # or Trainor_class for VAEs
        x_train,
        model_cls,
        latent_size=latent_size,
        channels=x_train.shape[1],
        height=x_train.shape[2],
        width=x_train.shape[3],
        k_max=k_max,
        folder=f"{problem}/{method}_{problem}/",
        file=f"{method}_{problem}.pkl",
        norm_in="None",
        norm_out="None",
        out_train=x_train,
        kwargs_enc={
            "width_CNNs": [32, 64],
            "kernel_conv": 3,
            "stride": 2,
            "padding": 1,
        },
        kwargs_dec={
            "width_CNNs": [256, 128, 32, 8],
            "kernel_conv": 3,
            "stride": 2,
            "padding": 1,
        },
    )

    # Step 5: Define the kw arguments for training. When using the Strong RRAE formulation,
    # you need to specify training kw arguments (first stage of training with SVD to
    # find the basis), and fine-tuning kw arguments (second stage of training with the
    # basis found in the first stage).
    device = "cuda" if torch.cuda.is_available() else "cpu"

    training_kwargs = {
        "step_st": [2],  # 7680*data_size/64
        "batch_size_st": [64],
        "lr_st": [1e-3, 1e-5, 1e-8],
        "print_every": 1,
        "loss_type": loss_type,
        "loss_kwargs": {"beta": 0.001},
        "eps_fn": eps_fn,
        "device": device,
    }


    ft_kwargs = {
        "step_st": [0], # Increase if you want to fine tune
        "batch_size_st": [64],
        "lr_st": [1e-4, 1e-6, 1e-7, 1e-8],
        "print_every": 1,
        "eps_fn": eps_fn,
        "device": device,
    }


    # Step 6: Train the model and get the predictions.
    trainor.fit(
        x_train,
        y_train,
        pre_func_inp=pre_func_inp,
        pre_func_out=pre_func_out,
        latent_size=latent_size,

        # Remove the next two lines and add **training_kwargs for VAEs.
        training_kwargs=training_kwargs,
        ft_kwargs=ft_kwargs,
    )

    trainor.save_model()

    preds = trainor.evaluate(
        x_train, y_train, x_test, y_test, None, pre_func_inp, pre_func_out, device
    )
            
    # pdb.set_trace()
