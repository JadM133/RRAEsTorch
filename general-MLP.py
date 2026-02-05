""" This script contains an example of how to use a Trainor class for any equinox model. Specifically,
The MLP."""
import RRAEsTorch.config # Include this in all your scripts
from torchvision.ops import MLP
from RRAEsTorch.training_classes import Trainor_class
import numpy.random as random
import numpy as np
import torch

if __name__ == "__main__":
    # Step 1: Get the data - in this case dummy data is generated.
    inp = random.normal(size=(80, 10))
    out = np.expand_dims(np.sum(inp, axis=-1) ** 2 / 4 + 3, 1)

    inp = torch.tensor(inp, dtype=torch.float32)
    out = torch.tensor(out, dtype=torch.float32)

    # The shape should be what's expected in your model as inputs and outputs.
    print(f"Shape of data is {inp.shape} (N x D1) and {out.shape} (N x D2)")

    # Step 2: Specify the model to use, do not declare an instance of the class.
    # i.e. do not open/close parenthesis.
    model_cls = MLP

    loss_type = "default"  # Specify the loss type, this uses the norm in %.

    # Step 3: Define your trainor, with the model, data, and parameters.
    # Use Trainor_class. It has some slight differences compared to RRAE_Trainor_class.
    trainor = Trainor_class(
        inp,
        model_cls,
        in_channels=inp.shape[1],
        hidden_channels=[100, out.shape[1]],
        folder="folder_name/",
        file="saved_model.pkl",
    )

    # Step 4: Define the kw arguments for training. When using the Strong RRAE formulation,
    # you need to specify training kw arguments (first stage of training with SVD to
    # find the basis), and fine-tuning kw arguments (second stage of training with the
    # basis found in the first stage).
    kwargs = {
        "step_st": [2], # Increase those to train well
        "batch_size_st": [20, 20],
        "lr_st": [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
        "print_every": 100,
        "loss_type": loss_type,
    }

    # Step 5: Train the model and get the predictions.
    trainor.fit(
        inp,
        out,
        **kwargs,
    )
    preds = trainor.evaluate(inp, out)  # could give test as well as inp_test, out_test
    trainor.save_model()

    # Uncomment the following line if you want to hold the session to check your
    # results in the console.
    # pdb.set_trace()
