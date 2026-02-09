
from __future__ import print_function
import numpy.random as random
import numpy as np
from RRAEsTorch.utilities import (
    remove_keys_from_dict,
    merge_dicts,
    loss_generator,
    eval_with_batches
)
import warnings
import os
import time
import dill
import shutil
from functools import partial
from RRAEsTorch.trackers import (
    Null_Tracker,
    RRAE_fixed_Tracker,
    RRAE_pars_Tracker,
    RRAE_gen_Tracker,
)
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import torch
from torch.utils.data import TensorDataset, DataLoader

from RRAEsTorch.utilities import get_basis

class Circular_list:
    """
    Creates a list of fixed size.
    Adds elements in a circular manner
    """

    def __init__(self, size):
        self.size = size
        self.buffer = [0.0] * size
        self.index = 0

    def add(self, value):
        self.buffer[self.index] = value
        self.index = (self.index + 1) % self.size

    def __iter__(self):
        for value in self.buffer:
            yield value

class Standard_Print():
    def __init__(self, aux, *args, **kwargs):
        self.aux = aux
    
    def __str__(self):
        message = ", ".join([f"{k}: {v}" for k, v in self.aux.items()])            
        return message
    
class Pretty_Print(PrettyTable):
    def __init__(self, aux, window_size=5, format_numbers=True, printer_settings={}):
        self.aux = aux
        self.format_numbers = format_numbers
        self.set_title = False
        self.first_print = True
        self.window_size = window_size
        self.index_new = 0
        self.index_old = 0
        super().__init__(**printer_settings)
        
    def format_number(self, n):
        if isinstance(n, int):
            return "{:.0f}".format(n)  
        else:
            return "{:.3f}".format(n)  
        
    def __str__(self):
        data = list(self.aux.values())
        if self.format_numbers == True:
            data = list(map(self.format_number, data))
            
        if self.first_print == True:
            titles = list(self.aux.keys())
            self.field_names = titles
            self.title = "Results"
            self.set_title = True
            for _ in range(self.window_size):
                self.add_row([" "]*len(titles))
                
            self._rows[self.index_new] = data
            print(super().__str__())
            print(f"\033[{self.window_size+1}A", end="")
            self.first_print = False
            
            
        self._rows[self.index_new] = data
        
        # This function does a lot of unnecessary things... Removed the parts that I don't want
        # print( "\n".join(self.get_string(start=self.index_new, 
        #                                  end=self.index_new+1,
        #                                  float_format="3.3").splitlines()[-2:]))
        
        options = self._get_options({})
   
        lines = []
   
        # Get the rows we need to print, taking into account slicing, sorting, etc.
        formatted_rows = [self._format_row(row) for row in self._rows[self.index_new : self.index_new+1]]
        
        # Compute column widths
        self._compute_widths(formatted_rows, options)
        self._hrule = self._stringify_hrule(options)
   
        # Add rows
        if formatted_rows:
            lines.append(
                self._stringify_row(
                    formatted_rows[-1],
                    options,
                    self._stringify_hrule(options, where="bottom_"),
                )
            )
   
        # Add bottom of border
        lines.append(self._stringify_hrule(options, where="bottom_"))
   
        print("\n".join(lines))
        

        # Update indices        
        self.index_old = self.index_new
        self.index_new = (self.index_new + 1) % self.window_size 
        
        # if we move to another printing cycle, push cursor back
        # Dirty trick but works on ubuntu...
        if (self.index_new - self.index_old) != 1:
            print(f"\033[{self.window_size*2}A", end="")
            # Factor of 2 is due to the lower line in the table
            
        return '\033[1A'

class Print_Info(PrettyTable):
    def __init__(self, print_type="std", aux={}, *args, **kwargs):
        check = (print_type.lower() == "std")
        if check == True:
            self.print_obj = Standard_Print(aux, *args, **kwargs)
        else:
            self.print_obj = Pretty_Print(aux, *args, **kwargs)
        
    def update_aux(self, aux):
        self.print_obj.aux = aux
        
    def __str__(self):
        return self.print_obj.__str__()


class Trainor_class:
    def __init__(
        self,
        in_train=None,
        model_cls=None,
        folder="",
        file=None,
        **kwargs,
    ):
        if model_cls is not None:
            orig_model_cls = model_cls
            self.model = model_cls(**kwargs)
        else:
            orig_model_cls = None

        self.all_kwargs = {
            "kwargs": kwargs,
            "orig_model_cls": orig_model_cls
        }

        self.folder = folder
        if folder != "":
            if not os.path.exists(folder):
                os.makedirs(folder)
        self.file = file

    def fit(
        self,
        input,
        output,
        loss_type="default",  # should be string to use pre defined functions
        loss=None,  # a function loss(pred, true) to differentiate in the model
        step_st=[3000, 3000],  # 000, 8000],
        lr_st=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
        print_every=np.nan,
        save_every=np.nan,
        batch_size_st=[16, 16, 16, 16, 32],
        regression=False,
        verbose=True,
        loss_kwargs={},
        flush=False,
        pre_func_inp=lambda x: x,
        pre_func_out=lambda x: x,
        fix_comp=lambda _: (),
        tracker=Null_Tracker(),
        stagn_window=20,
        eps_fn=lambda lat, bs: None,
        optimizer=torch.optim.Adam,
        verbatim = {
                    "print_type": "std",
                    "window_size" : 5,  
                    "printer_settings":{"padding_width": 3}
                    },
        save_losses=False,
        input_val=None,
        output_val=None,
        latent_size=0,
        device="cpu"
    ):
        assert isinstance(input, torch.Tensor), "Input should be a torch tensor"
        assert isinstance(output, torch.Tensor), "Output should be a torch tensor"
        
        from RRAEsTorch.utilities import v_print

        if flush:
            v_print = partial(v_print, f=True)
        else:
            v_print = partial(v_print, f=False)

        training_params = {
            "loss": loss,
            "step_st": step_st,
            "lr_st": lr_st,
            "print_every": print_every,
            "batch_size_st": batch_size_st,
            "regression": regression,
            "verbose": verbose,
            "loss_kwargs": loss_kwargs,
        }

        self.all_kwargs = merge_dicts(self.all_kwargs, training_params)  # Append dicts

        model = self.model # .double()  # Create alias for model

        fn = lambda x: x if fn is None else fn

        # Process loss function
        if callable(loss_type):
            loss_fun = loss_type
        else:
            loss_fun = loss_generator(loss_type, loss)

        # Make step funciton
        def make_step(model, input, out, optimizer, idx, epsilon, **loss_kwargs):

            optimizer.zero_grad(set_to_none=True)
            loss, aux = loss_fun(model, input, out, idx=idx, epsilon=epsilon, **loss_kwargs)

            loss.backward()
            optimizer.step()

            return loss, model, optimizer, aux

        # Create filter for splitting the model into differential and static portions

        for p in fix_comp(model): #e.g. model._encode.parameters()
            p.requires_grad = False


        # Loop variables
        t_all = 0.0  # Total time
        avg_loss = np.inf


        # Window to store averages
        store_window = min(stagn_window, sum(step_st))
        prev_losses = Circular_list(store_window)

        # Initialize tracker
        track_params = tracker.init()
        extra_track = {}
        
        # Initializer printer object
        print_info = Print_Info(**verbatim)

        if save_losses:
            all_losses = []

        
        model = model.to(device)

        if input_val is not None:
            dataset_val = TensorDataset(input_val, output_val, torch.arange(0, input_val.shape[0], 1))
            dataloader_val = DataLoader(dataset_val, batch_size=input_val.shape[0], shuffle=True, drop_last=True)

        # Outler Loop
        for steps, lr, batch_size in zip(step_st, lr_st, batch_size_st):
            try:
                t_t = 0.0  # Zero time
                optimizer_tr = optimizer(
                    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
                )

                if (batch_size > input.shape[0]) or batch_size == -1:
                    print(f"Setting batch size to: {input.shape[0]}")
                    batch_size = input.shape[0]

                # Inner loop (batch)

                dataset = TensorDataset(input, output, torch.arange(0, input.shape[0], 1))
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
                data_iter = iter(dataloader)

                for step in range(steps):
                    try:
                        input_b, out_b, idx_b = next(data_iter)
                    except StopIteration:
                        # reached the end, recreate iterator (reshuffle)
                        data_iter = iter(DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True))
                        input_b, out_b, idx_b = next(data_iter)
    
                    start_time = time.perf_counter()             # Start time
                    out_b = pre_func_out(out_b)   # Pre-process batch out values
                    input_b = pre_func_inp(input_b)              # Pre-process batch input values 
                    epsilon = eps_fn(latent_size, input_b.shape[0])

                    step_kwargs = merge_dicts(loss_kwargs, track_params)

                    # Compute loss
                    input_b = input_b.to(device)
                    out_b = out_b.to(device)
                    
                    loss, model, optimizer_tr, (aux, extra_track) = make_step(
                                                                model,
                                                                input_b,
                                                                out_b,
                                                                optimizer_tr,
                                                                idx_b,
                                                                epsilon,
                                                                **step_kwargs,
                                                            )
                    
                    if input_val is not None:
                        model.eval()
                        with torch.no_grad():
                            val_loss = []
                            for input_vb, out_vb, idx_b in dataloader_val:
                                out_vb = pre_func_out(out_vb) 
                                val_loss_batch = loss_fun(
                                    model, input_vb.to(device), out_vb.to(device), idx=idx_b, epsilon=None, **step_kwargs
                                )[0]
                                val_loss.append(val_loss_batch.item())
                            val_loss = sum(val_loss) / len(val_loss)
                            aux["val_loss"] = val_loss
                        model.train()
                    else:
                        aux["val_loss"] = None

                    if save_losses:
                        all_losses.append(aux)

                    prev_losses.add(loss.item())

                    if step > stagn_window:
                        avg_loss = sum(prev_losses) / stagn_window

                    track_params = tracker(loss, avg_loss, track_params, **extra_track)

                    if track_params.get("stop_train"):
                        break
                    
                    dt = time.perf_counter() - start_time  # Execution time
                    t_t += dt  # Batch execution time
                    t_all += dt  # Total execution time

                    if (step % print_every) == 0 or step == steps - 1:
                        t_t = 0.0               # Reset Batch execution time

                        print_info.update_aux({"Batch": step, **aux, "Time [s]": dt, "Total time [s]": t_all})
                        
                        print(print_info)
                    
                    if track_params.get("load"):
                        self.load_model(f"checkpoint_k_{track_params.get('k_max')}")
                        self.del_file(f"checkpoint_k_{track_params.get('k_max')}")
                        model = self.model

                        optimizer_tr = optimizer(
                            filter(lambda p: p.requires_grad, model.parameters()), lr=lr
                        )

                    if track_params.get("save") or ((step % save_every) == 0) or torch.isnan(loss):
                        if torch.isnan(loss):
                            raise ValueError("Loss is nan, stopping training...")

                        self.model = model

                        if track_params.get("save"):
                            orig = f"checkpoint_k_{track_params.get('k_max')+1}"
                            self.del_file(f"checkpoint_k_{track_params.get('k_max')+2}")
                            checkpoint_filename = orig

                        else:
                            orig = (
                                f"checkpoint_{step}"
                                if not torch.isnan(loss)
                                else "checkpoint_bf_nan"
                            )

                            checkpoint_filename = f"{orig}_0.pkl"

                            if os.path.exists(checkpoint_filename):
                                i = 1
                                new_filename = f"{orig}_{i}.pkl"
                                while self.path_exists(new_filename):
                                    i += 1
                                    new_filename = f"{orig}_{i}.pkl"
                                checkpoint_filename = new_filename
                        self.save_model(checkpoint_filename)

            except KeyboardInterrupt:
                pass
        
        if save_losses:
            orig = "all_losses"
            new_filename = f"{orig}_0.pkl"
            i = 0
            while self.path_exists(new_filename):
                i += 1
                new_filename = f"{orig}_{i}.pkl"

            self.save_object(all_losses, new_filename)

        model.eval()
        self.model = model
        self.batch_size = batch_size
        self.t_all = t_all
        return model, track_params | extra_track

    def plot_training_losses(self, idx=0):
        try:
            with open(os.path.join(self.folder, f"all_losses_{idx}.pkl"), "rb") as f:
                res_list = dill.load(f)
        except FileNotFoundError:
            raise ValueError("Losses where not saved during training, did you set save_losses=True in training_kwargs?")
        training_losses = [r["loss"] for r in res_list]
        val_losses = [r["val_loss"] for r in res_list]
        plt.plot(training_losses, label="training loss")
        if val_losses[0] is not None:
            plt.plot(val_losses, label="val loss")
        plt.legend()
        plt.xlabel("Forward pass")
        plt.show()

    @torch.no_grad()
    def evaluate(
        self,
        x_train_o=None,
        y_train_o=None,
        x_test_o=None,
        y_test_o=None,
        batch_size=None,
        pre_func_inp=lambda x: x,
        pre_func_out=lambda x: x,
        call_func=None,
        **kwargs,
    ):
        """Performs post-processing to find the relative error of the RRAE model.

        Parameters:
        -----------
        y_test: jnp.array
            The test data to be used for the error calculation.
        x_test: jnp.array
            The test input. If this is provided the error_test will be computed by sipmly giving
            x_test to the model.
        p_train: jnp.array
            The training data to be used for the interpolation. If this is provided along with p_test (next),
            the error_test will be computed by interpolating the latent space of the model and then decoding it.
        p_test: jnp.array
            The test parameters for which to interpolate.
        save: bool
            If anything other than False, the model as well as the results will be saved in f"{save}".pkl
        """
        call_func = (
            (lambda x: self.model(pre_func_inp(x))) if call_func is None else call_func
        )
        if x_train_o is not None:
            y_train_o = pre_func_out(y_train_o)
            assert (
                hasattr(self, "batch_size") or batch_size is not None
            ), "You should either provide a batch_size or fit the model first."

            dataset = TensorDataset(x_train_o)
            batch_size = self.batch_size if batch_size is None else batch_size
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            pred = []
            for x_b in dataloader:
                pred_batch = call_func(x_b[0])
                pred.append(pred_batch)

            y_pred_train_o = torch.concatenate(pred)

            self.error_train_o = (
                torch.linalg.norm(y_pred_train_o - y_train_o)
                / torch.linalg.norm(y_train_o)
                * 100
            )
            print("Train error on original output: ", self.error_train_o)

            y_pred_train = pre_func_out(y_pred_train_o)
            y_train = pre_func_out(y_train_o)
            self.error_train = (
                torch.linalg.norm(y_pred_train - y_train) / torch.linalg.norm(y_train) * 100
            )
            print("Train error on normalized output: ", self.error_train)

        if x_test_o is not None:
            y_test_o = pre_func_out(y_test_o)
            dataset = TensorDataset(x_test_o)
            batch_size = self.batch_size if batch_size is None else batch_size
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            pred = []
            for x_b in dataloader:
                pred_batch = call_func(x_b[0])
                pred.append(pred_batch)
            y_pred_test_o = torch.concatenate(pred)
            self.error_test_o = (
                torch.linalg.norm(y_pred_test_o - y_test_o)
                / torch.linalg.norm(y_test_o)
                * 100
            )
            
            print("Test error on original output: ", self.error_test_o)

            y_test = pre_func_out(y_test_o)
            y_pred_test = pre_func_out(y_pred_test_o)
            self.error_test = (
                torch.linalg.norm(y_pred_test - y_test) / torch.linalg.norm(y_test) * 100
            )
            print("Test error on normalized output: ", self.error_test)

        else:
            self.error_test = None
            self.error_test_o = None
            y_pred_test_o = None
            y_pred_test = None

        print("Total training time: ", self.t_all)
        return {
            "error_train": self.error_train,
            "error_test": self.error_test,
            "error_train_o": self.error_train_o,
            "error_test_o": self.error_test_o,
            "y_pred_train_o": y_pred_train_o,
            "y_pred_test_o": y_pred_test_o,
            "y_pred_train": y_pred_train,
            "y_pred_test": y_pred_test,
        }
    def path_exists(self, filename):
        return os.path.exists(os.path.join(self.folder, filename))

    def del_file(self, filename):
        filename = os.path.join(self.folder, filename)
        if os.path.exists(filename):
            os.remove(filename)

    def save_model(self, filename=None, erase=False, **kwargs):
        """Saves the trainor class."""
        if filename is None:
            if (self.folder is None) or (self.file is None):
                raise ValueError("You should provide a filename to save")
            filename = os.path.join(self.folder, self.file)
            if erase:
                shutil.rmtree(self.folder)
                os.makedirs(self.folder)
        else:
            filename = os.path.join(self.folder, filename)
            if not os.path.exists(filename):
                with open(filename, "a") as temp_file:
                    pass
                os.utime(filename, None)
        attr = merge_dicts(
            remove_keys_from_dict(self.__dict__, ("model", "all_kwargs")),
            kwargs,
        )

        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "all_kwargs": self.all_kwargs,
            "attr": attr
        }
        
        with open(filename, "wb") as f:
            dill.dump(save_dict, f)

        print(f"Model saved in {filename}")
    
    def load_object(self, filename):
        filename = os.path.join(self.folder, filename)
        with open(filename, "rb") as f:
            object = dill.load(f)
        return object
    
    def save_object(self, obj, filename):
        filename = os.path.join(self.folder, filename)
        with open(filename, "wb") as f:
            dill.dump(obj, f)
        print(f"Object saved in {filename}")

    def load_model(self, filename=None, erase=False, path=None, orig_model_cls=None, device="cpu",**fn_kwargs):
        """NOTE: fn_kwargs defines the functions of the model
        (e.g. final_activation, inner activation), if
        needed to be saved/loaded on different devices/OS.
        """

        if path == None:
            filename = self.file if filename is None else filename
            filename = os.path.join(self.folder, filename)
        else:
            filename = path

        with open(filename, "rb") as f:
            save_dict = dill.load(f)
            self.all_kwargs = save_dict["all_kwargs"]
            if orig_model_cls is None:
                orig_model_cls = self.all_kwargs["orig_model_cls"]
            else:
                orig_model_cls = orig_model_cls
            kwargs = self.all_kwargs["kwargs"]

            kwargs.update(fn_kwargs)
            
            model = orig_model_cls(**kwargs)
            model.load_state_dict(save_dict["model_state_dict"])
            self.model = model.to(device)
            attributes = save_dict["attr"]

            for key in attributes:
                setattr(self, key, attributes[key])
        if erase:
            os.remove(filename)


class AE_Trainor_class(Trainor_class):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, *args, training_kwargs,  **kwargs):
        if "pre_func_inp" not in kwargs:
            self.pre_func_inp = lambda x: x
        else:
            self.pre_func_inp = kwargs["pre_func_inp"]
        training_kwargs = merge_dicts(kwargs, training_kwargs)
        return super().fit(*args, **training_kwargs)  # train model

    # def AE_interpolate(
    #     self,
    #     p_train,
    #     p_test,
    #     x_train_o,
    #     y_test_o,
    #     batch_size=None,
    #     latent_func=None,
    #     decode_func=None,
    #     norm_out_func=None,
    # ):
    #     """Interpolates the latent space of the model and then decodes it to find the output."""
    #     batch_size = self.batch_size if batch_size is None else batch_size

    #     if latent_func is None:
    #         call_func = lambda x: self.model.latent(x)
    #     else:
    #         call_func = latent_func

    #     latent_train = eval_with_batches(
    #         x_train_o,
    #         batch_size,
    #         call_func=call_func,
    #         str="Finding train latent space used for interpolation...",
    #         key_idx=0,
    #     )

    #     interpolation = Objects_Interpolator_nD()
    #     latent_test_interp = interpolation(p_test, p_train, latent_train)

    #     if decode_func is None:
    #         call_func = lambda x: self.model.decode(x)
    #     else:
    #         call_func = decode_func

    #     y_pred_interp_test_o = eval_with_batches(
    #         latent_test_interp,
    #         batch_size,
    #         call_func=call_func,
    #         str="Decoding interpolated latent space ...",
    #         key_idx=0,
    #     )

    #     self.error_interp_test_o = (
    #         jnp.linalg.norm(y_pred_interp_test_o - y_test_o)
    #         / jnp.linalg.norm(y_test_o)
    #         * 100
    #     )
    #     print(
    #         "Test (interpolation) error over original output: ",
    #         self.error_interp_test_o,
    #     )

    #     if norm_out_func is None:
    #         call_func = lambda x: self.model.norm_out(x)
    #     else:
    #         call_func = norm_out_func

    #     y_pred_interp_test = eval_with_batches(
    #         y_pred_interp_test_o,
    #         batch_size,
    #         call_func=call_func,
    #         str="Finding Normalized pred of interpolated latent space ...",
    #         key_idx=0,
    #     )

    #     y_test = eval_with_batches(
    #         y_test_o,
    #         batch_size,
    #         call_func=call_func,
    #         str="Finding Normalized output of interpolated latent space ...",
    #         key_idx=0,
    #     )
    #     self.error_interp_test = (
    #         jnp.linalg.norm(y_pred_interp_test - y_test) / jnp.linalg.norm(y_test) * 100
    #     )
    #     print(
    #         "Test (interpolation) error over normalized output: ",
    #         self.error_interp_test,
    #     )
    #     return {
    #         "error_interp_test": self.error_interp_test,
    #         "error_interp_test_o": self.error_interp_test_o,
    #         "y_pred_interp_test_o": y_pred_interp_test_o,
    #         "y_pred_interp_test": y_pred_interp_test,
    #     }


class RRAE_Trainor_class(AE_Trainor_class):
    def __init__(self, *args, adapt=False, k_max=None, adap_type="None", **kwargs):
        self.k_init = k_max
        self.adap_type = adap_type
        if k_max is not None:
            kwargs["k_max"] = k_max

        super().__init__(*args, **kwargs)
        self.adapt = adapt

    def fit(self, *args, **kwargs):
        if self.adap_type == "pars":
            default_tracker = RRAE_pars_Tracker(k_init=self.k_init)
        elif self.adap_type == "gen":
            if self.k_init is None:
                warnings.warn(
                    "k_max can not be None when using gen adaptive scheme, choose a big initial k_max to start with."
                )
            default_tracker = RRAE_gen_Tracker(k_init=self.k_init)
        elif self.adap_type == "None":
            if self.k_init is None:
                warnings.warn(
                    "k_max can not be None when using fixed scheme, choose a fixed k_max to use."
                )
            default_tracker = RRAE_fixed_Tracker(k_init=self.k_init)

        print("Training RRAEs...")

        if "training_kwargs" in kwargs:
            training_kwargs = kwargs["training_kwargs"]
            kwargs.pop("training_kwargs")
        else:
            training_kwargs = {}

        if "ft_kwargs" in kwargs:
            ft_kwargs = kwargs["ft_kwargs"]
            kwargs.pop("ft_kwargs")
        else:
            ft_kwargs = {}

        if "pre_func_inp" not in kwargs:
            self.pre_func_inp = lambda x: x
        else:
            self.pre_func_inp = kwargs["pre_func_inp"]

        if "tracker" not in training_kwargs:
            training_kwargs["tracker"] = default_tracker


        training_kwargs = merge_dicts(kwargs, training_kwargs)

        model, track_params = super().fit(*args, training_kwargs=training_kwargs)  # train model
    
        self.track_params = track_params    # Save track parameters in class?

        if "batch_size_st" in training_kwargs:
            self.batch_size = training_kwargs["batch_size_st"][-1]
        else:
            self.batch_size = 16  # default value

        if ft_kwargs:
            if "get_basis_bool" in ft_kwargs:
                get_basis_bool = ft_kwargs["get_basis_bool"]
                ft_kwargs.pop("get_basis_bool")
            else:
                get_basis_bool = True
            
            if "ft_end_type" in ft_kwargs:
                ft_end_type = ft_kwargs["ft_end_type"]
                ft_kwargs.pop("ft_end_type")
            else:
                ft_end_type = "concat"
            
            if "basis_call_kwargs" in ft_kwargs:
                basis_call_kwargs = ft_kwargs["basis_call_kwargs"]
                ft_kwargs.pop("basis_call_kwargs")
            else:
                ft_end_type = "concat"
                basis_call_kwargs = {}

            device = ft_kwargs.get("device", "cpu")
            
            if "AE_func" in ft_kwargs:
                AE_func = ft_kwargs["AE_func"]
                ft_kwargs.pop("AE_func")
            else:
                AE_func = lambda m:m
                
            ft_model, ft_track_params = self.fine_tune_basis(
                None, args=args, kwargs=ft_kwargs, get_basis_bool=get_basis_bool, end_type=ft_end_type, basis_call_kwargs=basis_call_kwargs, device=device, AE_func=AE_func
            )  # fine tune basis
            self.ft_track_params = ft_track_params
        else:
            ft_model = None
            ft_track_params = {}
        return model, track_params, ft_model, ft_track_params

    def fine_tune_basis(self, basis=None, get_basis_bool=True, end_type="concat", basis_call_kwargs={}, device="cpu", AE_func=lambda m:m, *, args, kwargs):

        if "loss" in kwargs:
            norm_loss_ = kwargs["loss"]
        else:
            print("Defaulting to L2 norm")
            norm_loss_ = lambda x1, x2: 100 * (
                torch.linalg.norm(x1 - x2) / torch.linalg.norm(x2)
            )

        if (basis is None):
            inp = args[0] if len(args) > 0 else kwargs["input"]

            if "basis_batch_size" in kwargs:
                basis_batch_size = kwargs["basis_batch_size"]
                kwargs.pop("basis_batch_size")
            else:
                basis_batch_size = self.batch_size

            basis_call_kwargs = basis_call_kwargs | self.track_params

            model = self.model.to(device)
            k_max = self.track_params["k_max"]
            if isinstance(AE_func, list):
                bases = []
                for func in AE_func:
                    bases.append(get_basis(get_basis_bool, model, k_max, basis_batch_size, inp, end_type, device, basis_call_kwargs, self.pre_func_inp, func))
                self.basis = bases
            else:
                self.basis = get_basis(get_basis_bool, model, k_max, basis_batch_size, inp, end_type, device, basis_call_kwargs, self.pre_func_inp, AE_func)
            
        else:
            self.basis = basis

        def loss_fun(model, input, out, idx, epsilon, basis):
            pred = model(input, epsilon=epsilon, apply_basis=basis)
            aux = {"loss": norm_loss_(pred, out)}
            return norm_loss_(pred, out), (aux, {})

        if "loss_type" in kwargs :
            pass
        else:
            print("Defaulting to standard loss")
            kwargs["loss_type"] = loss_fun

        kwargs.setdefault("loss_kwargs", {}).update({"basis": self.basis})
        
        fix_comp = lambda model: AE_func(model)._encode.parameters()
        print("Fine tuning the basis ...")
        return super().fit(*args, fix_comp=fix_comp, training_kwargs=kwargs)

    def evaluate(
        self,
        x_train_o=None,
        y_train_o=None,
        x_test_o=None,
        y_test_o=None,
        batch_size=None,
        pre_func_inp=lambda x: x,
        pre_func_out=lambda x: x,
        device="cpu",
    ):

        call_func = lambda x: self.model(pre_func_inp(x.to(device)), apply_basis=self.basis.to(device), epsilon=None).to("cpu")
        res = super().evaluate(
            x_train_o,
            y_train_o,
            x_test_o,
            y_test_o,
            batch_size,
            call_func=call_func,
            pre_func_inp=pre_func_inp,
            pre_func_out=pre_func_out,
            device=device,
        )
        return res

    def AE_interpolate(
        self,
        p_train,
        p_test,
        x_train_o,
        y_test_o,
        batch_size=None,
        latent_func=None,
        decode_func=None,
        norm_out_func=None,
    ):
        call_func = lambda x: (
            self.model.latent(x, apply_basis=self.basis)
            if latent_func is None
            else latent_func
        )
        return super().AE_interpolate(
            p_train,
            p_test,
            x_train_o,
            y_test_o,
            batch_size,
            latent_func=call_func,
            decode_func=decode_func,
            norm_out_func=norm_out_func,
        )
