from RRAEsTorch.utilities import MLP_with_linear 
import jax.random as jrandom
import torch.nn as nn
from torch.func import vmap

def set_autoencoder_base(cls):
    global _AutoencoderBase
    _AutoencoderBase = cls

def get_autoencoder_base():
    global _AutoencoderBase
    return _AutoencoderBase or _default_autoencoder

def _default_autoencoder():
    class Autoencoder(nn.Module):
        _encode: MLP_with_linear
        _decode: MLP_with_linear
        _perform_in_latent: callable
        _perform_in_latent: callable
        map_latent: bool
        norm_funcs: list
        inv_norm_funcs: list
        count: int

        def __init__(
            self,
            in_size,
            latent_size,
            latent_size_after=None,
            _encode=None,
            _decode=None,
            map_latent=True,
            *,
            count=1,
            kwargs_enc={},
            kwargs_dec={},
            **kwargs,
        ):
            super().__init__()

            if latent_size_after is None:
                latent_size_after = latent_size

            if _encode is None:
                if "width_size" not in kwargs_enc.keys():
                    kwargs_enc["width_size"] = 64

                if "depth" not in kwargs_enc.keys():
                    kwargs_enc["depth"] = 1

                self._encode = MLP_with_linear(
                    in_size=in_size,
                    out_size=latent_size,
                    **kwargs_enc,
                )

            else:
                self._encode = _encode

            if not hasattr(self, "_perform_in_latent"):
                self._perform_in_latent = lambda x, *args, **kwargs: x 

            if _decode is None:
                if "width_size" not in kwargs_dec.keys():
                    kwargs_dec["width_size"] = 64
                if "depth" not in kwargs_dec.keys():
                    kwargs_dec["depth"] = 6

                self._decode = MLP_with_linear(
                    in_size=latent_size_after,
                    out_size=in_size,
                    **kwargs_dec,
                )
            else:
                self._decode = _decode

            self.count = count
            self.map_latent = map_latent
            self.inv_norm_funcs = ["decode"]
            self.norm_funcs = ["encode", "latent"]

        def encode(self, x, *args, **kwargs):
            return self._encode(x, *args, **kwargs)
        
        def decode(self, x, *args, **kwargs):
            return self._decode(x, *args, **kwargs)

        def perform_in_latent(self, y, *args, **kwargs):
            if self.map_latent:
                new_perform_in_latent = lambda x: self._perform_in_latent(
                    x, *args, **kwargs
                )
                for _ in range(self.count):
                    new_perform_in_latent = vmap(new_perform_in_latent, in_dims=-1, out_dims=-1) 
                return new_perform_in_latent(y)
            return self._perform_in_latent(y, *args, **kwargs)

        def forward(self, x, *args, **kwargs):
            return self.decode(self.perform_in_latent(self.encode(x), *args, **kwargs))

        def latent(self, x, *args, **kwargs):
            return self.perform_in_latent(self.encode(x), *args, **kwargs)

    return Autoencoder
