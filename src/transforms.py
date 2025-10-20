
import torch
import numpy as np


class ToTensor:
        """ Convert input (numpy arrays/scalars) to PyTorch tensors. """

        def __call__(self, sample):
                x, y = sample

                # Convert x to numpy array first if it's a list or scalar
                if not isinstance(x, np.ndarray):
                        x = np.array(x)
                x_t = torch.as_tensor(x, dtype=torch.float32)

                # Convert target y to a 1-D tensor (e.g. shape (1,) for scalar targets)
                if not isinstance(y, np.ndarray):
                        y = np.array(y)
                y = np.asarray(y)
                # Flatten any extra dimensions so target shape is (output_dim,) or scalar
                y_t = torch.as_tensor(y, dtype=torch.float32).reshape(-1)

                return x_t, y_t