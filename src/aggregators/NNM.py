import torch
from .base import _BaseAggregator


class NNM(_BaseAggregator):


    def __init__(self, f):
        self.f = f

    def __call__(self, X):
        """
        Implements the forward pass of the NNM algorithm.

        Args:
            ctx: Context object to save information for the backward pass.
            X (torch.Tensor): The input tensor of shape (n, d), where n is the
                              number of vectors and d is their dimension.
            f (int): The number of Byzantine inputs to tolerate.

        Returns:
            torch.Tensor: The output tensor Y of shape (n, d).
        """
        n, d = X.shape
        f = self.f
        assert f < n / 2, "f must be less than n/2"
        
        outputs = []
        neighbor_indices_list = []

        for i in range(n):
            distances = torch.norm(X - X[i], p=2, dim=1)
            
            sorted_indices = torch.argsort(distances)
            nearest_neighbor_indices = sorted_indices[:n - f]
            neighbor_indices_list.append(nearest_neighbor_indices)
            
            print(nearest_neighbor_indices)

            # Select the neighbor vectors using the indices
            neighbors = X[nearest_neighbor_indices]
            
            # Average the neighbors and append to the output list
            y_i = torch.mean(neighbors, dim=0)
            print(y_i)
            outputs.append(y_i)

        # Stack the results into a single tensor
        Y = torch.stack(outputs)
        
        # Save necessary information for the backward pass
        # neighbor_indices_tensor = torch.stack(neighbor_indices_list)
        # ctx.save_for_backward(neighbor_indices_tensor)
        # ctx.n = n
        # ctx.f = f
        # ctx.d = d
        
        print(Y.shape)

        return Y.mean(dim=0)