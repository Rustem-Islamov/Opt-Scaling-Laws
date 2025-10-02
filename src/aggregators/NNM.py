import torch
from .base import _BaseAggregator

from src.aggregators.coordinatewise_median import CM

class NNM(_BaseAggregator):


    def __init__(self, f):
        self.f = f
        self.CM = CM()

    def __str__(self):
        return "NNM+CM"

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
        n = len(X)
        f = self.f
        assert f < n / 2, "f must be less than n/2"
        
        outputs = []
        neighbor_indices_list = []

        orig_shape = X[0].shape
        X = torch.stack(X).reshape(n, -1)
        for i in range(n):
            distances = torch.norm(X - X[i], p=2, dim=1)
            
            sorted_indices = torch.argsort(distances)
            nearest_neighbor_indices = sorted_indices[:n - f]
            neighbor_indices_list.append(nearest_neighbor_indices)
            
            neighbors = X[nearest_neighbor_indices]
            
            y_i = torch.mean(neighbors, dim=0).reshape(orig_shape)
            
            outputs.append(y_i)
         
        return self.CM(outputs)