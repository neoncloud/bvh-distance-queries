# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# @author Vasileios Choutas
# Contact: vassilis.choutas@tuebingen.mpg.de
# Contact: ps-license@tuebingen.mpg.de

import sys

from typing import Tuple, NewType

import torch
import torch.nn as nn
import torch.autograd as autograd

import bvh_distance_queries_cuda

Tensor = NewType('Tensor', torch.Tensor)


class BVHFunction(autograd.Function):
    ''' Autograd wrapper for the BVH nearest neighbor kernel
    '''
    QUEUE_SIZE = 128
    SORT_POINTS_BY_MORTON = True

    @staticmethod
    def forward(ctx,
                triangles: Tensor,
                points: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        outputs = bvh_distance_queries_cuda.distance_queries(
            triangles, points,
            queue_size=BVHFunction.QUEUE_SIZE,
            sort_points_by_morton=BVHFunction.SORT_POINTS_BY_MORTON,
        )
        ctx.save_for_backward(triangles, points, *outputs[1:]) # dont save distance
        return outputs[0], outputs[1], outputs[2], outputs[3]

    @staticmethod
    def backward(ctx, grad_distances, grad_closest_points, grad_closest_faces, grad_closest_bcs):
        # Retrieve saved tensors
        triangles, points, closest_points, closest_faces, closest_bcs = ctx.saved_tensors

        # Compute gradient with respect to points
        grad_points = grad_distances.unsqueeze(-1) * 2 * (points - closest_points)
        if grad_closest_points is not None:
            grad_points += grad_closest_points

        # Compute total gradient for closest_points
        grad_closest_points_from_distances = grad_distances.unsqueeze(-1) * (-2) * (points - closest_points)
        total_grad_closest_points = grad_closest_points_from_distances
        if grad_closest_points is not None:
            total_grad_closest_points += grad_closest_points

        # Compute gradient with respect to the closest triangles
        grad_closest_triangles = total_grad_closest_points.unsqueeze(-2) * closest_bcs.unsqueeze(-1)  # (B, Q, 3, 3)

        # Identify the closest triangles
        batch_size, num_triangles, _, _ = triangles.shape
        _, num_queries = closest_faces.shape

        # Compute global indices for scattering
        batch_indices = torch.arange(batch_size, device=triangles.device).unsqueeze(1).expand(-1, num_queries).reshape(-1)
        face_indices = closest_faces.reshape(-1)
        global_face_indices = batch_indices * num_triangles + face_indices

        # Flatten gradients for scattering
        grad_closest_triangles_flat = grad_closest_triangles.reshape(-1, 3, 3)

        # Initialize gradient tensor for triangles
        grad_triangles_flat = torch.zeros((batch_size * num_triangles, 3, 3), device=triangles.device, dtype=triangles.dtype)

        # Scatter-add gradients to the corresponding triangles
        grad_triangles_flat.index_add_(0, global_face_indices, grad_closest_triangles_flat)

        # Reshape the gradient tensor back to the original shape
        grad_triangles = grad_triangles_flat.view(batch_size, num_triangles, 3, 3)

        # Return gradients with respect to inputs
        return grad_triangles, grad_points


class BVH(nn.Module):

    def __init__(self,
                 sort_points_by_morton: bool = True,
                 queue_size: int = 128) -> None:
        ''' Constructor for the BVH acceleration structure

            Parameters
            ----------
                sort_points_by_morton: bool, optional
                    Sort input points by their morton code. Helps improve query
                    speed. Default is true
                queue_size: int, optional
                    The size of the data structure used to store intermediate
                    distance computations
        '''
        super(BVH, self).__init__()
        assert queue_size in [32, 64, 128, 256, 512, 1024], (
            f'Queue/Stack size must be in {str[32, 64, 128, 256, 512, 1024]()}'
        )
        BVHFunction.QUEUE_SIZE = queue_size
        BVHFunction.SORT_POINTS_BY_MORTON = sort_points_by_morton

    @torch.no_grad()
    def forward(
            self,
            triangles: Tensor,
            points: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        ''' Forward pass of the search tree

            Parameters
            ----------
                triangles: torch.tensor
                    A BxFx3x3 PyTorch tensor that contains the triangle
                    locations.
                points: torch.tensor
                    A BxQx3 PyTorch tensor that contains the query point
                    locations.
            Returns
            -------
                distances: torch.tensor
                    A BxQ tensor with the *squared* distances to the closest
                    point
                closest_points: torch.tensor
                    A BxQx3 tensor with the closest points on the 3D mesh
                closest_faces: torch.tensor
                    A BxQ tensor that contains the index of the closest
                    triangle of each queyr point
                closest_bcs: torch.tensor
                    A BxQx3 tensor with the barycentric coordinates of the
                    closest point

        '''

        output = BVHFunction.apply(
            triangles, points)
        distances, closest_points, closest_faces, closest_bcs = output

        return distances, closest_points, closest_faces, closest_bcs
