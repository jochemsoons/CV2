# Copyright 2017 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Differentiable 3-D rendering of a triangle mesh."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from mesh_renderer import camera_utils
from texturing import rasterize_triangles


def bilinear_sampler(img, x, y):
    # prepare useful params
    B = tf.shape(img)[0]
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    C = tf.shape(img)[3]

    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')

    # grab 4 nearest corner points for each (x_i, y_i)
    # i.e. we need a rectangle around the point of interest
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1
    zero = tf.zeros_like(x0)

    # clip to range [0, H/W] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = _get_pixel_value(img, x0, y0)
    Ib = _get_pixel_value(img, x0, y1)
    Ic = _get_pixel_value(img, x1, y0)
    Id = _get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=2)
    wb = tf.expand_dims(wb, axis=2)
    wc = tf.expand_dims(wc, axis=2)
    wd = tf.expand_dims(wd, axis=2)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return out


def _get_pixel_value(img, x, y):
    shape = tf.shape(x)
    batch_size = shape[0]
    vertex_size = shape[1]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, [batch_size, 1])
    b = tf.tile(batch_idx, [1, vertex_size])

    indices = tf.stack([b, y, x], 2)

    return tf.gather_nd(img, indices)


def mesh_renderer_texture(vertices,
                          triangles,
                          normals,
                          camera_position,
                          camera_lookat,
                          camera_up,
                          image_width,
                          image_height,
                          texture_map, # batch * n_vertex * 2
                          texture_image, # batch * Ht * Wt * 3
                          texture_width,
                          texture_height,
                          fov_y=40.0,
                          near_clip=0.01,
                          far_clip=10.0):
  """Renders an input scene using phong shading, and returns an output image.

  Args:
    vertices: 3-D float32 tensor with shape [batch_size, vertex_count, 3]. Each
        triplet is an xyz position in world space.
    triangles: 2-D int32 tensor with shape [triangle_count, 3]. Each triplet
        should contain vertex indices describing a triangle such that the
        triangle's normal points toward the viewer if the forward order of the
        triplet defines a clockwise winding of the vertices. Gradients with
        respect to this tensor are not available.
    normals: 3-D float32 tensor with shape [batch_size, vertex_count, 3]. Each
        triplet is the xyz vertex normal for its corresponding vertex. Each
        vector is assumed to be already normalized.
    diffuse_colors: 3-D float32 tensor with shape [batch_size,
        vertex_count, 3]. The RGB diffuse reflection in the range [0,1] for
        each vertex.
    camera_position: 2-D tensor with shape [batch_size, 3] or 1-D tensor with
        shape [3] specifying the XYZ world space camera position.
    camera_lookat: 2-D tensor with shape [batch_size, 3] or 1-D tensor with
        shape [3] containing an XYZ point along the center of the camera's gaze.
    camera_up: 2-D tensor with shape [batch_size, 3] or 1-D tensor with shape
        [3] containing the up direction for the camera. The camera will have no
        tilt with respect to this direction.
    light_positions: a 3-D tensor with shape [batch_size, light_count, 3]. The
        XYZ position of each light in the scene. In the same coordinate space as
        pixel_positions.
    light_intensities: a 3-D tensor with shape [batch_size, light_count, 3]. The
        RGB intensity values for each light. Intensities may be above one.
    image_width: int specifying desired output image width in pixels.
    image_height: int specifying desired output image height in pixels.
    specular_colors: 3-D float32 tensor with shape [batch_size,
        vertex_count, 3]. The RGB specular reflection in the range [0, 1] for
        each vertex.  If supplied, specular reflections will be computed, and
        both specular_colors and shininess_coefficients are expected.
    shininess_coefficients: a 0D-2D float32 tensor with maximum shape
       [batch_size, vertex_count]. The phong shininess coefficient of each
       vertex. A 0D tensor or float gives a constant shininess coefficient
       across all batches and images. A 1D tensor must have shape [batch_size],
       and a single shininess coefficient per image is used.
    ambient_color: a 2D tensor with shape [batch_size, 3]. The RGB ambient
        color, which is added to each pixel in the scene. If None, it is
        assumed to be black.
    fov_y: float, 0D tensor, or 1D tensor with shape [batch_size] specifying
        desired output image y field of view in degrees.
    near_clip: float, 0D tensor, or 1D tensor with shape [batch_size] specifying
        near clipping plane distance.
    far_clip: float, 0D tensor, or 1D tensor with shape [batch_size] specifying
        far clipping plane distance.

  Returns:
    A 4-D float32 tensor of shape [batch_size, image_height, image_width, 4]
    containing the lit RGBA color values for each image at each pixel. RGB
    colors are the intensity values before tonemapping and can be in the range
    [0, infinity]. Clipping to the range [0,1] with tf.clip_by_value is likely
    reasonable for both viewing and training most scenes. More complex scenes
    with multiple lights should tone map color values for display only. One
    simple tonemapping approach is to rescale color values as x/(1+x); gamma
    compression is another common techinque. Alpha values are zero for
    background pixels and near one for mesh pixels.
  Raises:
    ValueError: An invalid argument to the method is detected.
  """
  if len(vertices.shape) != 3:
    raise ValueError('Vertices must have shape [batch_size, vertex_count, 3].')
  batch_size = vertices.shape[0].value
  if len(normals.shape) != 3:
    raise ValueError('Normals must have shape [batch_size, vertex_count, 3].')
  if camera_position.get_shape().as_list() == [3]:
    camera_position = tf.tile(
        tf.expand_dims(camera_position, axis=0), [batch_size, 1])
  elif camera_position.get_shape().as_list() != [batch_size, 3]:
    raise ValueError('Camera_position must have shape [batch_size, 3]')
  if camera_lookat.get_shape().as_list() == [3]:
    camera_lookat = tf.tile(
        tf.expand_dims(camera_lookat, axis=0), [batch_size, 1])
  elif camera_lookat.get_shape().as_list() != [batch_size, 3]:
    raise ValueError('Camera_lookat must have shape [batch_size, 3]')
  if camera_up.get_shape().as_list() == [3]:
    camera_up = tf.tile(tf.expand_dims(camera_up, axis=0), [batch_size, 1])
  elif camera_up.get_shape().as_list() != [batch_size, 3]:
    raise ValueError('Camera_up must have shape [batch_size, 3]')
  if isinstance(fov_y, float):
    fov_y = tf.constant(batch_size * [fov_y], dtype=tf.float32)
  elif not fov_y.get_shape().as_list():
    fov_y = tf.tile(tf.expand_dims(fov_y, 0), [batch_size])
  elif fov_y.get_shape().as_list() != [batch_size]:
    raise ValueError('Fov_y must be a float, a 0D tensor, or a 1D tensor with'
                     'shape [batch_size]')
  if isinstance(near_clip, float):
    near_clip = tf.constant(batch_size * [near_clip], dtype=tf.float32)
  elif not near_clip.get_shape().as_list():
    near_clip = tf.tile(tf.expand_dims(near_clip, 0), [batch_size])
  elif near_clip.get_shape().as_list() != [batch_size]:
    raise ValueError('Near_clip must be a float, a 0D tensor, or a 1D tensor'
                     'with shape [batch_size]')
  if isinstance(far_clip, float):
    far_clip = tf.constant(batch_size * [far_clip], dtype=tf.float32)
  elif not far_clip.get_shape().as_list():
    far_clip = tf.tile(tf.expand_dims(far_clip, 0), [batch_size])
  elif far_clip.get_shape().as_list() != [batch_size]:
    raise ValueError('Far_clip must be a float, a 0D tensor, or a 1D tensor'
                     'with shape [batch_size]')

  vertex_attributes = tf.concat([
    normals, vertices,
    tf.constant([[[1, -1]]], dtype=tf.float32) * (texture_map - tf.constant([[[0, 1]]], dtype=tf.float32))], axis=2)

  camera_matrices = camera_utils.look_at(camera_position, camera_lookat,
                                         camera_up)

  perspective_transforms = camera_utils.perspective(image_width / image_height,
                                                    fov_y, near_clip, far_clip)

  clip_space_transforms = tf.matmul(perspective_transforms, camera_matrices)

  pixel_attributes, z_buffer = rasterize_triangles.rasterize(
      vertices, vertex_attributes, triangles, clip_space_transforms,
      image_width, image_height, [-1] * vertex_attributes.shape[2].value)

  # Extract the interpolated vertex attributes from the pixel buffer and
  # supply them to the shader:
  pixel_normals = tf.nn.l2_normalize(pixel_attributes[:, :, :, 0:3], dim=3)
  pixel_positions = pixel_attributes[:, :, :, 3:6]
  pixel_texture_uv = pixel_attributes[:, :, :, 6:6+2] # UV is from 0 to 1

  pixel_mask = tf.cast(tf.reduce_any(pixel_texture_uv >= 0, axis=3), tf.float32)

  x = tf.reshape(pixel_texture_uv[..., 0], [batch_size, -1]) * texture_width
  y = tf.reshape(pixel_texture_uv[..., 1], [batch_size, -1]) * texture_height

  textured_output = bilinear_sampler(texture_image, x, y)
  textured_output = tf.reshape(
      textured_output, [batch_size, image_height, image_width, texture_image.shape[3].value])

  pixel_mask = tf.reshape(pixel_mask, [batch_size, image_height, image_width, 1])

  valid_rgb_values = tf.concat(3 * [pixel_mask > 0.5], axis=3)

  rgb_images = tf.where(valid_rgb_values, textured_output,
                        tf.zeros_like(textured_output, dtype=tf.float32))

  return tf.reverse(tf.concat([rgb_images, pixel_mask], axis=3), axis=[1]), z_buffer, pixel_normals


