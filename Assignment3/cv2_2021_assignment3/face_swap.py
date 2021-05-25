import tensorflow as tf 
import numpy as np
from PIL import Image
import os
import glob
import argparse
import cv2
import platform
import tqdm
import sys
import argparse
from scipy.io import loadmat,savemat

from preprocess_img import Preprocess
from load_data import *
from face_decoder import Face3D
import fio
import texture_manip

from collections import defaultdict
import time


def load_graph(graph_filename):
    with tf.gfile.GFile(graph_filename,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    return graph_def

class FaceSwap:
    def __init__(self):
        # read BFM face model
        # transfer original BFM model to our model
        if not os.path.isfile('./BFM/BFM_model_front.mat'):
            transferBFM09()

    def swap(self, fg, bg):
        """ Swaps background and foreground images.

        Inputs:
            - fg, bg: RGB images

        Outputs:
            - swapped: swapped RGB image
        """
        input_imgs = [fg, bg]

        U, V, tri, tri_with_mouth = texture_manip.get_uv()
        UV = np.concatenate([U.reshape((-1, 1)), V.reshape((-1, 1))], axis=1)

        batchsize = 1
        n = 0
        img_size = 224

        class TextureGenerator:
            def __init__(self):
                self.fr0 = Face3D(tri=None)
                self.n_vertex = self.fr0.facemodel.meanshape.shape[0].value // 3

                self.texture_image = tf.placeholder(
                    name='texture_image', shape=[batchsize, img_size,img_size, 3], dtype=tf.float32)
                self.new_coeff = tf.placeholder(
                    name='new_coeff', shape=[batchsize, 257], dtype=tf.float32)

                self.fr0.Reconstruction_Block(self.new_coeff, None, self.texture_image, batchsize)

            def generate_texture(self, coeff, input_img, sess):
                face_shape = self.fr0.face_shape_t
                face_norm = self.fr0.face_norm
                z_buffer = self.fr0.z_buffer
                norm_imgs = self.fr0.norm_imgs

                face_shape_, face_norm_, uv, z_buffer_, norm_imgs_ = sess.run([
                    face_shape, face_norm, self.fr0.uv, z_buffer, norm_imgs],feed_dict = {
                        self.new_coeff: coeff,
                        self.texture_image: input_img[..., ::-1]})

                input_img = np.squeeze(input_img)
                z_buffer_ = np.squeeze(z_buffer_)
                face_shape_ = np.squeeze(face_shape_, (0))
                face_norm_ = np.squeeze(face_norm_, (0))
                norm_imgs = np.squeeze(norm_imgs_, (0))

                texture_target = texture_manip.create_final_texture(
                    uv[0], input_img, UV, tri_with_mouth, face_norm_, z_buffer_, len(tri))

                return texture_target

            def generate_texture_cached(self, coeff, input_img, sess, cache):
                face_shape = self.fr0.face_shape_t
                face_norm = self.fr0.face_norm
                z_buffer = self.fr0.z_buffer

                face_shape_, face_norm_, uv, z_buffer_ = sess.run([
                    face_shape, face_norm, self.fr0.uv, z_buffer],feed_dict = {
                        self.new_coeff: coeff,
                        self.texture_image: input_img[..., ::-1]})

                input_img = np.squeeze(input_img)
                z_buffer_ = np.squeeze(z_buffer_)
                face_shape_ = np.squeeze(face_shape_, (0))
                face_norm_ = np.squeeze(face_norm_, (0))

                if cache is None:
                    t1 = time.time()
                    cache = texture_manip.bar_coord_cache(
                        uv[0], input_img, UV, tri_with_mouth, face_norm_, z_buffer_, len(tri))
                    print("Cache %.3f secs" % (time.time() - t1))

                t1 = time.time()
                texture_target = texture_manip.create_final_texture_from_cache(
                    cache, uv[0], input_img, UV, tri_with_mouth, face_norm_, z_buffer_, len(tri))
                print("Texture from cache %.3f secs" % (time.time() - t1))

                return cache, texture_target

        class TextureRenderer:
            def __init__(self, tri=None):
                self.fr0 = Face3D(tri=tri)
                self.n_vertex = self.fr0.facemodel.meanshape.shape[0].value // 3

                texture_map = tf.constant(UV[np.newaxis], dtype=tf.float32)
                self.texture_image = tf.placeholder(
                    name='texture_image', shape=[batchsize, img_size,img_size, 3], dtype=tf.float32)
                self.new_coeff = tf.placeholder(
                    name='new_coeff', shape=[batchsize, 257], dtype=tf.float32)

                self.fr0.Reconstruction_Block(self.new_coeff, texture_map, self.texture_image, batchsize)

            def render_texture(self, coeff, texture_target, sess):
                face_shape = self.fr0.face_shape_t
                face_norm = self.fr0.face_norm
                z_buffer = self.fr0.z_buffer
                render_img = self.fr0.render_imgs

                norm_img = self.fr0.norm_imgs

                r_img, = sess.run([render_img],feed_dict = {
                        self.new_coeff: coeff,
                        self.texture_image: texture_target[None]})

                return r_img[0]

        with tf.Graph().as_default() as graph,tf.device('/cpu:0'):
            # To render without mouth, for texture generation
            texture_generator = TextureGenerator()
            texture_renderer = TextureRenderer()

            self.images = tf.placeholder(name='input_imgs', shape=[batchsize,224,224,3], dtype=tf.float32)
            graph_def = load_graph('network/FaceReconModel.pb')
            tf.import_graph_def(graph_def,name='resnet',input_map={'input_imgs:0': self.images})
            self.coeff = graph.get_tensor_by_name('resnet/coeff:0')

            with tf.Session() as sess:
                print('reconstructing...')
                
                coeffs = []

                for input_img in input_imgs:
                    coeff_, = sess.run([self.coeff], feed_dict = {self.images: input_img})
                    # print(coeff_.shape, self.coeff.shape)
                    coeffs.append(coeff_)

                # Generates a texture image
                cache = None
                coeff_ = coeffs[0]
                input_img = input_imgs[0]
                cache, texture_target = texture_generator.generate_texture_cached(coeff_, input_img, sess, cache)

                swapped = texture_renderer.render_texture(coeffs[1], texture_target, sess)

                return swapped[...,:3][...,::-1].astype(np.uint8)
