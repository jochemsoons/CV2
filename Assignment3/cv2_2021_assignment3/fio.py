# Copyright (c) 3DUniversum BV. All rights reserved.

# THIS SOFTWARE IS PROVIDED BY 3DUNIVERSUM B.V. "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import cv2
import time
import glob
from collections import defaultdict

import numpy as np
import skimage.io
from scipy import misc


__all__ = [
    "load_obj",
    "save_obj"
]


def load_obj(obj_path):
    """Loads an OBJ file. Currently supports only vertex and facet tags.

    Parameters
    ----------
    obj_path : string
        Path to an object file.

    """
    vertices = []
    colors = []
    triangles = []
    with open(obj_path, 'r') as f:
        for line in f:
            tokens = line.split()
            if len(tokens) == 0:
                continue

            if tokens[0] == 'v':
                vertices.append(list(map(float, tokens[1:4])))
                if len(tokens) > 4:
                    colors.append(list(map(float, tokens[4:])))
            elif tokens[0] == 'f':
                triangles.append(
                    list(map(lambda x: int(x.split('/', 1)[0]) - 1, tokens[1:4])))

    if len(colors) > 0:
        return np.asarray(vertices, dtype=np.float32), np.asarray(colors, dtype=np.float32), np.asarray(triangles, dtype=np.int32)
    return np.asarray(vertices, dtype=np.float32), np.asarray(triangles, dtype=np.int32)


def save_obj(file_path, shp, t, tl, norms=None, vertex_mask=None, uv=None, texture=None):
    t1 = time.time()
    prefix = os.path.splitext(file_path)[0]
    with open(file_path, 'wb') as f:
        if (uv is not None) and (texture is not None):
            f.write(b"mtllib %s.mtl\n" %  str.encode(os.path.split(prefix)[-1]))
        old_to_new = np.zeros(shp.shape[0], dtype=np.int32)
        if vertex_mask is None:
            vertex_mask = np.ones(shp.shape[0], dtype=np.int32)

        old_indexes = np.arange(shp.shape[0])[vertex_mask > 0.5]

        if t is not None:
            data = np.hstack((shp, t))
        else:
            data = shp
        
        data = data[old_indexes]
        
        np.savetxt(
            f, data,
            fmt=' '.join(['v'] + ['%.5f'] * data.shape[1]))

        if uv is not None:
            np.savetxt(
                f, uv,
                fmt=' '.join(['vt'] + ['%.5f'] * 2))

        old_to_new[old_indexes] = np.arange(len(old_indexes))

        if norms is not None:
            np.savetxt(f, norms, fmt='vn %.5f %.5f %.5f')

        tl = tl.astype(np.int32)
        triangle_mask = np.all(vertex_mask[tl] > 0.5, axis=1)
        tl = tl[triangle_mask]

        tt = old_to_new[tl] + 1

        tt = np.tile(np.expand_dims(tt, 2), ((1, 1, 2)))
        tt = np.reshape(tt, (tt.shape[0], -1))

        np.savetxt(f, tt, fmt='f %d/%d %d/%d %d/%d')

    if (uv is not None) and (texture is not None):
        with open(prefix + ".mtl", 'w') as f:
            f.write("newmtl material0\n")
            f.write("Ka 1.000000 1.000000 1.000000\n")
            f.write("Kd 1.000000 1.000000 1.000000\n")
            f.write("Ks 0.000000 0.000000 0.000000\n")
            f.write("Tr 1.000000\n")
            f.write("illum 1\n")
            f.write("Ns 0.000000\n")
            f.write("map_Kd %s.jpg\n" % os.path.split(prefix)[-1])

        cv2.imwrite("%s.jpg" % prefix, texture) 

    print('Export to an obj took %.5f secs' % (time.time() - t1))
