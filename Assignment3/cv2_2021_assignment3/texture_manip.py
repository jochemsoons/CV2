import numpy as np
import cv2

import fio


def rotate(l, n):
    return l[n:] + l[:n]

def get_uv():
    vertex, color, tri = fio.load_obj("annotation/000002._mesh.obj")
    x = vertex[:,0]
    y = vertex[:,1]
    z = vertex[:,2]

    # rho = np.sqrt(z**2 + x**2)
    # rho = np.sqrt(x**2 + y**2)

    phi = np.arctan2(-z, x)
    u = (phi + np.pi) / (np.pi)

    v = (y + 1) / 2
    print(np.min(u), np.max(u), np.min(v), np.max(v))

    edges = set()
    for id0, id1, id2 in tri:
        edges.update({(id0, id1), (id1, id2), (id2, id0)})

    edges2 = set()
    neigb = dict()

    # finds all non-manifold edges
    for edge in edges:
        if (edge[1], edge[0]) in edges:
            continue

        edges2.add(edge)

        neigb[edge[0]] = edge[1]

    polygons = []

    # finds all closed polygons
    while len(edges2) > 0:
        current_edge = next(iter(edges2))
        print("CE", current_edge)
        polygon = [current_edge[0]]

        while neigb[polygon[-1]] != polygon[0]:
            polygon.append(neigb[polygon[-1]])

            edges2.remove(tuple(polygon[-2:]))

        edges2.remove((polygon[-1], polygon[0]))

        print("Found polygon", len(polygon), "edges")
        polygons.append(polygon)

    assert len(polygons) == 2

    mouth_curve2 = min(polygons, key=lambda x: len(x))

    solutions = []

    # Triangulates, minimizing the sum of distance. A better solution should require dynamic programming.
    for i in range(len(mouth_curve2)):
        mouth_curve = rotate(mouth_curve2, i)
        edge0, mouth_curve = mouth_curve[:2], mouth_curve[2:]

        total_area = 0

        extra_tri = []
        while len(mouth_curve) > 0:
            option1 = mouth_curve[0]
            option2 = mouth_curve[-1]

            area1 = np.sum((vertex[option1] - vertex[edge0[0]]) ** 2)
            area2 = np.sum((vertex[edge0[1]] - vertex[option2]) ** 2)

            if area1 < area2:
                extra_tri.append([edge0[0], edge0[1], option1][::-1])
                mouth_curve = mouth_curve[1:]
                edge0 = (edge0[0], option1)
                total_area += area1
            else:
                extra_tri.append([option2, edge0[0], edge0[1]][::-1])
                mouth_curve = mouth_curve[:-1]
                edge0 = (option2, edge0[1])
                total_area += area2

        print('solution %d, area %.8f' % (i, total_area))

        solutions.append((total_area, extra_tri))

    extra_tri = min(solutions, key=lambda x: x[0])[1]

    tri1 = np.concatenate((tri, extra_tri), axis=0)

    return u, v, tri, tri1


def cross_product(a, b):
    return a[0] * b[1] - b[0] * a[1]


def bar_coord_cache(uv_source, texture_source, uv_target, triangles, vertex_norm, z_buffer, n_base_tri, do_vertex_cooling=True):
   # vertex_norm = vertex_norm / np.sqrt(np.sum(vertex_norm ** 2, axis=-1, keepdims=True))
    tn = vertex_norm[triangles.reshape((-1,)).astype(int)].reshape((-1, 3, 3))
    tn = np.sum(tn, axis=1)
    tn = tn / np.sqrt(np.sum(tn ** 2, axis=-1, keepdims=True))

    texture_target = np.zeros_like(texture_source)

    H = texture_source.shape[0]
    W = texture_source.shape[1]

    def bbox(v0, v1, v2):
        u_min = int(max(0, np.floor(min(v0[0], v1[0], v2[0]))))
        u_max = int(min(W - 1, np.ceil(max(v0[0], v1[0], v2[0]))))

        v_min = int(max(0, np.floor(min(v0[1], v1[1], v2[1]))))
        v_max = int(min(H - 1, np.ceil(max(v0[1], v1[1], v2[1]))))

        return u_min, u_max, v_min, v_max

    uv_target = np.asarray([[0, 1]]) + uv_target * np.asarray([[1, -1]])

    p = np.empty([2])

    cache = []
    for i, triangle, norm in zip(range(len(triangles)), triangles.astype(int), tn):
        if do_vertex_cooling and (norm[2] < 0) and (i < n_base_tri):
            continue

        id0, id1, id2 = triangle

        v0 = uv_target[id0] * H
        v1 = uv_target[id1] * H
        v2 = uv_target[id2] * H
        v02 = v2 - v0
        v01 = v1 - v0

        u_min, u_max, v_min, v_max = bbox(v0, v1, v2)
        tri_a = cross_product(v1 - v0, v2 - v0)
        for v in range(v_min, v_max + 1):
            p[1] = v
            for u in range(u_min, u_max + 1):
                p[0] = u
                v0p = p - v0

                b1 = cross_product(v0p, v02) / tri_a
                b2 = cross_product(v01, v0p) / tri_a
                if (b1 < 0) or (b2 < 0) or (b1 + b2 > 1):
                    continue

                cache.append((u, v, b1, b2, id0, id1, id2))
    return cache


def create_final_texture_from_cache(cache, uv_source, texture_source, uv_target, triangles, vertex_norm, z_buffer, n_base_tri, do_vertex_cooling=True):
    # vertex_norm = vertex_norm / np.sqrt(np.sum(vertex_norm ** 2, axis=-1, keepdims=True))
    tn = vertex_norm[triangles.reshape((-1,)).astype(int)].reshape((-1, 3, 3))
    tn = np.sum(tn, axis=1)
    tn = tn / np.sqrt(np.sum(tn ** 2, axis=-1, keepdims=True))

    texture_target = np.zeros_like(texture_source)

    H = texture_source.shape[0]
    W = texture_source.shape[1]

    uv_source = (np.asarray([[0, 1]]) + uv_source * np.asarray([[1, -1]])) * H

    cache = np.asarray(cache)
    b1 = cache[:, 2]
    b2 = cache[:, 3]
    b0 = 1 - b1 - b2
    u = cache[:,0].astype(np.int32)
    v = cache[:,1].astype(np.int32)
    ids = cache[:,4:].astype(np.int32)
    v_source = uv_source[ids]
    source_coords = np.sum(np.expand_dims(np.vstack((b0, b1, b2)).T, 2) * v_source, axis=1)
    x0_y0s = np.floor(source_coords).astype(np.int32)
    x2_y2s = np.ceil(source_coords).astype(np.int32)
    
    z_buffer2 = -np.ones_like(z_buffer) * np.inf
    for u, v, x0_y0s, x2_y2s, source_coord in zip(u, v, x0_y0s, x2_y2s, source_coords):
        x0, y0 = x0_y0s[0], x0_y0s[1]
        x2, y2 = x2_y2s[0], x2_y2s[1]
        x1, y1 = source_coord

        z = (z_buffer[y0, x0] + z_buffer[y2, x2]) / 2
        if z > z_buffer2[v, u]:
            z_buffer2[v, u] = z

            texture_target[v, u] = (x1 - x0) * (y1 - y0) * texture_source[y2, x2] +\
                (x2 - x1) * (y1 - y0) * texture_source[y2, x0] +\
                (x2 - x1) * (y2 - y1) * texture_source[y0, x0] +\
                (x1 - x0) * (y2 - y1) * texture_source[y0, x2]


    mask = np.all(texture_target < 0.000001, axis=-1).astype(np.uint8)
    texture_target = cv2.inpaint(texture_target, mask, 3, cv2.INPAINT_TELEA)

    return texture_target



def create_final_texture(uv_source, texture_source, uv_target, triangles, vertex_norm, z_buffer, n_base_tri, do_vertex_cooling=True):
    # vertex_norm = vertex_norm / np.sqrt(np.sum(vertex_norm ** 2, axis=-1, keepdims=True))
    tn = vertex_norm[triangles.reshape((-1,)).astype(int)].reshape((-1, 3, 3))
    tn = np.sum(tn, axis=1)
    tn = tn / np.sqrt(np.sum(tn ** 2, axis=-1, keepdims=True))

    texture_target = np.zeros_like(texture_source)

    H = texture_source.shape[0]
    W = texture_source.shape[1]

    def bbox(v0, v1, v2):
        u_min = int(max(0, np.floor(min(v0[0], v1[0], v2[0]))))
        u_max = int(min(W - 1, np.ceil(max(v0[0], v1[0], v2[0]))))

        v_min = int(max(0, np.floor(min(v0[1], v1[1], v2[1]))))
        v_max = int(min(H - 1, np.ceil(max(v0[1], v1[1], v2[1]))))

        return u_min, u_max, v_min, v_max

    uv_source = np.asarray([[0, 1]]) + uv_source * np.asarray([[1, -1]])
    uv_target = np.asarray([[0, 1]]) + uv_target * np.asarray([[1, -1]])

    p = np.empty([2])
    z_buffer2 = -np.ones_like(z_buffer) * np.inf
    for i, triangle, norm in zip(range(len(triangles)), triangles.astype(int), tn):
        if do_vertex_cooling and (norm[2] < 0) and (i < n_base_tri):
            continue

        id0, id1, id2 = triangle

        v0_source = uv_source[id0] * H
        v1_source = uv_source[id1] * H
        v2_source = uv_source[id2] * H

        v0 = uv_target[id0] * H
        v1 = uv_target[id1] * H
        v2 = uv_target[id2] * H
        v02 = v2 - v0
        v01 = v1 - v0

        u_min, u_max, v_min, v_max = bbox(v0, v1, v2)
        tri_a = cross_product(v1 - v0, v2 - v0)
        for v in range(v_min, v_max + 1):
            p[1] = v
            for u in range(u_min, u_max + 1):
                p[0] = u
                v0p = p - v0

                b1 = cross_product(v0p, v02) / tri_a
                b2 = cross_product(v01, v0p) / tri_a
                if (b1 < 0) or (b2 < 0) or (b1 + b2 > 1):
                    continue
                
                b0 = 1 - b1 - b2

                source_coord = b0 * v0_source + b1 * v1_source + b2 * v2_source

                x0, y0 = np.floor(source_coord).astype(np.int32)
                x1, y1 = source_coord
                x2, y2 = np.ceil(source_coord).astype(np.int32)

                z = (z_buffer[y0, x0] + z_buffer[y2, x2]) / 2
                if z > z_buffer2[v, u]:
                    z_buffer2[v, u] = z

                    texture_target[v, u] = (x1 - x0) * (y1 - y0) * texture_source[y2, x2] +\
                        (x2 - x1) * (y1 - y0) * texture_source[y2, x0] +\
                        (x2 - x1) * (y2 - y1) * texture_source[y0, x0] +\
                        (x1 - x0) * (y2 - y1) * texture_source[y0, x2]


    mask = np.all(texture_target < 0.000001, axis=-1).astype(np.uint8)
    texture_target = cv2.inpaint(texture_target, mask, 3, cv2.INPAINT_TELEA)


    return texture_target
