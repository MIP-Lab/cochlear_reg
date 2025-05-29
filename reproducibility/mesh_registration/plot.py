import numpy as np
import pyvista as pv

def mesh_to_pyvista(vertices, triangles):
    faces_4d = np.zeros((triangles.shape[0], 4))
    faces_4d[:, 1:4] = triangles
    faces_4d[:, 0] = 3
    faces_flatten = faces_4d.flatten().astype(np.int32)
    mesh = pv.PolyData(vertices, faces_flatten)
    return mesh

def read_mesh_file(file):
    with open(file, 'rb') as f:
        mid = np.fromfile(f, 'int32', 1)[0]
        numverts = np.fromfile(f, 'int32', 1)[0]
        numtris = np.fromfile(f, 'int32', 1)[0]
        n = np.fromfile(f, 'int32', 1)[0]
        if n == -1:
            orient = np.fromfile(f, 'int32', 3)
            dim = np.fromfile(f, 'int32', 3)
            sz = np.fromfile(f, 'float32', 3)
            color = np.fromfile(f, 'int32', 3)
        else:
            color = np.fromfile(f, 'int32', 2)
        vertices = np.fromfile(f, 'float32', numverts * 3)
        vertices = vertices.reshape((-1, 3))
        triangles = np.fromfile(f, 'int32', numtris * 3)
        triangles = triangles.reshape((-1, 3))
        
        return mesh_to_pyvista(vertices, triangles)

atlas_mesh = read_mesh_file('atlas_CC.mesh')
patient_mesh = read_mesh_file('patient_CC.mesh')
atlas2patient_mesh = read_mesh_file('patient_CC.mesh')

pl = pv.Plotter()

pl.add_mesh(atlas_mesh, color='gray', opacity=0.5)
pl.add_mesh(patient_mesh, color='red', opacity=0.5)
pl.add_mesh(atlas2patient_mesh, color='green', opacity=0.5)

pl.show()