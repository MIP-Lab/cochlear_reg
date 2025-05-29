import numpy as np
import os
from subprocess import check_call, DEVNULL, STDOUT, PIPE
import time
from numpy.lib.ufunclike import fix
import math
import re
try:
    import nibabel as nib
except:
    pass
from .utils import create_temp_folder, write_mesh, im2volume, delete_temp_folder

cur_folder = __file__.replace('api.py', '').replace('\\', '/')[: -1]

def mesh2mask(vertices, triangles, shape, volsz, remove_temp_folder=True):

    assert min(volsz) > 0, 'The vol size needs to be greater than 0, if it is not, multiply by 10 until it is. Otherwise Mesh2MaskS.exe will fail.'

    folder_id = create_temp_folder()
    # print(f'using folder {folder_id}')

    meshfile_path = f'{cur_folder}/temp/{folder_id}/mesh.mesh'
    maskfile_path = f'{cur_folder}/temp/{folder_id}/mesh.mask'

    write_mesh(vertices, triangles, meshfile_path)

    cmd = f'{cur_folder}/bin/Mesh2MaskS.exe'
    check_call([cmd, meshfile_path, *map(str, shape), *map(str, volsz), '1', maskfile_path], stdout=DEVNULL, stderr=STDOUT)

    mask = im2volume(maskfile_path, shape, np.uint8)
    mask[mask < 127] = 0
    mask[mask > 0] = 1

    if remove_temp_folder:
        delete_temp_folder(folder_id)

    return mask