import pycip
import pyvista as pv
import numpy as np
import nibabel as nib

structures = ['MD', 'ST', 'SV', 'CC']
methods = ['chamfer', 'p2p', 'cgan', 'nnunet', 'segnet', 'aba', 'elastix']
method_names = ['chamfer', 'p2p', 'cGAN+ASM', 'nnU-Net', 'SegNet', 'ABA', 'Elastix']

case = 'PID_3055_PLANID_2567_RIGHT'

pl = pv.Plotter(shape=(7, 4), window_size=[400, 700])


camera_positions = [[(0.9977974966682144, 19.462603591715702, -42.28922551013209),
 (46.531570155284165, 44.51185579452931, 48.450885037682845),
 (0.5714596147280697, -0.8183713966686471, -0.06084542587125358)],
[(129.62115248106613, -79.46970198015029, 41.28232769488554),
 (50.923348484584984, 42.6869733062812, 50.896389348754845),
 (0.2646485765653247, 0.09496867518626928, 0.9596572730174624)],
[(133.60886118553086, -70.02355832992285, 40.628182811027216),
 (52.96943983382692, 37.75880111440111, 50.29712881717731),
 (0.15327661036325846, 0.026059634733845115, 0.9878396510330452)],
[(291.31898821913876, -88.61037678585527, 78.49777062221963),
 (62.08215922916863, 62.356971224481114, 62.000562961051),
 (-0.26443497190378973, -0.3014250588598967, 0.9160879212856972)]]

camera_positions = [[(-32.744992382375614, 5.337800962093128, -6.757675338836411),
 (46.63755205675612, 45.656742457642814, 48.0816512956516),
 (0.5677098384618235, -0.7863460706377966, -0.24365015186974032)],
[(128.67697542036856, -79.80851825402475, 37.858593491994334),
 (49.9791714238874, 42.34815703240672, 47.47265514586364),
 (0.2646485765653247, 0.09496867518626928, 0.9596572730174624)],
[(133.35548064478672, -70.06663733804096, 38.99519168311268),
 (52.71605929308279, 37.715722106282996, 48.66413768926277),
 (0.15327661036325846, 0.026059634733845115, 0.9878396510330452)],
[(256.1161768875775, -131.2482381290654, 80.99344250595236),
 (61.91425247065466, 61.67941447791391, 54.96739394702119),
 (-0.2150465182185342, -0.08522509016660032, 0.9728780391231886)]]

colors = {'MD': 'red', 'CC': 'orange', 'ST': 'green', 'SV': 'blue'}

for j, m in enumerate(methods):
    for i, s in enumerate(structures):

        gt_mesh = pycip.utils.read_mesh_asPyVista(f'../data/mesh/gt/{s}/{case}.mesh')
        gt_mask = nib.load(f'../data/mask/gt/{s}/{case}.nii.gz').get_fdata().astype(np.uint8)

        pred_mask = nib.load(f'../data/mask/{m}/{s}/{case}.nii.gz').get_fdata().astype(np.uint8)

        pred = pycip.utils.read_mesh_asPyVista(f'../data/mesh/{m}/{s}/{case}.mesh')

        p2p = np.median((np.sqrt(((pred.points - gt_mesh.points) ** 2).sum(axis=1)) * 0.2))
        dice = 2 * (pred_mask * gt_mask).sum() / ((pred_mask + gt_mask).sum() + 10e-6)

        pl.subplot(j, i)
        pl.add_text('DICE = %.2f' % dice, font_size=9, position='upper_left')
        pl.add_text('P2P_E = %.2fmm' % p2p, font_size=10, position='lower_left')
        pl.camera_position = camera_positions[i]
        if i == 3:
            pl.camera.zoom(1.5)
        else:
            pl.camera.zoom(1.3)
        pl.add_mesh(pred, color=colors[s], opacity=1)
        pl.add_mesh(gt_mesh, color='gray', opacity=0.4)


def cam_position():
    pl.subplot(0, 0)
    print(pl.camera_position)
    pl.subplot(0, 1)
    print(pl.camera_position)
    pl.subplot(0, 2)
    print(pl.camera_position)
    pl.subplot(0, 3)
    print(pl.camera_position)

pl.add_key_event('c', lambda : cam_position())
pl.add_key_event('s', lambda : pl.save_graphic(f'mesh_different_methods_solid.pdf'))

pl.show()