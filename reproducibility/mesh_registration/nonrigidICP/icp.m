
addpath('../');

set(groot,'defaultFigureVisible','off')

% fv = stlread('../data/cochlear/128/atlas/CC_mesh/atlas.stl');

mesh1 = meshread('../../data/cochlear/128/atlas/CC_mesh/atlas_0p7_nc.mesh');

testfiles = dir('../../data/cochlear/128/test93/CC_mesh/*_0p8.mesh');

for i=1:length(testfiles)
    
    name = replace(testfiles(i).name, '.mesh', '');

    mesh2 = meshread(['../../data/cochlear/128/test93/CC_mesh/', name, '.mesh']);

    [registered,targetV,targetF] = nonrigidICPv1(mesh1.vertices', mesh2.vertices', ...
        mesh1.triangles'+1, mesh2.triangles'+1, 20, 0);
    
    mesh2.vertices = registered';
%     meshwrite(['../../data/cochlear/128/test93/CC_mesh/', name, '_icpnonrgd_nc.mesh'], mesh2)
    break
end