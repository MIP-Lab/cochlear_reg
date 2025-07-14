
addpath('nonrigidICP');

set(groot,'defaultFigureVisible','off')

% fv = stlread('../data/cochlear/128/atlas/CC_mesh/atlas.stl');

all_strcutreus = ["MD_upsample1", "MD_upsample2"];


for si=1:length(all_strcutreus)

    s = convertStringsToChars(all_strcutreus(si));

    % testfiles = dir('segnet_mesh/MD/*.mesh');
    
    testfiles = dir(['segnet_mesh/', s, '/*.mesh']);

    for i=1:length(testfiles)

        name = replace(testfiles(i).name, '.mesh', '');

        if exist(['segnet_a2p_mesh_post/', s, '/', name, '.mesh'])
            continue
        end
        
        mesh1 = meshread(['atlas_mesh/atlas_', 'md', '.mesh']);
    
        mesh2 = meshread(['segnet_mesh_post/',s, '/', name, '.mesh']);
        
        moving = mesh1;
        fixed = mesh2;
    
        % [registered,targetV,targetF] = nonrigidICPv1(fixed.vertices', moving.vertices', ...
        %         fixed.triangles'+1, moving.triangles'+1, 20, 0);
        
        try
            [registered,targetV,targetF] = nonrigidICPv1(fixed.vertices', moving.vertices', ...
                fixed.triangles'+1, moving.triangles'+1, 20, 0);
            
            moving.vertices = registered';
            meshwrite(['segnet_a2p_mesh_post/',s, '/', name, '.mesh'], moving)
        catch
            mesh2b = surfaceMesh(fixed.vertices', fixed.triangles' + 1);
            mesh2b = removeFreeEdges(mesh2b);
            [registered,targetV,targetF] = nonrigidICPv1(mesh2b.Vertices, moving.vertices', ...
                double(mesh2b.Faces), moving.triangles'+1, 20, 0);
            
            moving.vertices = registered';
            meshwrite(['segnet_a2p_mesh_post/', s, '/', name, '.mesh'], moving)
        end
    
    end

end