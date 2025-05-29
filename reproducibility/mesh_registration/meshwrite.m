function meshwrite(filename,mesh)
% Author: Jack H. Noble
% Date: 6/20/13
% Purpose: Used to write a mesh struct to a type '.mesh' file readable by the MeshEditor.exe
 
% inputs:
% filename - desired output path. You should use a filename with extension '.mesh'
% mesh - the mesh object to be written
 
% example call:
% meshwrite('tempcyl.mesh',msh);
 
 
fid = fopen(filename,'wb');
fwrite(fid,mesh.id,'int32');
fwrite(fid,mesh.numverts,'int32');
fwrite(fid,mesh.numtris,'int32');
fwrite(fid,255,'int32');
fwrite(fid,0,'int32');
fwrite(fid,0,'int32');
% fwrite(fid,mesh.orient,'int32');
% fwrite(fid,mesh.dim,'int32');
% fwrite(fid,mesh.sz,'float'); % fwrite(fid,mesh.sz,'float');
% fwrite(fid,mesh.color,'int32');
fwrite(fid,mesh.vertices,'float');
fwrite(fid,mesh.triangles,'int32');
if isfield(mesh,'opacity')
    fwrite(fid,mesh.opacity,'float');
    if isfield(mesh,'colormap')
        fwrite(fid,mesh.colormap.numcols,'int32');
        fwrite(fid,mesh.colormap.numverts,'int32');
        fwrite(fid,mesh.colormap.cols,'double');
        fwrite(fid,mesh.colormap.vertexindexes,'int32');
    end
end
fclose(fid);
end