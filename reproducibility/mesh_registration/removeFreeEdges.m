function[mesh1]=removeFreeEdges(mesh)

V = mesh.Vertices;
F = mesh.Faces;

fk1 = F(:,1);
fk2 = F(:,2);
fk3 = F(:,3);

ed1=sort([fk1 fk2 ]')';
ed2=sort([fk1 fk3 ]')';
ed3=sort([fk2 fk3 ]')';

%single edges
ed=[ed1 ;ed2 ;ed3];
[etemp1,ia,ic]=unique(ed,'rows','stable');
esingle=ed(ia,:);

%dubbles
edouble=removerows(ed,ia);

C = setdiff(esingle,edouble,'rows');

[tf, free_index] = ismember(C, ed, 'rows');

F1 = F;

for i=1:length(free_index)
    face_index = free_index(i) - length(F) * floor(free_index(i)/length(F));
    F1 = [F1; F(face_index, :)];
end

mesh1 = surfaceMesh(V, F1);


