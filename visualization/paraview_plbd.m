 %Specify the file pat
file_path = 'sam_geometry.nc';
addpath("/home/vturino/PhD/projects/forearc_deformation/slab_geometries/");
% the variable(s) from the NetCDF file
% temp_path = 'temp_with_slab2supp_modified_and_SAVANI.nc';
% addpath("/home/vturino/PhD/projects/forearc_deformation/lithosphere_data/");

% 


% Get the variable ID
lon = ncread(file_path, 'glon1');
lat=ncread(file_path, 'glat1');
dep = ncread(file_path, 'gdepth1');
slabs = ncread(file_path, 'slabs1');
comp = ncread(file_path, 'Composi1');
temp = ncread(file_path, 'slab_temp1');

% Access the data
[glonm,glatm,gdepthm] = meshgrid(lon,lat,dep);
x = (6371-gdepthm).*cosd(glonm).*sind(90-glatm);
y = (6371-gdepthm).*sind(glonm).*sind(90-glatm);
z = (6371-gdepthm).*cosd(90-glatm);
temp1=permute(temp, [3,2,1]);
slabs1=permute(slabs, [3,2,1]);
comp1=permute(comp, [3,2,1]);

% Print some statistics about the data for debugging
disp(['Temperature range: ', num2str(min(temp1(:))), ' to ', num2str(max(temp1(:)))]);
disp(['Slab geometry range: ', num2str(min(slabs1(:))), ' to ', num2str(max(slabs1(:)))]);
disp(['Composition range: ', num2str(min(comp1(:))), ' to ', num2str(max(comp1(:)))]);


vtkwrite('paraview_inputs/sam_temp.vtk', ...
    'structured_grid', x*1000, y*1000, z*1000, ...
    'scalars', 'temperature', temp1, ...
    'scalars', 'slab_geometry', slabs1, ...
    'scalars', 'crust_geometry', comp1, ...
    'binary');
