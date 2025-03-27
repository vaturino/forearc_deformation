 %Specify the file pat
% file_path = 'sam_geometry.nc';
% addpath("/home/vturino/PhD/projects/forearc_deformation/slab_geometries/");
% the variable(s) from the NetCDF file
temp_path = 'temp_with_slab2supp_modified_and_SAVANI.nc';
addpath("/home/vturino/PhD/projects/forearc_deformation/lithosphere_data/");


% Get the variable ID
lon = ncread(temp_path, 'lon');
lat=ncread(temp_path, 'lat');
dep = ncread(temp_path, 'dep');
ftemp = ncread(temp_path, 'finaltemp');

% Access the data
[glonm,glatm,gdepthm] = meshgrid(lon,lat,dep);
x = (6371-gdepthm).*cosd(glonm).*sind(90-glatm);
y = (6371-gdepthm).*sind(glonm).*sind(90-glatm);
z = (6371-gdepthm).*cosd(90-glatm);
ftemp1=permute(ftemp,[3,2,1]);

disp(['Temperature range: ', num2str(min(ftemp1(:))), ' to ', num2str(max(ftemp1(:)))]);


vtkwrite('paraview_inputs/background_temp.vtk', ...
    'structured_grid', x*1000, y*1000, z*1000, ...
    'scalars', 'final_tenperature', ftemp1, ...
    'binary');
