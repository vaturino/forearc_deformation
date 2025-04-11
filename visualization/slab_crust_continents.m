 %Specify the file pat
file_path = 'sam_geometry_continents.nc';
addpath("/home/vturino/PhD/projects/forearc_deformation/slab_geometries/");
% the variable(s) from the NetCDF file
% temp_path = 'temp_with_slab2supp_modified_and_SAVANI.nc';
% addpath("/home/vturino/PhD/projects/forearc_deformation/lithosphere_data/");


% Get the variable ID
lon = ncread(file_path, 'glon_var');
lat=ncread(file_path, 'glat_var');
dep = ncread(file_path, 'gdepth_var');
bound = ncread(file_path, 'C_bound');
slab = ncread(file_path, 'C_slab');
crust = ncread(file_path, 'C_crust');
% cont = ncread(file_path, 'continents');
op = ncread(file_path, 'C_OP');

% Access the data
[glonm,glatm,gdepthm] = meshgrid(lon,lat,dep);
x = (6371-gdepthm).*cosd(glonm).*sind(90-glatm);
y = (6371-gdepthm).*sind(glonm).*sind(90-glatm);
z = (6371-gdepthm).*cosd(90-glatm);
bound1=permute(bound, [3,2,1]);
% cont1=permute(cont, [3,2,1]);
slab1=permute(slab, [3,2,1]);
crust1=permute(crust, [3,2,1]);
op1=permute(op, [3,2,1]);

% Print some statistics about the data for debugging
disp(['Plate boundary range: ', num2str(min(bound1(:))), ' to ', num2str(max(bound1(:)))]);
% disp(['Continent range: ', num2str(min(cont1(:))), ' to ', num2str(max(cont1(:)))]);
disp(['Slab range: ', num2str(min(slab1(:))), ' to ', num2str(max(slab1(:)))]);
disp(['Crust range: ', num2str(min(crust1(:))), ' to ', num2str(max(crust1(:)))]);
disp(['OP range: ', num2str(min(op1(:))), ' to ', num2str(max(op1(:)))]);



vtkwrite('paraview_inputs/sam_cont.vtk', ...
    'structured_grid', x*1000, y*1000, z*1000, ...
    'scalars', 'plate_boundaries', bound1, ...
    'scalars', 'crust', crust1, ...
    'scalars', 'slab', slab1, ...
    'scalars', 'op', op1, ...
    'binary');
    % 'scalars', 'continent', cont1, ...
    
