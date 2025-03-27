% Define file paths
file_path = '/home/vturino/PhD/projects/forearc_deformation/slab_geometries/sam_geometry.nc';
temp_file_path = '/home/vturino/PhD/projects/forearc_deformation/lithosphere_data/temp_with_slab2supp_modified_and_SAVANI.nc';

% Read data from sam_geometry.nc
lon = ncread(file_path, 'glon1');
lat = ncread(file_path, 'glat1');
dep = ncread(file_path, 'gdepth1');
slabs = ncread(file_path, 'slabs1');
comp = ncread(file_path, 'Composi1');
temp = ncread(file_path, 'slab_temp1');

% Access the data from the temperature file
temp_data = ncread(temp_file_path, 'finaltemp');  % Temperature data
temp_lon = ncread(temp_file_path, 'lon');         % Longitude from SAVANI model
temp_lat = ncread(temp_file_path, 'lat');         % Latitude from SAVANI model
temp_depth = ncread(temp_file_path, 'dep');       % Depth from SAVANI model

% Depth adjustment: limit the temperature depth to match sam_geometry depth
temp_depth_limited = temp_depth(temp_depth <= max(dep));

% Create meshgrids for the temperature data (after depth adjustment)
[temp_lon_grid, temp_lat_grid, temp_depth_grid] = meshgrid(temp_lon, temp_lat, temp_depth_limited);

% Check the size of temp_data and the meshgrid dimensions
disp(['Shape of temp_data: ', num2str(size(temp_data))]);
disp(['Shape of temp_lon_grid: ', num2str(size(temp_lon_grid))]);
disp(['Shape of temp_lat_grid: ', num2str(size(temp_lat_grid))]);
disp(['Shape of temp_depth_grid: ', num2str(size(temp_depth_grid))]);

% Ensure the temp_data has the correct shape (matching lon, lat, depth grid)
% Check if the number of elements in temp_data matches the grid dimensions
if numel(temp_data) ~= numel(temp_lon_grid)
    error('temp_data size does not match the meshgrid size.');
end

% Reshape temp_data to match the meshgrid dimensions (lon, lat, depth)
% Assuming the original data is in a shape like (lon, lat, depth)
temp_data_reshaped = reshape(temp_data, length(temp_lon), length(temp_lat), length(temp_depth_limited));

% Now we can proceed with interpolation

% Create meshgrids for the sam_geometry.nc grid (slab geometry grid)
[sam_lon_grid, sam_lat_grid, sam_depth_grid] = meshgrid(lon, lat, dep);

% Check the dimensions of the sam_geometry grid
disp(['Shape of sam_lon_grid: ', num2str(size(sam_lon_grid))]);
disp(['Shape of sam_lat_grid: ', num2str(size(sam_lat_grid))]);
disp(['Shape of sam_depth_grid: ', num2str(size(sam_depth_grid))]);

% Interpolate the temperature data onto the sam_geometry grid
temp_resampled = interp3(temp_lon_grid, temp_lat_grid, temp_depth_grid, ...
                         temp_data_reshaped, ...
                         sam_lon_grid, sam_lat_grid, sam_depth_grid, 'linear');

% Ensure the temperature does not exceed slab temperature
temp_resampled = min(temp_resampled, temp);  % Adjust temperature to not exceed slab temperature

% Save the resampled data back into a new NetCDF file
newfn = '/home/vturino/PhD/projects/forearc_deformation/lithosphere_data/temp_with_slab2supp_modified_and_SAVANI_resampled.nc';
ds = netcdf.create(newfn, 'NC_WRITE');

% Define dimensions
lon_dim = netcdf.defDim(ds, 'lon', length(lon));
lat_dim = netcdf.defDim(ds, 'lat', length(lat));
dep_dim = netcdf.defDim(ds, 'dep', length(dep));

% Define variables
lons = netcdf.defVar(ds, 'lon', 'double', lon_dim);
lats = netcdf.defVar(ds, 'lat', 'double', lat_dim);
deps = netcdf.defVar(ds, 'dep', 'double', dep_dim);
vartemp = netcdf.defVar(ds, 'finaltemp', 'double', [lat_dim lon_dim dep_dim]);

% Write data to the file
netcdf.putVar(ds, lons, lon);
netcdf.putVar(ds, lats, lat);
netcdf.putVar(ds, deps, dep);
netcdf.putVar(ds, vartemp, temp_resampled);

% Close the NetCDF file
netcdf.close(ds);

disp('Temperature data resampled and saved successfully.');
