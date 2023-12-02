%%

clear; clc; close all;

set(0,'DefaultFigureWindowStyle','docked')
addpath(genpath('utils'));


%%

savepath = 'h5_data/multicoil_train/';
savename = 'train_data.h5';

compare_ref_map     = 1;
% 1 (compare ssl-qalas with reference maps (e.g., dictionary matching) during training)
% 0 (no comparison > will use zeros)

load_b1_map         = 1;
% 1 (load pre-acquired b1 map)
% 0 (no b1 map)


%% LOAD DATA

dpath       = 'dicom_data/image/';

if compare_ref_map == 1
    load('map_data/ref_map.mat');
end

fprintf('loading data ... ');
tic
input_img   = single(dicomread_dir(dpath));
input_img = input_img(:,:,1:2:end);
input_img   = reshape(input_img,[size(input_img,1),size(input_img,2),size(input_img,3)/5,1,5]);
toc


%% Brain Mask (simple thresholding mask)

threshold = 50;

[Nx,Ny,Nz,~,~]  = size(input_img);
bmask           = zeros(Nx,Ny,Nz,'single');

for slc = 1:size(input_img,3)
   bmask(:,:,slc) = imfill(squeeze(rsos(input_img(:,:,slc,1,:),5)) > threshold, 'holes');
end


%%

input_img       = input_img./max(input_img(:));

sens            = ones(Nx,Ny,Nz,1,'single');
mask            = ones(Nx,Ny,'single');
if compare_ref_map == 0
    T1_map = ones(Nx,Ny,Nz,'single');
    T2_map = ones(Nx,Ny,Nz,'single');
    PD_map = ones(Nx,Ny,Nz,'single');
    IE_map = ones(Nx,Ny,Nz,'single');
end
if load_b1_map == 0
    B1_map = ones(Nx,Ny,Nz,'single');
end

input_img   = permute(input_img,[2,1,4,3,5]);
sens        = permute(sens,[2,1,4,3]);

T1_map      = permute(T1_map,[2,1,3]);
T2_map      = permute(T2_map,[2,1,3]);
PD_map      = permute(PD_map,[2,1,3]);
IE_map      = permute(IE_map,[2,1,3]);
B1_map      = permute(B1_map,[2,1,3]);

bmask       = permute(bmask,[2,1,3]);
mask        = permute(mask,[2,1]);

kspace_acq1 = single(input_img(:,:,:,:,1));
kspace_acq2 = single(input_img(:,:,:,:,2));
kspace_acq3 = single(input_img(:,:,:,:,3));
kspace_acq4 = single(input_img(:,:,:,:,4));
kspace_acq5 = single(input_img(:,:,:,:,5));


%% SAVE DATA

fprintf('save h5 data ... ');

tic

file_name   = strcat(savepath,savename);

att_patient = '0000';
att_seq     = 'QALAS';

kspace_acq1     = permute(kspace_acq1,[4,3,2,1]);
kspace_acq2     = permute(kspace_acq2,[4,3,2,1]);
kspace_acq3     = permute(kspace_acq3,[4,3,2,1]);
kspace_acq4     = permute(kspace_acq4,[4,3,2,1]);
kspace_acq5     = permute(kspace_acq5,[4,3,2,1]);
coil_sens       = permute(sens,[4,3,2,1]);

saveh5(struct('kspace_acq1', kspace_acq1, 'kspace_acq2', kspace_acq2, ...
              'kspace_acq3', kspace_acq3, 'kspace_acq4', kspace_acq4, ...
              'kspace_acq5', kspace_acq5), ...
              file_name, 'ComplexFormat',{'r','i'});

h5create(file_name,'/reconstruction_t1',[Ny,Nx,Nz],'Datatype','single');
h5write(file_name, '/reconstruction_t1', T1_map);
h5create(file_name,'/reconstruction_t2',[Ny,Nx,Nz],'Datatype','single');
h5write(file_name, '/reconstruction_t2', T2_map);
h5create(file_name,'/reconstruction_pd',[Ny,Nx,Nz],'Datatype','single');
h5write(file_name, '/reconstruction_pd', PD_map);
h5create(file_name,'/reconstruction_ie',[Ny,Nx,Nz],'Datatype','single');
h5write(file_name, '/reconstruction_ie', IE_map);
h5create(file_name,'/reconstruction_b1',[Ny,Nx,Nz],'Datatype','single');
h5write(file_name, '/reconstruction_b1', B1_map);

h5create(file_name,'/mask_acq1',[Ny,1],'Datatype','single');
h5write(file_name, '/mask_acq1', mask(:,1));
h5create(file_name,'/mask_acq2',[Ny,1],'Datatype','single');
h5write(file_name, '/mask_acq2', mask(:,1));
h5create(file_name,'/mask_acq3',[Ny,1],'Datatype','single');
h5write(file_name, '/mask_acq3', mask(:,1));
h5create(file_name,'/mask_acq4',[Ny,1],'Datatype','single');
h5write(file_name, '/mask_acq4', mask(:,1));
h5create(file_name,'/mask_acq5',[Ny,1],'Datatype','single');
h5write(file_name, '/mask_acq5', mask(:,1));

h5create(file_name,'/mask_brain',[Ny,Nx,Nz],'Datatype','single');
h5write(file_name, '/mask_brain', single(bmask));

att_norm_t1 = norm(T1_map(:));
att_max_t1  = max(T1_map(:));
att_norm_t2 = norm(T2_map(:));
att_max_t2  = max(T2_map(:));
att_norm_pd = norm(PD_map(:));
att_max_pd  = max(PD_map(:));
att_norm_ie = norm(IE_map(:));
att_max_ie  = max(IE_map(:));
att_norm_b1 = norm(B1_map(:));
att_max_b1  = max(B1_map(:));

h5writeatt(file_name,'/','norm_t1',att_norm_t1);
h5writeatt(file_name,'/','max_t1',att_max_t1);
h5writeatt(file_name,'/','norm_t2',att_norm_t2);
h5writeatt(file_name,'/','max_t2',att_max_t2);
h5writeatt(file_name,'/','norm_pd',att_norm_pd);
h5writeatt(file_name,'/','max_pd',att_max_pd);
h5writeatt(file_name,'/','norm_ie',att_norm_ie);
h5writeatt(file_name,'/','max_ie',att_max_ie);
h5writeatt(file_name,'/','norm_b1',att_norm_b1);
h5writeatt(file_name,'/','max_b1',att_max_b1);
h5writeatt(file_name,'/','patient_id',att_patient);
h5writeatt(file_name,'/','acquisition',att_seq);


%%

dset = ismrmrd.Dataset(file_name);

header = [];

% Experimental Conditions (Required)
header.experimentalConditions.H1resonanceFrequency_Hz   = 128000000; % 3T

% Acquisition System Information (Optional)
header.acquisitionSystemInformation.receiverChannels    = 32;

% The Encoding (Required)
header.encoding.trajectory = 'cartesian';
header.encoding.encodedSpace.fieldOfView_mm.x   = Nx;
header.encoding.encodedSpace.fieldOfView_mm.y   = Ny;
header.encoding.encodedSpace.fieldOfView_mm.z   = Nz;
header.encoding.encodedSpace.matrixSize.x       = Nx*2;
header.encoding.encodedSpace.matrixSize.y       = Ny;
header.encoding.encodedSpace.matrixSize.z       = Nz;

% Recon Space
header.encoding.reconSpace.fieldOfView_mm.x     = Nx;
header.encoding.reconSpace.fieldOfView_mm.y     = Ny;
header.encoding.reconSpace.fieldOfView_mm.z     = Nz;
header.encoding.reconSpace.matrixSize.x         = Nx;
header.encoding.reconSpace.matrixSize.y         = Ny;
header.encoding.reconSpace.matrixSize.z         = Nz;

% Encoding Limits
header.encoding.encodingLimits.kspace_encoding_step_1.minimum   = 0;
header.encoding.encodingLimits.kspace_encoding_step_1.maximum   = Nx-1;
header.encoding.encodingLimits.kspace_encoding_step_1.center    = Nx/2;
header.encoding.encodingLimits.kspace_encoding_step_2.minimum   = 0;
header.encoding.encodingLimits.kspace_encoding_step_2.maximum   = 0;
header.encoding.encodingLimits.kspace_encoding_step_2.center    = 0;

% Serialize and write to the data set
xmlstring = ismrmrd.xml.serialize(header);
dset.writexml(xmlstring);

% Write the dataset
dset.close();
toc
