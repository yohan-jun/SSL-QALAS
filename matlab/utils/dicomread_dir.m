function [ img ] = dicomread_dir( dicom_dir )
%DICOMREAD_DIR Summary of this function goes here
%   Detailed explanation goes here


curr_path = pwd;

cd(dicom_dir)

d = dir;


for t = 3:length(d)
    img(:,:,t-2) = dicomread(d(t).name);
end
% for t = 5:length(d)
%     img(:,:,t-4) = dicomread(d(t).name);
% end

cd(curr_path)



end

