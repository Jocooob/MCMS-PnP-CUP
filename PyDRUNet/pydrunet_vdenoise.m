function denoisedv = pydrunet_vdenoise(noisyv, para)
% 预处理数据，并调用PyDenoiser去噪
netvnorm = true;
if isfield(para,'netvnorm'), netvnorm = para.netvnorm;   end
if netvnorm
    %%%%% [start] normalization, like VBM4D %%%%%
    maxz = max(noisyv(:));
    minz = min(noisyv(:));
    scale = 0.8;
    shift = (1-scale)/2;
    noisyv = (noisyv-minz)/(maxz-minz);
    noisyv = noisyv*scale+shift;

    sigma = para.sigma;%/(maxz-minz)*scale;
    %%%%% [start] normalization, like VBM4D %%%%%
else
    % set noise level map
    sigma = para.sigma;
end
[h,w,~]=size(noisyv);

if para.sigma > 0.5
    noisyv = noisyv*(0.5/para.sigma);
    sigma = 0.5;
end
noisyv = single(noisyv);

if mod(h,8)~=0
    d=8-mod(h,8);
    noisyv_cup = cat(1,noisyv(:,:,:), noisyv(end-d+1:end,:,:)) ;
else
    noisyv_cup = noisyv;
end
if mod(w,8)~=0
    d=8-mod(w,8);
    noisyv_cup = cat(2,noisyv_cup(:,:,:), noisyv_cup(:,end-d+1:end,:)) ;
end

%ceil 向正无穷取整 进一法
drunet_denoiser = para.pydenoise;
%denoisedv_py = py.fastdvdnet_vdenoise.main...
%(pyargs('vnoisy',noisyv_cup,'sigma',sigma));
denoisedv_py = drunet_denoiser(noisyv_cup, max(sigma,0.004));  % Denoise
denoisedv = double(denoisedv_py);

if mod(h,8)~=0
    d=8-mod(h,8);
    denoisedv = double(denoisedv(1:end-d,:,:));
end
if mod(w,8)~=0
    d=8-mod(w,8);
    denoisedv = double(denoisedv(:,1:end-d,:));
end

if para.sigma > 0.5
    denoisedv = denoisedv*(para.sigma/0.5);
end

if netvnorm
    denoisedv = (denoisedv-shift)/scale;
    denoisedv = denoisedv*(maxz-minz)+minz;
end

% GPU=gpuDevice(1);
% reset(GPU);
end