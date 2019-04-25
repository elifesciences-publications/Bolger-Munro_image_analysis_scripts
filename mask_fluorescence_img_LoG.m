clear
close all
clc

%% Image stack
img_fullpath = '/home/joshscurll/Documents/MATLAB/Research/Madison/Madison_test.tif';
%img_fullpath = '/home/jscurll/Documents/MATLAB/Research/Image processing/Madison_test.tif';


%% LoG filter sizes
% sigmas is a vector (i.e. list) of different spot sizes to look for in the image,
% where each sigma value is a standard deviation (about 1/2 of the visual
% radius of a circular spot). E.g. sigmas = 0.75:0.25:3 will look for
% fluorsecent spots with standard deviations ranging (approx.) from 0.75 pixels
% to 3 pixels in intervals of 0.25 pixels (i.e. spots with diameters ranging 
% from about 3 pixels to 12 pixels). Or, e.g. sigmas = [1, 2.5, 5] will
% look for fluorscent spots with standard deviations of 1, 2.5 and 5 pixels
% (i.e. diameters of about 4, 10 and 20 pixels). Or, e.g. sigmas = 2 will
% only look for fluorescent spots with a standard deviation of approx. 2
% pixels (i.e. diameter of approx. 8 pixels).

sigmas = 0.5:0.25:3;


%% Thresholds
% Thresholds for detecting fluorescent features, specified as numbers of
% estimated background standard deviations (SD_bg, estimated from median 
% absolute deviations using SD_bg = 1.4826*MAD). Pixels will be included in 
% the mask if they meet the following criteria:
%
%   (1) Their intensity in the image filtered by a 3x3 averaging filter (to
%   smooth out noise) is > median(intensity) + threshold_intensity * SD_bg;
% AND
%   (2) Their maximum response to the LoG filters over all scales after
%   normalizing by the response SD_bg for each scale is > threshold_LoG;
% AND
%   (3) The difference between the absolute values of their most positive 
%   and most negative responses to the LoG filters at multiple scales, after
%   normalizing by each scale's response SD_bg as in (2), is > threshold_LoG_diff.
%
% To not use a particular threshold, set it to -Inf or NaN. E.g. to only 
% perform threshoding on LoG-filtered images but not on fluorescence intensity,
% set   threshold_intensity = -Inf;  or  threshold_intensity = NaN;
% Note that threshold_LoG has no effect unless it is > threshold_LoG_diff.
% Also note that thresholds can be < 0.

threshold_intensity = 3; % This applies directly to (averaged) fluorescence intensities.

threshold_LoG = 4;  % This applies to pixels' responses to LoG filters 
                    % (has no effect if not > threshold_LoG_diff).

threshold_LoG_diff = 0; % This applies to differences of LoG filter responses.
                        % A value of 0 simply means that the maximum
                        % positive LoG filter response of a pixel must be
                        % greater in magnitude than its maximum negative
                        % response.  E.g. imagine a small fluorescent spot 
                        % inside the hole of a fluorescent donut shape. At
                        % small sigma values the small spot will have a
                        % strong positive response to the LoG filter. But
                        % at large sigma values the donut will dominate the
                        % response to the LoG filter and its hole
                        % (including the small spot inside) will have a
                        % negative response to the LoG filter. This
                        % threshold parameter sets the relative importance
                        % of these two effects. Negative values of this
                        % threshold bias the relative importance towards
                        % the small spot (meaning that the dark ring
                        % between the spot and the donut might end up being
                        % included in the mask in order to guarantee the 
                        % spot's inclusion). Positive values of this
                        % threshold bias the relative importance towards
                        % the hole of the donut (meaning that the small
                        % spot might be excluded from the mask in order to
                        % prevent the dark ring between the spot and the
                        % donut from being included in the mask).


%% Specify whether to correct for photobleaching
% Even if this is set to true/1, photobleaching will only be corrected for
% if the image stack contains at least 15 frames. For movies with less than
% 15 frames, photobleaching is not corrected for.
correct_photobleach = true; % 1 or true to correct, 0 or false to use raw data


%% Apply variance-stabilizing transformation (VST) before filtering?
% Specify whether to apply the Anscombe VST before applying any filters to
% the images. This is a type of square-root transformation used to make the
% noise approximately Gaussian instead of Poissonian. In fluorescence
% microscopy images, noise is mostly Poissonian and its variance increases
% as fluorescence increases. This means that bright pixels typically have 
% greater noise variance than dim pixels. A VST transforms the data so that
% the noise variance becomes approximately independent of the pixel
% brightness. A good idea in principle, but spot detection results appear
% to be better without using the Anscombe VST before image filtering.

use_AnscombeVST = false; % 1 or true to use the Anscombe VST, 0 or false to use no VST.


%% Saving results

% Specify whether to save processed image stack as new TIFFs
save_tiffs = 1; % true or 1 to save new TIFF stacks, false or 0 to not save.

% Specify whether to save all results to a MAT-file
save_all = 1; % true or 1, or false or 0.

% Specify whether to save total_intensity to a csv file
save_csv = 1; % true or 1, or false or 0.

% Full path and name of folder where to save new TIFF stacks and results:
save_folder = '/home/joshscurll/Documents/MATLAB/Research/Madison/';


%% Show figures?
show_figs = true; % 1 or true, or 0 or false


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %% Random data to test
% img = poissrnd(5,[60,80]);
% blob = fspecial('disk',15);
% blob(blob>0) = 10;
% blob2 = fspecial('gaussian',size(blob),2);
% blob2 = blob2/max(blob2(:))*20;
% blob = [blob(:,1:end-2)+blob2(:,1:end-2),blob(:,3:end)];
% blob = [zeros(size(img,1)/2-ceil(size(blob,1)/2),size(blob,2)); blob; zeros(size(img,1)/2-floor(size(blob,1)/2),size(blob,2))];
% blob = [zeros(size(img,1),size(img,2)/2-ceil(size(blob,2)/2)), blob, zeros(size(img,1),size(img,2)/2-floor(size(blob,2)/2))];
% blob3 = fspecial('gaussian',9,0.75); blob3 = blob3/max(blob3(:))*20;
% blob4 = fspecial('gaussian',11,1.5); blob4 = blob4/max(blob4(:))*20;
% img = img + blob;
% img(3:2+size(blob3,1),3:2+size(blob3,2)) = img(3:2+size(blob3,1),3:2+size(blob3,2)) + blob3;
% img(end-size(blob4,1):end-1,end-size(blob4,2):end-1) = img(end-size(blob4,1):end-1,end-size(blob4,2):end-1) + blob4;
% img = img / max(img(:)) * (2^16-1); 
% nframes = 1;
% clear blob blob2 blob3 blob4


% For variance stabilizing transform (VST) for a low-pass filtered image,
% see
%   https://ieeexplore-ieee-org.ezproxy.library.ubc.ca/document/4379564
%   https://hal.archives-ouvertes.fr/hal-00017925/document
%   https://hal.archives-ouvertes.fr/hal-00813572/document


%% Read the entire image stack
imf = imfinfo(img_fullpath);
nframes = numel(imf);
img = zeros(imf(1).Height, imf(1).Width, nframes, 'single');
for ii = 1:nframes
    img(:,:,ii) = imread(img_fullpath, 'Index', ii);
end


%% Correct for photobleaching using median frame intensities and curve fitting
x = 1:nframes;
% Get the median fluorescence intensity of each frame
medf = zeros(nframes,1);
for ii = 1:nframes
    im = img(:,:,ii);
    medf(ii) = median(im(:));
end
if nframes > 14 && correct_photobleach
    % Fit a decaying curve to median intensity vs frame number
    opt = fitoptions('exp2','lower',[0,-Inf,0,0],'upper',[Inf,0,Inf,0]);
    f = fit((1:nframes)',medf,'exp2',opt);
    y = f.a * exp(f.b * x) + f.c * exp(f.d * x);
    s = y/y(1);
    % Rescale fluorescent intensity in each frame
    for ii = 1:nframes
        img(:,:,ii) = img(:,:,ii)/s(ii);
    end
    if max(img(:)) > 2^16 - 1
        img = img/max(img(:)) * (2^16 - 1);
    end
    % Get the corrected median fluorescence intensity of each frame
    medf_corrected = zeros(nframes,1);
    for ii = 1:nframes
        im = img(:,:,ii);
        medf_corrected(ii) = median(im(:));
    end
    % Plot median fluorescence intensity vs frame before and after
    % photobleaching correction
    if show_figs
        figure;
        subplot(1,2,1); plot(x,medf,'-b',x,y,'-r'); title('Raw data');
        xlabel('Frame'); ylabel('Median fluorescence intensity');
        yspan = ylim;
        subplot(1,2,2); plot(x,medf_corrected,'-b'); title('After correction');
        xlabel('Frame'); ylabel('Median fluorescence intensity');
        ylim(yspan);
    end
end


%% Create scale-normalised LoG filters for each scale
scales = sigmas.^2;
filters = cell(1,length(scales));
% tau = zeros(3,length(scales));
tt = 0;
for t = scales
    tt = tt+1;
    s = sqrt(t);
    % Create Laplacian of Gaussian filter
    LoG_filter = fspecial('log', ceil(s*4)*2+1, s);
    LoG_filter = -LoG_filter.*t;
    filters{tt} = LoG_filter;
%     for ii = 1:3    
%         tau(ii,tt) = sum(LoG_filter(:).^ii);
%     end
    clear LoG_filter;
end


%% Define VST and inverse VST

% Classical Anscombe VST
b1 = 2;
c1 = 3/8;
VST1 = @(X) b1*sign(X+c1).*sqrt(abs(X+c1));
VST1inv = @(Y) sign(Y).*(Y/b1).^2 - c1;

% % VST from https://hal.archives-ouvertes.fr/hal-00017925/document and 
% % https://hal.archives-ouvertes.fr/hal-00813572/document (not suitable
% % for LoG filter -- requires filter > 0).
% b2 = @(tt) 2*sqrt(abs(tau(1,tt))/tau(2,tt));
% c2 = @(tt) 7*tau(2,tt)/(8*tau(1,tt)) - tau(3,tt)/(2*tau(2,tt));
% VST2 = @(X,tt) b2(tt)*sign(X+c2(tt)).*sqrt(abs(X+c2(tt)));
% VST2inv = @(Y,tt) sign(Y).*(Y/b2(tt)).^2 - c2(tt);



%% Image processing
% Order of dimensions: im_log(:,:,scale,frame).

% Create 3x3 averaging filter
H = fspecial('average',3);

im_log = zeros([size(img,1), size(img,2), length(scales), nframes]);
im_log_norm = im_log;
Smin = zeros(size(img));
Smax = zeros(size(img));
mask = zeros(size(img));
ave_noise = zeros(1,1,nframes);
total_intensity = zeros(nframes,1);
for j = 1:nframes
    im = img(:,:,j);
    
    if use_AnscombeVST
        % Apply Anscombe VST before filtering
        im = VST1(im);
        %figure; imshow(im,[],'border','tight','initialmagnification',600);
    end
    
    if ~isinf(threshold_intensity) && ~isnan(threshold_intensity)
        % First define a binary mask by filtering im with a 3x3 averaging
        % filter and thresholding the filtered image.
        h = imfilter(im, H, 'replicate', 'conv');
        h = h > median(h(:)) + threshold_intensity*1.4826*mad(h(:),1);
    else
        h = ones(size(im));
    end
    
    if (~isinf(threshold_LoG) && ~isnan(threshold_LoG)) || ...
            (~isinf(threshold_LoG_diff) && ~isnan(threshold_LoG_diff))
        tt = 0;
        for t = scales
            tt = tt+1;

            % Retrieve scale-normalised Laplacian of Gaussian filter
            LoG_filter = filters{tt};

            % Convolve LoG filter with image
            im_log(:,:,tt,j) = imfilter(im, LoG_filter, 'replicate', 'conv');
            Y = im_log(:,:,tt,j);

            % Plot LoG filtered image of frame j at scale t(tt)
            %figure; imshow(Y,[],'border','tight','initialmagnification',600);
            %figure; histogram(Y(:));

            % Compute the median absolute deviation (MAD) of pixel values
            MADY = mad(Y(:),1); %median(abs(Y(:) - median(Y(:))));

            % Convert the MAD to something more resembling a standard deviation.
            % This is used instead of calculating the standard deviation
            % directly because the median absolute deviation ignores outliers
            % (and actual spots), giving a result after normalisation closer to the
            % standard deviation of bg/noise alone, with other image features ignored.
            % (See http://en.wikipedia.org/wiki/Median_absolute_deviation#Relation_to_standard_deviation
            % for the conversion factor)
            bg_std = 1.4826 * MADY;
            
            % Normalize filtered image by bg_std
            Y = Y / bg_std;
            im_log_norm(:,:,tt,j) = Y;
        end
      
        % Find the maximum positive and negative normalized responses to
        % the LoG filters over all scales
        [U,tU] = min(im_log_norm(:,:,:,j),[],3);
        [V,tV] = max(im_log_norm(:,:,:,j),[],3);
        Smin(:,:,j) = tU;
        Smax(:,:,j) = tV;
        
        % Replace positive min responses and negative max responses by 0
        U(U>0) = 0;  V(V<0) = 0;
        
        % Find the difference between the max positive and max negative
        % responses
        W = V + U;
        
        %% Apply thresholds to define mask
        % The LoG response threshold is applied to the difference between
        % max positive and negative responses over all scales.
        mask(:,:,j) = (V > threshold_LoG) & (W > threshold_LoG_diff) & h;
    else
        mask(:,:,j) = h;
    end
    
    % Get the average intensity of pixels outside the mask (average bg
    % intensity) for each frame.
    im = img(:,:,j);
    im_bg = im(~mask(:,:,j));
    if ~isempty(im_bg)
        ave_noise(:,:,j) = median(im_bg(:));
    else
        ave_noise(:,:,j) = 0;
    end
end

% Subtract average noise intensity from each frame
IMG = img - repmat(ave_noise,size(img,1),size(img,2),1);

% Retain only image pixels that appear in mask
IMG = IMG.*mask;

% Update mask to remove pixels that were below average bg intensity
mask = IMG > 0;

IMG = uint16(IMG);
mask = uint8(255*mask);

% Sum pixel intensities for each frame
for j = 1:nframes
    im = IMG(:,:,j);
    total_intensity(j) = sum(im(:));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Saving and plotting

[~,fname,~] = fileparts(img_fullpath);
if save_csv
    csvwrite(fullfile(save_folder,[fname,'_fluorescence_in_mask.csv']), total_intensity);
end

if save_tiffs
    %% Save TIFF stack
    imwrite(IMG(:,:,1),fullfile(save_folder,[fname,'_processed.tif']));
    imwrite(mask(:,:,1),fullfile(save_folder,[fname,'_mask.tif']));
    for j = 2:nframes
        imwrite(IMG(:,:,j),fullfile(save_folder,[fname,'_processed.tif']),'WriteMode','append');
        imwrite(mask(:,:,j),fullfile(save_folder,[fname,'_mask.tif']),'WriteMode','append');
    end
end


%% Plot fluorescence intensity vs frame number
if nframes > 1 && show_figs
    figure; plot(1:nframes, total_intensity, 'b', 'LineWidth', 2)
    set(gca,'FontSize',12)
    title('Total fluorescence in mask','FontSize',14)
    xlabel('Frame','FontSize',14)
    ylabel('Fluorescence intensity','FontSize',14)
end


%% Show some frames from mask and processed image
if show_figs
    if nframes < 2
        plotframes = 1;
    else
        plotframes = [1,nframes];
    end
    
    for j = plotframes
        figure; imshow(img(:,:,j),[],'border','tight','initialmagnification',600); %title('Raw image');
        figure; imshow(mask(:,:,j),[],'border','tight','initialmagnification',600); %title('Mask');
        figure; imshow(IMG(:,:,j),[],'border','tight','initialmagnification',600); %title('Processed and masked image');
    end
end

clear ii im im_bg t tt s j h LoG_filter Y U V tU tV W MADY bg_std b1 c1

if save_all
    save(fullfile(save_folder,[fname,'_processed.mat']), 'sigmas', ...
        'img_fullpath', 'img', 'IMG', 'mask', 'total_intensity', ...
        'scales', 'threshold_intensity', 'threshold_LoG', 'threshold_LoG', ...
        'filters', 'im_log', 'im_log_norm', 'ave_noise', 'Smin', 'Smax', ...
        'x', 'medf', 'save_folder', 'fname');
    if length(x) > 14 && correct_photobleach
        save(fullfile(save_folder,[fname,'_processed.mat']), ...
            'f', 'y', 'medf_corrected', '-append');
    end
end
    

%% END