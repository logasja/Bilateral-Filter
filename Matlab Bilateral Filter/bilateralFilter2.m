function B = bilateralFilter2(img, width, sigma)
%BINARYFILTER2 Summary of this function goes here
%   Detailed explanation goes here
    if ~exist('img', 'var') || isempty(img)
        error('The input image is either undefined or invalid.');
    end
    if ~isfloat(img) || ~sum([1,3] == size(img,3))...
            || min(img(:)) < 0 || max(img(:)) > 1
        error('Image A must be in double format of size NxMx1 or NxMx3 on interval [0,1].');
    end

    % Verify filter window size, if not valid then set to a default of 5
    if ~exist('width', 'var') || isempty(width) || numel(width) ~= 1 || width < 1
        width = 5;
    end
    % Ensure the width is a whole number
    width = ceil(width);

    % Verify sigma, if not valid then set to default [3 0.1]
    if ~exist('sigma', 'var') || isempty(sigma) || numel(sigma) ~= 2 || sigma(1) <= 0 || sigma(2) <= 0
        sigma = [3 0.1];
    end

    % Decide which filter to use, color or grayscale
    if size(img, 3) == 1
        B = bilatGray(img, width, sigma(1), sigma(2));
    else
        B = bilatColor(img, width, sigma(1), sigma(2));
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Implements bilateral filtering on grayscale images.                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function B = bilatGray(img, width, d, r)
    % Computation of Gaussian distance weights
    [X,Y] = meshgrid(-width:width,-width:width);
    G = exp(-(X.^2+Y.^2)/(2*d^2));
    
    % Apply the filter to the input image
    dim = size(img);
    B = zeros(dim);
    for i = 1:dim(1)
        for j = 1:dim(2)
            % Extract the local region
            iMin = max(i-width,1);
            iMax = min(i+width,dim(1));
            jMin = max(j-width,1);
            jMax = min(j+width,dim(2));
            I = img(iMin:iMax, jMin:jMax);
            
            % Compute the Gaussian intensity weights.
            H = exp(-(I - img(i,j)).^2/(2*r^2));
            
            % Calcuate filter response.
            F = H.*G((iMin:iMax)-i+width+1,(jMin:jMax)-j+width+1);
            B(i,j) = sum(F(:).*I(:))/sum(F(:));
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Implements bilateral filtering on color images.                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function B = bilatColor(img, width, d, r)
    
    img = applycform(img, makecform('srgb2lab'));

    % Computation of Gaussian distance weights
    [X,Y] = meshgrid(-width:width,-width:width);
    G = exp(-(X.^2+Y.^2)/(2*d^2));
    
    % Rescaling of range variance (using a max luminance of 100)
    r = 100*r;
    
    dim = size(img);
    B = zeros(dim);
    for i = 1:dim(1)
        for j = 1:dim(2)
            % Extract the local region
            iMin = max(i-width,1);
            iMax = min(i+width,dim(1));
            jMin = max(j-width,1);
            jMax = min(j+width,dim(2));
            I = img(iMin:iMax, jMin:jMax,:);
            
            % Compute range weights
            dL = I(:,:,1)-img(i,j,1);
            da = I(:,:,2)-img(i,j,2);
            db = I(:,:,3)-img(i,j,3);
            H = exp(-(dL.^2+da.^2+db.^2)/(2*r^2));
            
            % Calculate the response
            F = H.*G((iMin:iMax)-i+width+1, (jMin:jMax)-j+width+1);
            nF = sum(F(:));
            B(i,j,1) = sum(sum(F.*I(:,:,1)))/nF;
            B(i,j,2) = sum(sum(F.*I(:,:,2)))/nF;
            B(i,j,3) = sum(sum(F.*I(:,:,3)))/nF;            
        end
    end
    
    B = applycform(B, makecform('lab2srgb'));
end