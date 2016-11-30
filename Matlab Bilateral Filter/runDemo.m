function [] = runDemo( )
%RUNDEMO Summary of this function goes here
%   Detailed explanation goes here
    figure, subplot(1,2,1)
    A = im2double(imread('../Test Images/test-small.jpg'));
    imshow(A);
    profile on
    B = bilateralFilter2(A);
    profile off
    profile viewer
    subplot(1,2,2)
    imshow(B);

    figure, subplot(1,2,1)
    A = im2double(imread('../Test Images/test-medium.jpg'));
    imshow(A);
    profile on
    B = bilateralFilter2(A);
    profile off
    profile viewer
    subplot(1,2,2)
    imshow(B);

    figure, subplot(1,2,1)
    A = im2double(imread('../Test Images/test-large.jpg'));
    imshow(A);
    profile on
    B = bilateralFilter2(A);
    profile off
    profile viewer
    subplot(1,2,2)
    imshow(B);
    
    figure, subplot(1,2,1)
    A = im2double(imread('../Test Images/test-xlarge.jpg'));
    imshow(A);
    profile on
    B = bilateralFilter2(A);
    profile off
    profile viewer
    subplot(1,2,2)
    imshow(B);
end

