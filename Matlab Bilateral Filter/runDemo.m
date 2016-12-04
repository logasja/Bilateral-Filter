function [] = runDemo( )
%RUNDEMO Summary of this function goes here
%   Detailed explanation goes here

    grayImage = imread('bw.png');
    grayImage = rgb2gray(grayImage);
    grayImage = imnoise(grayImage, 'gaussian');
    % Get the dimensions of the image.
    % numberOfColorBands should be = 1.
    % Display the original gray scale image.
    figure,subplot(2, 4, 1);
    imshow(grayImage, []);
    title('Noisy Image', 'FontSize', 16);
    subplot(1,4,2);
    surf(double(grayImage));
    title('No Filter', 'FontSize', 16);
    subplot(1,4,3);
    B = imfilter(grayImage,fspecial('gaussian',[23 23],100),'same');
    surf(double(B));
    title('Gaussian Filter', 'FontSize', 16);
    subplot(1,4,4);
    A = im2double(grayImage);
    B = bilateralFilter2(A, 5, [100 .2]);
    surf(double(B));
    title('Bilateral Filter', 'FontSize', 16);
    subplot(2,4,5);
    imshow(B);
    title('Bilateral Filter Result', 'FontSize', 16);
    
    figure, subplot(1,3,1)
    A = im2double(imread('../Test Images/test-medium.jpg'));
    imshow(A);
    title('Original Image', 'FontSize', 16);
    subplot(1,3,2)
    B = imfilter(A,fspecial('gaussian',[5 5],2),'same');
    imshow(B);
    title('Gaussian Filter', 'FontSize', 16);
    %profile on
    B = bilateralFilter2(A);
    %profile off
    %profile viewer
    subplot(1,3,3)
    imshow(B);
    title('Bilateral Filter', 'FontSize', 16);
    

    figure, subplot(1,3,1)
    A = im2double(imread('../Test Images/test-large.jpg'));
    imshow(A);
    title('Original Image', 'FontSize', 16);
    subplot(1,3,2)
    B = imfilter(A,fspecial('gaussian',[5 5],10),'same');
    imshow(B);
    title('Gaussian Filter', 'FontSize', 16);
    %profile on
    B = bilateralFilter2(A);
    %profile off
    %profile viewer
    subplot(1,3,3)
    imshow(B);
    title('Bilateral Filter', 'FontSize', 16);
    
    figure, subplot(1,3,1)
    A = im2double(imread('../Test Images/test-xlarge.jpg'));
    imshow(A);
    title('Original Image', 'FontSize', 16);
    subplot(1,3,2)
    B = imfilter(A,fspecial('gaussian',[10 10],4),'same');
    imshow(B);
    title('Gaussian Filter', 'FontSize', 16);
    %profile on
    B = bilateralFilter2(A, 10);
    %profile off
    %profile viewer
    subplot(1,3,3)
    imshow(B);
    title('Bilateral Filter', 'FontSize', 16);
end

