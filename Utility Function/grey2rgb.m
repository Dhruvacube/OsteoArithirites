function [preProcessImage] = grey2rgb(Image)
%Gives a grayscale image an extra dimension
%in order to use color within it
filler = zeros(size(Image),'uint8');
preProcessImage = cat(3,Image,filler,filler);
end