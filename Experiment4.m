clc;clear all;close all

VidObject = VideoReader('Dataset\OneShopOneWait2cor.mpg');
numF = VidObject.NumberOfFrames;
imagePatch = cell(1,numF);
L = 170;
for d = 1:L
   imagePatch{d} = read(VidObject,d);
end

clf; figure(1); set(gca, 'Position', [0 0 1 1]);
imshow(imagePatch{170});

VIDEO_WIDTH = size(imagePatch{1},2);  VIDEO_HEIGHT = size(imagePatch{1},1);

disp('Initialize the bounding box by cropping the object');
[patch, bB] = imcrop();

s = 200:200:2000;
t = zeros(1,length(s));
for i = 1:length(s)
    clc;disp(i)
    t(i) = sum(PF_SMOG_time(s(i),patch,bB,VIDEO_WIDTH,VIDEO_HEIGHT,imagePatch{170}));
end
close all
plot(s,t,'--r','LineWidth',4)
axis([min(s) max(s) 0 6])
xlabel('Number of Particles')
ylabel('CPU Time (Second)')