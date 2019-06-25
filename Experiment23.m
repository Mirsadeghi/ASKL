clc
close all

VidObject = VideoReader('Dataset\OneShopOneWait2cor.mpg');
l = 170;
imagePatch = read(VidObject, l);

% "ch" parameter define which change accured to images
%  ch = 1 for scale change
%  ch = 2 for x - translation
%  ch = 3 for y - translation

itr = 5;
fac = -10:5:10;
KL = struct;
M = zeros(1,length(fac));
t = struct;

target_rect = [107 72 31 71];

if rem(itr, 2) == 0
    error('itr must be an odd number')
end
x_trans_rect = zeros(itr, 4);
x_trans = 10;
% build x translation
for i = (1-itr)/2:(itr-1)/2
    x_trans_rect(i - (1-itr)/2 + 1, :) = [target_rect(1)+i*x_trans target_rect(2:end)];
end

y_trans_rect = zeros(itr, 4);
y_trans = 10;
% build y translation
for i = (1-itr)/2:(itr-1)/2
    y_trans_rect(i - (1-itr)/2 + 1, :) = [target_rect(1) target_rect(2)+i*y_trans target_rect(3:end)];
end

scale_rect = zeros(itr, 4);
scale_factor = 0.075;
% build scale
for i = (1-itr)/2:(itr-1)/2    
    scale_rect(i - (1-itr)/2 + 1, :) = round([target_rect(1)-i*target_rect(3)*scale_factor
                                        target_rect(2)-i*target_rect(4)*scale_factor
                                        target_rect(3)*(1 + 2*i*scale_factor)
                                        target_rect(4)*(1 + 2*i*scale_factor)]);
end
%%
figure(1); set(gca, 'Position', [0 0 1 1]);
ax(1) = subplot(2,3,1);
imshow(imagePatch)
title('X translation')
hold(ax(1), 'on')

ax(2) = subplot(2,3,2);
imshow(imagePatch)
hold(ax(2), 'on')
title('Y translation')

ax(3) = subplot(2,3,3);
imshow(imagePatch)
hold(ax(3), 'on')
title('Scale')

for i = 1:itr
    
    [patchf, bBf] = imcrop(imagePatch, x_trans_rect(i, :));
    bBf = round(bBf);
    r_x_trans = [bBf(1) bBf(2) 0 0 bBf(3)/bBf(4) bBf(4)];
    [KL.Xtrans{i}, t.Xtrans(i)] = ASKL(imagePatch,2,fac,patchf,bBf);
    
    axes(ax(1))%#ok
    if i == (itr-1)/2 + 1
        drawBox(r_x_trans, 'g');
    else
        drawBox(r_x_trans);
    end
    
    [patchf, bBf] = imcrop(imagePatch, y_trans_rect(i, :));
    bBf = round(bBf);
    r_y_trans = [bBf(1) bBf(2) 0 0 bBf(3)/bBf(4) bBf(4)];
    [KL.Ytrans{i}, t.Ytrans(i)] = ASKL(imagePatch,3,fac,patchf,bBf);
    
    axes(ax(2))%#ok
    if i == (itr-1)/2 + 1
        drawBox(r_y_trans, 'g');
    else
        drawBox(r_y_trans);
    end
    
    [patchf, bBf] = imcrop(imagePatch, scale_rect(i, :));
    bBf = round(bBf);
    r_scale = [bBf(1) bBf(2) 0 0 bBf(3)/bBf(4) bBf(4)];
    [KL.Scale{i},   t.Scale(i)] = ASKL(imagePatch,1,fac,patchf,bBf);
    
    axes(ax(3))%#ok
    if i == (itr-1)/2 + 1
        drawBox(r_scale, 'g');
    else
        drawBox(r_scale);
    end
    
end

for j = 1:itr
    M = M + KL.Xtrans{j};
end

subplot 234
M = M / itr;
M = M / max(M);
plot(fac,M,'r','LineWidth',3);
axis([min(fac) max(fac) 0 max(M)])
title('X translation')
grid on
for j = 1:itr
    M = M + KL.Ytrans{j};
end
subplot 235
M = M / itr;
M = M / max(M);
plot(fac,M,'r','LineWidth',3);
axis([min(fac) max(fac) 0 max(M)])
title('Y translation')
grid on
for j = 1:itr
    M = M + KL.Scale{j};
end
subplot 236
M = M / itr;
M = M / max(M);
plot(fac,M,'r','LineWidth',3);
axis([min(fac) max(fac) 0 max(M)])
title('Scale')
grid on