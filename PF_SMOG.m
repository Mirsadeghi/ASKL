%   June 2013
%
% Spatial-color Mixture of Gaussians (SMOG) appearance model
%
%   Author: Ehsan Mirsadeghi
%   All Right Reserved.
%                                                                     
%   This program is distributed WITHOUT ANY WARRANTY; without even the 
%   implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
%   PURPOSE.  See the GNU General Public License for more details.

function PF_SMOG(N)

K = 3;
dim = 3;
pdff = struct;
pdfg = struct;
tmp = zeros(1,K);
pdff.cov = cell(1,K);
pdfg.cov = cell(1,K);
pdff.alpha = zeros(1,K);

%% Particle Filtering Procedure

% Read the video sequence into videoSeq as a cell 
VidObject = VideoReader('Dataset\EnterExitCrossingPaths1cor.mpg');
numF = VidObject.NumberOfFrames;
videoSeq = cell(1,numF);
for d = 1:numF
   videoSeq{d} = read(VidObject,d);
end
% Define a path to save your results images and make sure it exists
writerObj = VideoWriter('Results\track.avi');
open(writerObj);

% Create a figure we will use to display the video sequence
clf; figure(1); set(gca, 'Position', [0 0 1 1]);
imshow(videoSeq{1});

% You might need to know the size of video frames 
% (i.e. to constrain the state so it doesn't wander outside of the video).
VIDEO_WIDTH = size(videoSeq{1},2);  VIDEO_HEIGHT = size(videoSeq{1},1);

% Because we don't have an automatic detection method to initialize the
% tracker, we will do it by hand. A GUI interface allows us to select a 
% bouding box from the first frame of the sequence.
disp('Initialize the bounding box by cropping the object');
[patch, bB] = imcrop();
bB = round(bB);
patch = rgb2gray(patch);
Imdf = ((double(patch)/255)*2)-1;
% Imdf = double(patch);
[rowsf, colsf] = size(Imdf);
Y_Idxf = kron(bB(1):bB(1)+bB(3),ones(rowsf,1));
X_Idxf = kron((bB(2):bB(2)+bB(4))',ones(1,colsf));
Y_Idxf = ((Y_Idxf/VIDEO_WIDTH)*2)-1;
X_Idxf = ((X_Idxf/VIDEO_HEIGHT)*2)-1;

stf = min(min(patch));
enf = max(max(patch));
Durf = (enf - stf) / K;

% Define GMM Model of Object
for ix = 1:K
    idxf = (patch <= stf+(Durf*(ix))) & (patch >= stf+(Durf*(ix-1))+1);   
    pdff.alpha(ix) = sum(sum(idxf)) / numel(idxf); 
    idataf =[(Imdf(idxf)) (X_Idxf(idxf)) (Y_Idxf(idxf))];  
    pdff.mu(ix,:) = mean(idataf);  
    pdff.var(ix,:) = var(idataf);
    for jx = 1:dim
        pdff.cov{ix}(jx,jx) = pdff.var(ix,jx);
    end
        pdff.prob{ix} = mvnpdf(idataf,pdff.mu(ix,:),pdff.cov{ix});
        pdff.prob{ix} = pdff.prob{ix} / sum(pdff.prob{ix});
        pdff.prob{ix} = pdff.alpha(ix) .* pdff.prob{ix};    
end

% % Fill the initial state vector with our hand initialization
x_init = [bB(1) bB(2) 0 0 bB(3)/bB(4) bB(4)];
% 
% % Draw a bouding box on the image showing our initial state estimate
drawBox(x_init);

% Particle Initialization
if nargin == 0
    N = 50;    % the number of samples (particles) to use
end

Alpha = max(VIDEO_WIDTH,VIDEO_HEIGHT)/30;
V = 1 / N;

% MAIN LOOP: we loop through the video sequence and estimate the state at
% each time step, t
T = length(videoSeq);
% Loc_Error = zeros(1,100);
for t = 1:T

    % load the current image and display it in the figure
    I = videoSeq{t}; figure(1); cla; imshow(I); hold on;
    
    Particle = [x_init(1) + Alpha * (rand(1,N)-rand(1,N));  x_init(2) + Alpha * (rand(1,N)-rand(1,N));
                 ones( 1 , N ) * x_init(3) ; ones( 1 , N ) * x_init(4) ;
                 ones( 1 , N ) * x_init(5) ; ones( 1 , N ) * x_init(6) ;
                 ones( 1 , N ) * V ] ;    

    Particle(1:end-1,:) = motionPredict(Particle(1:end-1,:));

    for id = 1:N
        Particle(end,id) = observationModel(Particle(1:end-1,id));
    end    

    % Normalize Weight of each Particle, So thaat SUm of them equal to One.     
    Particle(end,:) = Particle(end,:)./sum(Particle(end,:));
%     output = Particle(end,:);
    % Use Mode Of PDF as an answer   
    [ V , idx] = max(Particle(end,:));
    estimate_t = (Particle(1:end-1,idx));

    % Plot all particles (Box)
%     for kk = 1 : N
%         drawBox(Particle(1:end-1,kk), [0 0 1]);
%     end

    % Plot all particles(Circles)
%         scatter(Particle(1,:)+(Particle(5,:).*Particle(6,:)/2),Particle(2,:)+(Particle(6,:)/2))

    drawBox(estimate_t, [1 0 0]);    
%     [GT_x GT_y] = getpts;
%     Loc_Error(t) = abs(GT_x-(estimate_t(1)+(estimate_t(6)*estimate_t(5)/2))) + abs(GT_y-(estimate_t(2)+(estimate_t(6)/2))); 
    x_init = estimate_t;    
    % allow the figure to refresh so we can see our results
    pause(0.001); refresh;
    imtemp = getframe;
    writeVideo(writerObj,imtemp.cdata);
end
close(writerObj);
% ========================== SUPPLEMENTARY FUNCTIONS ======================

%% Draws a bounding box given a state estimate.  Color of the bounding box
% can be given as an optional second argument: drawbox(x_t, [0 1 0]) will
% give a green bounding box.
function drawBox(x, varargin)
    r = x(1);
    c = x(2);
    w = x(6)*x(5);
    h = x(6);

    x1 = r;
    x2 = r + w;
    y1 = c;
    y2 = c + h;

    if nargin == 1
        line([x1 x1 x2 x2 x1], [y1 y2 y2 y1 y1]);
    else
        line([x1 x1 x2 x2 x1], [y1 y2 y2 y1 y1], 'Color', varargin{1}, 'LineWidth', 1.5);
    end
end

%% OBSERVATION MODEL.  Computes the likelihood that data observed in the
% image I (z_t) supports the state estimated by x_t.  I computed the
% likelihood by comparing a color histogram extracted from the bounding box
% defined by x_to to a known color model which I extracted beforehand. I
% modeled p(z_t|x_t) \propto exp(-lambda*dist(h, colormodel)) where dist
% is the KL divergence of two histograms, h is a color histogram
% corresponding to x_t, colorModel is a known color model, and lambda is a
% hyperparameter used to adjust the pickiness of the likelihood function.
% You may, however, use any likelihood model you prefer.
function z_t = observationModel(x_t)
KL_tmp = zeros(1,K);
% You might find image patch corresponding to the x_t bounding box useful
r = round(x_t(2)); c = round(x_t(1));
w = round(x_t(5)*x_t(6)); h = round(x_t(6));
r2 = min(VIDEO_HEIGHT, r+h+1);
c2 = min(VIDEO_WIDTH, c+w+1);
imagePatch = rgb2gray(I(r:r2, c:c2,:));

Imdg = ((double(imagePatch)/255)*2)-1;
% Imdg = double(imagePatch);
[rowsg, colsg] = size(Imdg);
Y_Idxg = kron(c:c2,ones(rowsg,1));
X_Idxg = kron((r:r2)',ones(1,colsg));
Y_Idxg = ((Y_Idxg/colsg)*2)-1;
X_Idxg = ((X_Idxg/rowsg)*2)-1;

stg = min(min(imagePatch));
eng = max(max(imagePatch));
Durg = (eng - stg) / K;

for i = 1:K
    idxg = (imagePatch <= stg+(Durg*(i))) & (imagePatch >= stg+(Durg*(i-1))+1);
    pdfg.beta(i) = sum(sum(idxg)) / numel(idxg);
    idatag =[(Imdg(idxg)) (X_Idxg(idxg)) (Y_Idxg(idxg))];
    pdfg.mu(i,:) = mean(idatag);
    pdfg.var(i,:) = var(idatag);
    for j = 1:dim
        pdfg.cov{i}(j,j) = pdfg.var(i,j);
    end
        pdfg.prob{i} = mvnpdf(idatag,pdfg.mu(i,:),pdfg.cov{i});
        pdfg.prob{i} = pdfg.prob{i} / sum(pdfg.prob{i});
        pdfg.prob{i} = pdff.alpha(i) .* pdfg.prob{i};
end

    for z = 1:K
        for k = 1:K
            tmp(k) = KLdivergence(pdff.prob{z},pdfg.prob{k}) + (log(pdff.alpha(z)/pdfg.beta(k)));
        end
        KL_tmp(z) = (pdff.alpha(z) * min(tmp));
    end
    KL_fg = sum(KL_tmp);
    for z = 1:K
        for k = 1:K
            tmp(k) = KLdivergence(pdfg.prob{z},pdff.prob{k}) + (log(pdfg.beta(z)/pdff.alpha(k)));
        end
        KL_tmp(z) = (pdfg.beta(z) * min(tmp));
    end
    KL_gf = sum(KL_tmp);
    z_t = exp(-0.5 * (KL_fg + KL_gf));
end

%% MOTION PREDICTION MODEL. Given a past state vector x_t1, predicts a new 
% state vector x_t according to the motion model.
function x_t = motionPredict(x_t1)

% You might find that constraining certain elements of the state such as
% the bounding box height or aspect ratio can improve your tracking
MIN_h = 10;  MAX_h = 70;
MIN_a = .06; MAX_a = 4.75;
MIN_x = 1;   MAX_x = VIDEO_WIDTH;
MIN_y = 1;   MAX_y = VIDEO_HEIGHT;

% Motion_model
F = [1 0 1 0 0 0;
     0 1 0 1 0 0;
     0 0 1 0 0 0;
     0 0 0 1 0 0;
     0 0 0 0 1 0;
     0 0 0 0 0 1];

Beta = 1;
Q = Beta * (randn(6,N));
Q(3:6,:) = Q(3:6,:) * 0.005;
x_t = (F * x_t1) + Q;

% Threshold for controlling Aspect Ratio
X1_max = (MAX_a - x_t(5,:) < 0); S1_max = sum(X1_max);
X1_min = (x_t(5,:) - MIN_a < 0); S1_min = sum(X1_min);
if S1_max > 0 ; x_t(5,X1_max) = MAX_a ; end
if S1_min > 0 ; x_t(5,S1_min) = MIN_a ; end

% Threshold for controlling Height
X2_max = (MAX_h - x_t(6,:) < 0); S2_max = sum(X2_max);
X2_min = (x_t(6,:) - MIN_h < 0); S2_min = sum(X2_min);
if S2_max > 0 ; x_t(6,X2_max) = MAX_h ; end
if S2_min > 0 ; x_t(6,S2_min) = MIN_h ; end

% Threshold for controlling X Location
X3_max = (MAX_x - x_t(1,:) < 0); S3_max = sum(X3_max);
X3_min = (x_t(1,:) - MIN_x < 0); S3_min = sum(X3_min);
if S3_max > 0 ; x_t(1,X3_max) = MAX_x ; end
if S3_min > 0 ; x_t(1,S3_min) = MIN_x ; end

% Threshold for controlling Y Location
X_max4 = (MAX_y - x_t(2,:) < 0); S4_max = sum(X_max4);
X_min4 = (x_t(2,:) - MIN_y < 0); S4_min = sum(X_min4);
if S4_max > 0 ; x_t(2,X_max4) = MAX_y ; end
if S4_min > 0 ; x_t(2,S4_min) = MIN_y ; end
end

%% KL DIVERGENCE Computes the KL divergence (a distance measure) for two 
% 1-D histograms. 
function kl = KLdivergence(P1,P2)

    if length(P1) == length(P2)
        eta = eps * ones(size(P1));
        H1n = P1+eta;
        H2n = P2+eta;
        temp = H1n.*log(H1n./H2n);
        temp(isnan(temp)) = 0;
        kl = sum(temp);
    else
        tx = max(length(P1),length(P2));
        P1 = padarray(P1,tx-length(P1),'post');
        P2 = padarray(P2,tx-length(P2),'post');
        
        eta = eps * ones(size(P1));
        H1n = P1+eta;
        H2n = P2+eta;
        temp = H1n.*log(H1n./H2n);
        temp(isnan(temp)) = 0;
        kl = sum(temp);
    end
end

end
