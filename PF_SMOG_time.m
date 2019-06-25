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

function tt = PF_SMOG_time(N,patch,bB,VIDEO_WIDTH,VIDEO_HEIGHT,I)

% ImModel = rgb2gray(imread('D:\University\MSC Thesis\Dataset\Tracking\Car\sample.jpg'));
% ImInput = rgb2gray(imread('D:\University\MSC Thesis\Dataset\Tracking\Car\0800.jpg'));

K = 3;
dim = 3;
pdff = struct;
pdfg = struct;
pdff.cov = cell(1,K);
pdfg.cov = cell(1,K);
pdff.alpha = zeros(1,K);
tt = zeros(1,N);
%% Particle Filtering Procedure

bB = round(bB);
patch = rgb2gray(patch);
Imdf = ((double(patch)/255)*2)-1;
[rowsf colsf] = size(Imdf);
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

Alpha = max(VIDEO_WIDTH,VIDEO_HEIGHT)/30;
V = 1 / N;

% MAIN LOOP: we loop through the video sequence and estimate the state at
% each time step, t
for t = 1
    Particle = [x_init(1) + Alpha * (rand(1,N)-rand(1,N));  x_init(2) + Alpha * (rand(1,N)-rand(1,N));
                 ones( 1 , N ) * x_init(3) ; ones( 1 , N ) * x_init(4) ;
                 ones( 1 , N ) * x_init(5) ; ones( 1 , N ) * x_init(6) ;
                 ones( 1 , N ) * V ] ;    

    Particle(1:end-1,:) = motionPredict(Particle(1:end-1,:));

    for id = 1:N
        tt(id) = observationModel(Particle(1:end-1,id));
    end       
end

% ========================== SUPPLEMENTARY FUNCTIONS ======================


%% OBSERVATION MODEL.
function t_temp = observationModel(x_t)

% You might find image patch corresponding to the x_t bounding box useful
r = round(x_t(2)); c = round(x_t(1));
w = round(x_t(5)*x_t(6)); h = round(x_t(6));
r2 = min(VIDEO_HEIGHT, r+h+1);
c2 = min(VIDEO_WIDTH, c+w+1);
imagePatch = rgb2gray(I(r:r2, c:c2,:));

Imdg = ((double(imagePatch)/255)*2)-1;
% Imdg = double(imagePatch);
[rowsg colsg] = size(Imdg);
Y_Idxg = kron(c:c2,ones(rowsg,1));
X_Idxg = kron((r:r2)',ones(1,colsg));
Y_Idxg = ((Y_Idxg/colsg)*2)-1;
X_Idxg = ((X_Idxg/rowsg)*2)-1;

stg = min(min(imagePatch));
eng = max(max(imagePatch));
Durg = (eng - stg) / K;
    tic
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
        pdfg.prob{i} = pdff.alpha(i) .*(pdfg.prob{i} / sum(pdfg.prob{i}));
end
    t_temp = toc;
end

%% MOTION PREDICTION MODEL.
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

end
