%% SMOG
function [KL, time_cost] = ASKL(imagePatch,ch,Cfactor,patchf,bBf)

% Initialization
K = 4;
dim = 3;
N = length(Cfactor);
pdff = struct;
pdfg = struct;
tmp = zeros(1,K);
KL = zeros(1,N);
KL_fg = zeros(1,N);
KL_gf = zeros(1,N);
KL_tmp = zeros(1,K);
pdff.cov = cell(1,K);
pdfg.cov = cell(1,K);
pdff.alpha = zeros(1,K);
Vid_W = size(imagePatch,2);
Vid_H = size(imagePatch,1);


bBf = round(bBf);
patchf = rgb2gray(patchf);
% Imf = ((double(patchf)/255)*2)-1;
Imf = double(patchf);
[rowsf, colsf] = size(Imf);
Y_Idxf = kron(bBf(1):((bBf(1)+bBf(3))),ones(rowsf,1));
X_Idxf = kron((bBf(2):(bBf(2)+bBf(4)))',ones(1,colsf));
Y_Idxf = ((Y_Idxf/Vid_W)*2)-1;
X_Idxf = ((X_Idxf/Vid_H)*2)-1;

stf = min(min(patchf));
enf = max(max(patchf));
Durf = (enf - stf) / K;

for Loop_idx = 1:N
    if ch == 1
        % Scale change
        bBg = [bBf(1)+Cfactor(Loop_idx) bBf(2)+Cfactor(Loop_idx) bBf(3)-(2*Cfactor(Loop_idx)) bBf(4)-(2*Cfactor(Loop_idx))];
    elseif ch == 2
        % X translation
        bBg = [bBf(1)+Cfactor(Loop_idx) bBf(2) bBf(3) bBf(4)];
    elseif ch == 3
        % Y translation
        bBg = [bBf(1) bBf(2)+Cfactor(Loop_idx) bBf(3) bBf(4)];
    end
% Choose new patch according to new coordinates come from variable "bBg"
patchg = imagePatch(bBg(2):bBg(2)+bBg(4)-1,bBg(1):bBg(1)+bBg(3)-1,:);

% Draw a bouding box on the image showing our initial state estimate
% pts = ([bBg(1) bBg(2) 0 0 bBg(3)/bBg(4) bBg(4)]);
% drawBox(pts,[0 1 1]);

patchg = rgb2gray(patchg);
% Normalized gray - value of patch into interval [-1 1]
% Img = ((double(patchg)/255)*2)-1;
Img = double(patchg);
[rowsg, colsg] = size(Img);
% Build index matrix for speed up coordinate recall of each pixel
Y_Idxg = kron(bBg(1):((bBg(1)+bBg(3))),ones(rowsg,1));
X_Idxg = kron((bBg(2):(bBg(2)+bBg(4)))',ones(1,colsg));
% Normalized index matrix.
Y_Idxg = ((Y_Idxg/Vid_W)*2)-1;
X_Idxg = ((X_Idxg/Vid_H)*2)-1;

% Specify K component limitation for new patch
stg = min(min(patchg));
eng = max(max(patchg));
Durg = (eng - stg) / K;

% Sub-Main loop: in this loop we look for computing statistical parameters
% of GMM of new candidate patch and compare GMM of candidate patch with
% object.
tic
    for i = 1:K
        % Stage 1: for input candidate patch, Gray space evenly divided 
        % into K subspace.

        idxf = (patchf <= stf+(Durf*(i))) & (patchf >= stf+(Durf*(i-1))+1);
        idxg = (patchg <= stg+(Durg*(i))) & (patchg >= stg+(Durg*(i-1))+1);
        
        % Stage 2: compute weight of each K component of GMM according to 
        % number of pixel accured in each gray subspace.         
        pdff.alpha(i) = sum(sum(idxf)) / numel(idxf);
        pdfg.beta(i) = sum(sum(idxg)) / numel(idxg);

        % Stage3: build feature matrix contain Color-Spatial features          
        idataf =[(Imf(idxf)) (X_Idxf(idxf)) (Y_Idxf(idxf))];
        idatag =[(Img(idxg)) (X_Idxg(idxg)) (Y_Idxg(idxg))];

        % Stage4: compute statistical parameters of data(Mean,Covariance)         
        % Compute mean         
        pdff.mu(i,:) = mean(idataf);
        pdfg.mu(i,:) = mean(idatag);
        % Compute variance         
        pdff.var(i,:) = var(idataf);
        pdfg.var(i,:) = var(idatag);
        % Build Covariance matrix using variance     
        for j = 1:dim
            pdff.cov{i}(j,j) = pdff.var(i,j);
            pdfg.cov{i}(j,j) = pdfg.var(i,j);
            if j == dim
               pdff.invcov{i} = inv(pdff.cov{i});
               pdfg.invcov{i} = inv(pdfg.cov{i});
            end
        end

        % Compute probability of each entry of feature matrix         
        pdff.prob{i} = mvnpdf(idataf,pdff.mu(i,:),pdff.cov{i});
        pdff.prob{i} = pdff.prob{i} / sum(pdff.prob{i});
        pdff.prob{i} = pdff.alpha(i) .* pdff.prob{i};    
        
        pdfg.prob{i} = mvnpdf(idatag,pdfg.mu(i,:),pdfg.cov{i});
        pdfg.prob{i} = pdfg.prob{i} / sum(pdfg.prob{i});
        pdfg.prob{i} = pdff.alpha(i) .* pdfg.prob{i};

    end
        time_cost = toc;
    for z = 1:K
        for k = 1:K
            tmp(k) = KL_Div(pdff.prob{z},pdfg.prob{k}) + (log(pdff.alpha(z)/pdfg.beta(k)));
        end
        KL_tmp(z) = (pdff.alpha(z) * min(tmp));
    end
    KL_fg(Loop_idx) = sum(KL_tmp);

    for z = 1:K
        for k = 1:K
            tmp(k) = KL_Div(pdfg.prob{z},pdff.prob{k}) + (log(pdfg.beta(z)/pdff.alpha(k)));
        end
        KL_tmp(z) = (pdfg.beta(z) * min(tmp));
    end
    KL_gf(Loop_idx) = sum(KL_tmp);
    KL(Loop_idx) = exp(-0.5 * (KL_fg(Loop_idx) + KL_gf(Loop_idx)));
end

% [~,idx] = max(KL);
%     if ch == 1
%         % Scale change
%         bBg = [bBf(1)+Cfactor(idx) bBf(2)+Cfactor(idx) bBf(3)-(2*Cfactor(idx)) bBf(4)-(2*Cfactor(idx))];
%     elseif ch ==2
%         % X translation
%         bBg = [bBf(1)+Cfactor(idx) bBf(2) bBf(3) bBf(4)];
%     elseif ch == 3
%         % Y translation
%         bBg = [bBf(1) bBf(2)+Cfactor(idx) bBf(3) bBf(4)];
%     end
% pts = ([bBg(1) bBg(2) 0 0 bBg(3)/bBg(4) bBg(4)]);
% % Draw a bouding box on the image showing our initial state estimate
% drawBox(pts,[1 0 0]);

%% KL-Divergence Function
% --------------------------
function kl = KL_Div(P1,P2)
    if length(P1) == length(P2)
        eta = eps * ones(size(P1));
        H1n = P1+eta;
        H2n = P2+eta;
        temp = H1n.*log(H1n./H2n);
        temp(isnan(temp)) = 0;
        kl = sum(temp);
    else
        t = max(length(P1),length(P2));
        P1 = padarray(P1,t-length(P1),'post');
        P2 = padarray(P2,t-length(P2),'post');        

        eta = eps * ones(size(P1));
        H1n = P1+eta;
        H2n = P2+eta;
        temp = H1n.*log(H1n./H2n);
        temp(isnan(temp)) = 0;
        kl = sum(temp);
    end
end

end