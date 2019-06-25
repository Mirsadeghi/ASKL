function KL= Experiment1(cand,N)
% "cand" define condition to compre object model with candidate number
% 1,2,3.
% "N" define number of iteration

SI = 0.1;
alpha = struct;
Xo  = [mvnrnd(0,SI,400)  ; mvnrnd(5,SI,1600)];
alpha.Xo(1) = 400/2000;
alpha.Xo(2) = 1600/2000;

Xc1 = [mvnrnd(2,SI,400)  ; mvnrnd(7,SI,1600)];
alpha.Xc1(1) = 400/2000;
alpha.Xc1(2) = 1600/2000;

Xc2 = [mvnrnd(0,SI,1125) ; mvnrnd(5,SI,875) ];
alpha.Xc2(1) = 1125/2000;
alpha.Xc2(2) = 875/2000;

Xc3 = [mvnrnd(0,SI,750)  ; mvnrnd(5,SI,1250)];
alpha.Xc3(1) = 750/2000;
alpha.Xc3(2) = 1250/2000;

options = statset('Display','final');
objo  = gmdistribution.fit(Xo ,2,'Options',options);
objc1 = gmdistribution.fit(Xc1,2,'Options',options);
objc2 = gmdistribution.fit(Xc2,2,'Options',options);
objc3 = gmdistribution.fit(Xc3,2,'Options',options);

x = -1.5:0.07:8.5;
tmp  = (pdf(objo ,x'));
tmp = tmp/(sum(tmp));
idx = round(length(x)/2);
Po{1}  = tmp(1:idx); Po{2}  = tmp(idx+1:end);

tmp = (pdf(objc1,x'));
tmp = tmp/(sum(tmp));
Pc1{1} = tmp(1:idx); Pc1{2} = tmp(idx+1:end);

tmp = (pdf(objc2,x'));
tmp = tmp/(sum(tmp));
Pc2{1} = tmp(1:idx); Pc2{2} = tmp(idx+1:end);

tmp = (pdf(objc3,x'));
tmp = tmp/(sum(tmp));
Pc3{1} = tmp(1:idx); Pc3{2} = tmp(idx+1:end);

figure
hold on
plot([Po{1} ;Po{2} ],'r','Linewidth',2.5)
plot([Pc1{1};Pc1{2}],'g','Linewidth',2.5)
plot([Pc2{1};Pc2{2}],'b','Linewidth',2.5)
plot([Pc3{1};Pc3{2}],'k','Linewidth',2.5)
legend('Object Distribution','Candidate Distribution 1','Candidate Distribution2','Candidate Distribution3','Location','NorthWest');

K = 2;
clear tmp
tmp = zeros(1,K);
KL_tmp = zeros(1,K);

if cand == 1
    Pc = Pc1;
    alpha.Xc = alpha.Xc1;
elseif cand == 2
    Pc = Pc2;
    alpha.Xc = alpha.Xc2;
elseif cand == 3
    Pc = Pc3;
    alpha.Xc = alpha.Xc3;
end

KL = zeros(1,N);
for L_idx = 1:N
    for z = 1:K
        for k = 1:K
            tmp(k) = KL_Div(Po{z},Pc{k}) + (log(alpha.Xo(z)/alpha.Xc(k)));
        end
        KL_tmp(z) = (alpha.Xo(z) * min(tmp));
    end
    KL_fg = sum(KL_tmp);

    for z = 1:K
        for k = 1:K
            tmp(k) = KL_Div(Pc{z},Po{k}) + (log(alpha.Xc(z)/alpha.Xo(k)));
        end
        KL_tmp(z) = (alpha.Xc(z) * min(tmp));
    end
    KL_gf = sum(KL_tmp);
    co = 1.394473187771202;
    KL(L_idx) = exp(-0.5 * (KL_fg + KL_gf))/co;
end
% KL-Divergence Function
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