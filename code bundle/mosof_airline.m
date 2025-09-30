function out = mosof_airline_feasible(excelFile, attrFile)
% MOSOF_AIRLINE_FEASIBLE
% Fast MOSOF optimisation with FEASIBLE initialisation + REPAIR.
% - Derives NDCI (SP,S,U) per subsystem from DataB.xlsx
% - Merges cost/MTBF/weight from sensor_attributes.xlsx (optional)
% - Multi-objective GA with feasible creation + repair (no degenerate Pareto)
% - Publication-grade figures + summary CSV/XLSX
%
% Usage:
%   out = mosof_airline_feasible('DataB.xlsx','sensor_attributes.xlsx');

if nargin<1||isempty(excelFile), excelFile='DataB.xlsx'; end
if nargin<2||isempty(attrFile),  attrFile ='sensor_attributes.xlsx'; end
rng(42); clc;

%% ---------------- Airline config (tweak as needed) ----------------
cfg.alpha   = 0.6;  cfg.beta = 0.4;                     % NDCI vs U
cfg.wSys    = struct('Engine',0.40,'Fuel',0.15,'Elec',0.15,'ECS',0.30);
cfg.Budget  = 8e4;                                      % hard (trim-to-budget)
cfg.Wmax    = [];                                       % [] = ignore weight
cfg.RelTarget = 1.80e5;                                 % soft target (not a floor)

cfg.kMin = struct('Engine',4,'Fuel',2,'Elec',2,'ECS',3);
cfg.kMax = struct('Engine',10,'Fuel',4,'Elec',5,'ECS',6);

cfg.CovTopK = struct('Engine',5,'Fuel',2,'Elec',2,'ECS',4);  % define "top" set per system
cfg.CovMin  = struct('Engine',1,'Fuel',1,'Elec',1,'ECS',1);  % at least this many from "top"

cfg.PopSize = 180;  cfg.MaxGen = 220;

%% ---------------- Groups & labels ----------------
sensorGroups.Engine = { ...
'W_S1','W_S2','W_S21','W_S24','W_S3','W_S4','W_S45','W_S5','W_S7', ...
'h_S1','h_S2','h_S21','h_S24','h_S3','h_S4','h_S45','h_S5','h_S7','h_S17', ...
'Tt_S1','Tt_S2','Tt_S21','Tt_S24','Tt_S3','Tt_S4','Tt_S45','Tt_S5','Tt_S7','Tt_S17', ...
'Pt_S1','Pt_S21','Pt_S3','Pt_S4','Pt_S45','Pt_S5','FanTrq','LPCTrq','HPCTrq','HPTTrq','LPTTrq','Thrust', ...
'h_Fan','Tt_Fan','Pt_Fan','Wf','FS_Motor_Torque', ...
'W_BfBleed','W_AfBleed','W_BfCDP','W_AfCDP','h_BfBleed','h_AfBleed','h_BfCDP','h_AfCDP', ...
'Tt_BfBleed','Tt_AfBleed','Tt_BfCDP','Tt_AfCDP','Pt_BfBleed','Pt_AfBleed','Pt_BfCDP','Pt_AfCDP','Tsfc'};
sensorGroups.Fuel = {'P1','P2','P3','P4','P5','P6','P7','Flow'};
sensorGroups.Elec = {'Power_GenOnly','Power_AC_Gen_load','Power_ACLamp','Power_TRU', ...
'AC_Fluoro_I','AC_Fluoro_V','AC_Instru_V','Eng_BleedValve_I','Eng_BleedValve_V'};
sensorGroups.ECS  = {'PVOT','ThiPHX','ThoPHX','TiC','ToC','ThiSHX','ThoSHX','ThiRHX','ThoRHX', ...
'ThiCHX','ThoCHX','TciRHX','TcoRHX','TiT','ToT'};

FM.Elec  = compose("FM%d",1:7);
FM.Fuel  = compose("FM%d",8:13);
FM.Engine= compose("FM%d",14:20);
FM.ECS   = compose("FM%d",21:25);

%% ---------------- Load & prune constants (quick) ----------------
T = readtable(excelFile);
T.Properties.VariableNames = matlab.lang.makeValidName(T.Properties.VariableNames);
fprintf('--- Data Loaded (%s) ---\n', excelFile);
[Tclean, sensorGroups] = preprocessConstantsOnly(T, sensorGroups);

%% ---------------- MOSOF features per subsystem ----------------
Snames = {'Engine','Fuel','Elec','ECS'};
All = initAll();
A = []; if isfile(attrFile), A = readtable(attrFile); A.Sensor = string(A.Sensor); end
startIdx=1;

for s=1:numel(Snames)
    sub = Snames{s}; cand = sensorGroups.(sub); if isempty(cand), continue; end
    switch sub
        case 'Engine', subFM = FM.Engine;
        case 'Fuel',   subFM = FM.Fuel;
        case 'Elec',   subFM = FM.Elec;
        case 'ECS',    subFM = FM.ECS;
    end
    Tsub = Tclean(ismember(Tclean.FaultMode, subFM) | strcmp(Tclean.FaultMode,'Healthy'), :);
    [ndci,~,~,U,~,~] = calculateNDCI(Tsub, cand);
    ndci = rescaleSafe(ndci);  U = rescaleSafe(U);

    [Cost,MTBF,Weight] = attachAttributes(cand, A);

    K = min(getfield(cfg.CovTopK, sub), numel(cand)); %#ok<GFLD>
    proxy = cfg.alpha*ndci + cfg.beta*U; [~,ord]=sort(proxy,'descend');
    coverMask = false(1,numel(cand)); coverMask(ord(1:K)) = true;

    All.NDCI=[All.NDCI, ndci]; All.U=[All.U, U];
    All.Cost=[All.Cost, Cost]; All.MTBF=[All.MTBF, MTBF]; All.Weight=[All.Weight, Weight];
    All.names=[All.names, cand]; 
    All.sysNames=[All.sysNames, repmat({sub},1,numel(cand))];
    All.sysIdx=[All.sysIdx; startIdx, startIdx+numel(cand)-1];
    All.topMask{end+1}=coverMask;
    All.kMin=[All.kMin, getfield(cfg.kMin, sub)]; 
    All.kMax=[All.kMax, getfield(cfg.kMax, sub)];
    All.covMin=[All.covMin, getfield(cfg.CovMin, sub)];
    startIdx = startIdx + numel(cand);
end

% system weights aligned with All.sysIdx
wSys = zeros(1,size(All.sysIdx,1));
for j=1:size(All.sysIdx,1)
    sysName = All.sysNames{All.sysIdx(j,1)};
    wSys(j) = getfield(cfg.wSys, sysName);
end

% quick feasibility probe (upper bound on achievable RelHM with kMin)
bestHM = probe_bestcase_harmonic_mtbf(All);
fprintf('Best-case harmonic MTBF with kMin per system ≈ %.0f h; RelTarget = %.0f h.\n', bestHM, cfg.RelTarget);

%% ---------------- GA: feasible creation + repair ----------------
nvars = numel(All.NDCI);
lb = zeros(1,nvars); ub = ones(1,nvars);

createFcn = @(GenomeLength,FitnessFcn,options) createPopFeasible(GenomeLength, options, All, cfg);

opts = optimoptions('gamultiobj', ...
    'PopulationSize', cfg.PopSize, ...
    'MaxGenerations', cfg.MaxGen, ...
    'CreationFcn', createFcn, ...
    'CrossoverFcn', @crossoverscattered, ...
    'MutationFcn', @mutationadaptfeasible, ... % silence warning; we still repair
    'SelectionFcn', @selectiontournament, ...
    'Display','iter','FunctionTolerance',1e-6,'UseVectorized',false);

fprintf('\nMulti-objective optimization:\n%d Variables\n\n', nvars);
fprintf('Options:\nCreationFcn:       %s\nCrossoverFcn:      %s\nSelectionFcn:      %s\nMutationFcn:       %s\n\n', ...
    func2str(opts.CreationFcn), func2str(opts.CrossoverFcn), func2str(opts.SelectionFcn), func2str(opts.MutationFcn));

fit = @(x) obj_with_repair(x, All, cfg, wSys);   % always feasible after repair
[X,F] = gamultiobj(fit, nvars, [],[],[],[], lb, ub, [], opts);
S = X>0.5;

% Decode + knee
sol = struct('Sensors',{});
for i=1:size(S,1)
    z = repair_to_feasible(S(i,:), All, cfg);    % store repaired final picks
    sol(i).Sensors = All.names(z);
    sol(i).kTotal = sum(z);
    sol(i).SysCounts = zeros(1,size(All.sysIdx,1));
    for j=1:size(All.sysIdx,1), rg=All.sysIdx(j,1):All.sysIdx(j,2); sol(i).SysCounts(j)=sum(z(rg)); end
end
knee = kneeSelect(F);
out = struct('ParetoX',X,'ParetoSel',S,'F',F,'Solutions',{sol},'All',All,'cfg',cfg,'kneeIndex',knee);
save('mosof_airline_feasible_out.mat','out','cfg');

% Plots + exports
plotPareto(out);
plotKneeBreakdown(out);
plotCountsWindows(out);
paretoQuality(out);
exportTables(out);

fprintf('\nSaved: mosof_airline_feasible_out.mat, pareto_feasible_*.png, pareto_parallel.png,\n');
fprintf('       knee_subsystem_breakdown.png, knee_lollipop_ndci.png, counts_vs_windows.png,\n');
fprintf('       anchors_summary.csv, pareto_metrics.csv, pareto_summary_feasible.xlsx, knee_suite_feasible.xlsx\n\n');
end

%% ============================== Helpers ==============================

function All=initAll()
All=struct('NDCI',[],'U',[],'Cost',[],'MTBF',[],'Weight',[], ...
           'names',{{}},'sysNames',{{}},'sysIdx',[],'topMask',{{}}, ...
           'kMin',[],'kMax',[],'covMin',[]);
end

function [Tclean, groups] = preprocessConstantsOnly(T, groups)
existS = unique([groups.Engine, groups.Fuel, groups.Elec, groups.ECS], 'stable');
existS = intersect(existS, T.Properties.VariableNames, 'stable');
v = var(T{:, existS}, 0, 1,'omitnan'); const = existS(v<1e-6);
if ~isempty(const), fprintf('Remove const: %s\n', strjoin(const,',')); end
Tclean = T; Tclean(:,const)=[];
fn = fieldnames(groups);
for i=1:numel(fn), groups.(fn{i}) = intersect(groups.(fn{i}), Tclean.Properties.VariableNames, 'stable'); end
end

function [ndci,SPn,Sn,Un,fData,labels,baseline] = calculateNDCI(Tsub, sensors)
sev=Tsub.Severity; X=Tsub{:,sensors}; h=(sev==0); f=(sev>0);
if any(h), base=mean(X(h,:),1,'omitnan'); r=max(X(h,:),[],1,'omitnan')-min(X(h,:),[],1,'omitnan');
else,      base=median(X,1,'omitnan');    r=max(X,[],1,'omitnan')-min(X,[],1,'omitnan'); end
r(~isfinite(r)|r<eps)=eps; Xf=X(f,:); labels=categorical(Tsub.FaultMode(f));
if isempty(Xf), z=zeros(1,numel(sensors)); ndci=z; SPn=z; Sn=z; Un=z; fData=array2table([], 'VariableNames', sensors); baseline=base; return; end
SP_raw=mean(abs(Xf-base)./r,1,'omitnan');
den=max(1-Tsub.Severity(f),eps);
S_raw=mean(abs(Xf-base)./den,1,'omitnan');
Z=zscoreSafe(Xf); C=corrcoef(Z,'Rows','pairwise'); if any(isnan(C),'all'), C(1:size(C,1)+1:end)=1; C(~isfinite(C))=0; end
D=1-abs(C); D(1:size(D,1)+1:end)=NaN; U_raw=nanmean(D,2)';
SPn=rescaleSafe(SP_raw); Sn=rescaleSafe(S_raw); Un=rescaleSafe(U_raw);
ndci=(SPn+Sn+Un)/3; fData=array2table(Xf,'VariableNames',sensors); baseline=base;
end

function Z=zscoreSafe(X)
mu=mean(X,1,'omitnan'); sg=std(X,0,1,'omitnan'); sg(~isfinite(sg)|sg<eps)=1; Z=(X-mu)./sg;
end
function y=rescaleSafe(x)
a=min(x); b=max(x); d=b-a; if d<eps, y=zeros(size(x)); else, y=(x-a)./d; end
end

function [Cost,MTBF,Weight]=attachAttributes(cand,A)
n=numel(cand); Cost=ones(1,n)*1.2e4; MTBF=2.0e5*ones(1,n); Weight=zeros(1,n);
if ~isempty(A)
    [lia,loc]=ismember(string(cand), A.Sensor);
    if any(lia)
        if ismember('Cost',A.Properties.VariableNames),   Cost(lia)=A.Cost(loc(lia)); end
        if ismember('MTBF',A.Properties.VariableNames),   MTBF(lia)=A.MTBF(loc(lia)); end
        if ismember('Weight',A.Properties.VariableNames), Weight(lia)=A.Weight(loc(lia)); end
    end
end
Cost(~isfinite(Cost)|Cost<=0)=median(Cost(Cost>0),'omitnan');
MTBF(~isfinite(MTBF)|MTBF<=0)=median(MTBF(MTBF>0),'omitnan');
Weight(~isfinite(Weight)|Weight<0)=0;
end

function bestHM = probe_bestcase_harmonic_mtbf(D)
sel=false(1,numel(D.MTBF));
for j=1:size(D.sysIdx,1)
    rg=D.sysIdx(j,1):D.sysIdx(j,2);
    [~,ord]=sort(D.MTBF(rg),'descend');
    keep=ord(1:D.kMin(j)); sel(rg(keep))=true;  % best case with kMin per system
end
m=D.MTBF(sel); if isempty(m), bestHM=0; else, bestHM = numel(m)/sum(1./m); end
end

%% -------------------- GA creation / repair / objective -------------------
function Pop = createPopFeasible(GenomeLength, options, D, cfg)
N = options.PopulationSize;
Pop = zeros(N, GenomeLength);
score = 0.5*D.NDCI + 0.5*D.U; % tie-break score
for i=1:N
    z = false(1,GenomeLength);
    for j=1:size(D.sysIdx,1)
        rg = D.sysIdx(j,1):D.sysIdx(j,2);
        kj = randi([D.kMin(j) D.kMax(j)]);
        s = score(rg) + 0.15*rand(1,numel(rg));
        [~,ord] = sort(s,'descend');
        pick = ord(1:kj);
        z(rg(pick)) = true;
        % ensure coverage
        need = D.covMin(j);
        if need>0 && sum(z(rg) & D.topMask{j}) < need
            idxTop = find(D.topMask{j}); sTop = s(idxTop); [~,o2]=sort(sTop,'descend');
            for t=1:min(need, numel(o2))
                selTop = idxTop(o2(t));
                if ~z(rg(selTop))
                    nonTopSel = find(z(rg) & ~D.topMask{j});
                    if ~isempty(nonTopSel)
                        [~,lowIdx] = min(s(nonTopSel));
                        z(rg(nonTopSel(lowIdx))) = false;
                    end
                    z(rg(selTop)) = true;
                end
            end
        end
    end
    z = trim_to_budget(z, D, cfg);
    Pop(i,:) = double(z);
end
end

function F = obj_with_repair(x, D, cfg, wSys)
z = repair_to_feasible(x>0.5, D, cfg);   % repair offspring
% Performance per system (per-k normalised) + weights
PerfSys = zeros(1,size(D.sysIdx,1));
for j=1:size(D.sysIdx,1)
    rg = D.sysIdx(j,1):D.sysIdx(j,2);
    sel = z(rg); kj = sum(sel);
    if kj==0, continue; end
    PerfSys(j) = cfg.alpha*sum(D.NDCI(rg(sel)))/kj + cfg.beta*mean(D.U(rg(sel)));
end
Perf = sum(wSys(:)'.*PerfSys);
Cost = D.Cost*z'; Benefit = sum(D.NDCI(z));
RelHM = numel(D.MTBF(z))/sum(1./D.MTBF(z)); if ~isfinite(RelHM), RelHM=0; end
penRel = max(0,(cfg.RelTarget-RelHM)/max(cfg.RelTarget,1)); % 0..1
BC = Benefit/(Cost+1e-9);
% objectives to MINIMISE
F = [ -Perf,  Cost*(1+2*penRel),  -RelHM,  -(1-0.3*penRel)*BC ];
end

function z = repair_to_feasible(z, D, cfg)
z = logical(z);
score = 0.6*D.NDCI + 0.4*D.U;

% 1) Per-system windows + coverage
for j=1:size(D.sysIdx,1)
    rg = D.sysIdx(j,1):D.sysIdx(j,2);
    sel = find(z(rg)); kj = numel(sel);
    if kj > D.kMax(j)
        [~,ord] = sort(score(rg(sel)),'descend');
        keep = sel(ord(1:D.kMax(j)));
        z(rg) = false; z(rg(keep)) = true; kj = D.kMax(j);
    end
    if kj < D.kMin(j)
        avail = find(~z(rg));
        [~,ord] = sort(score(rg(avail)),'descend');
        add = avail(ord(1:min(D.kMin(j)-kj, numel(avail))));
        z(rg(add)) = true; kj = sum(z(rg));
    end
    need = D.covMin(j);
    if need>0 && sum(z(rg) & D.topMask{j}) < need
        idxTop = find(D.topMask{j}); sTop = score(rg(idxTop));
        [~,o2]=sort(sTop,'descend');
        for t=1:min(need,numel(o2))
            tgt = idxTop(o2(t));
            if ~z(rg(tgt))
                nonTopSel = find(z(rg) & ~D.topMask{j});
                if ~isempty(nonTopSel)
                    [~,lo] = min(score(rg(nonTopSel)));
                    z(rg(nonTopSel(lo))) = false;
                end
                z(rg(tgt)) = true;
            end
        end
        sel = find(z(rg)); if numel(sel) > D.kMax(j)
            [~,ord] = sort(score(rg(sel)),'descend');
            z(rg) = false; z(rg(sel(ord(1:D.kMax(j))))) = true;
        end
    end
end

% 2) Budget/weight trim (global)
z = trim_to_budget(z, D, cfg);
end

function z = trim_to_budget(z, D, cfg)
z = logical(z);
if ~isempty(cfg.Wmax)
    while D.Weight*z' > cfg.Wmax
        canDrop = find(z);
        [~,ord] = sort((0.6*D.NDCI(canDrop)+0.4*D.U(canDrop))./max(D.Weight(canDrop),1e-6),'ascend');
        dropped=false;
        for ii=1:numel(ord)
            idx = canDrop(ord(ii));
            if ok_to_drop(idx, z, D), z(idx)=false; dropped=true; break; end
        end
        if ~dropped||~any(z), break; end
    end
end
if ~isempty(cfg.Budget)
    while D.Cost*z' > cfg.Budget
        canDrop = find(z);
        [~,ord] = sort((0.6*D.NDCI(canDrop)+0.4*D.U(canDrop))./max(D.Cost(canDrop),1e-9),'ascend');
        dropped=false;
        for ii=1:numel(ord)
            idx = canDrop(ord(ii));
            if ok_to_drop(idx, z, D), z(idx)=false; dropped=true; break; end
        end
        if ~dropped, break; end
    end
end
end

function tf = ok_to_drop(idx, z, D)
z(idx)=false; tf=true;
j = find(idx>=D.sysIdx(:,1) & idx<=D.sysIdx(:,2),1,'first');
rg = D.sysIdx(j,1):D.sysIdx(j,2);
if sum(z(rg)) < D.kMin(j), tf=false; end
if sum(z(rg) & D.topMask{j}) < D.covMin(j), tf=false; end
end

function idx = kneeSelect(F)
Y=F; Y(:,[1,3,4])=-Y(:,[1,3,4]); mn=min(Y,[],1); mx=max(Y,[],1); rng=max(mx-mn,1e-9);
Z=(Y-mn)./rng; ideal=[1 0 1 1]; d=sqrt(sum((Z-ideal).^2,2)); [~,idx]=min(d);
end

%% --------------------------- Plots & reports ----------------------------
function plotPareto(out)
F=out.F; Perf=-F(:,1); Cost=F(:,2); Rel=-F(:,3); BC=-F(:,4); k=out.kneeIndex;
Cost_k = Cost/1e3; Rel_kh = Rel/1e3;

% 3D
fig=figure('Color','w','Position',[100 100 980 720]);
scatter3(Perf,Cost_k,Rel_kh,60,BC,'filled','MarkerFaceAlpha',0.85,'MarkerEdgeAlpha',0.35); hold on; grid on;
plot3(Perf(k),Cost_k(k),Rel_kh(k),'kd','MarkerFaceColor','k','MarkerSize',9);
xlabel('Performance (↑)','FontSize',12); ylabel('Cost (k$) (↓)','FontSize',12); zlabel('Reliability (kh) (↑)','FontSize',12);
cb=colorbar; cb.Label.String='Benefit / Cost (↑)'; title('Airline Pareto — feasible','FontSize',14);
view(36,24); box on; exportgraphics(fig,'pareto_feasible_3D.png','Resolution',220); close(fig);

% 2D Perf vs Cost coloured by Rel
fig=figure('Color','w','Position',[100 100 980 700]);
scatter(Perf,Cost_k,55,Rel_kh,'filled','MarkerFaceAlpha',0.9); grid on; hold on;
plot(Perf(k),Cost_k(k),'kd','MarkerFaceColor','k','MarkerSize',9);
xlabel('Performance (↑)','FontSize',12); ylabel('Cost (k$) (↓)','FontSize',12);
cb=colorbar; cb.Label.String='Reliability (kh) (↑)'; title('Performance vs Cost','FontSize',14);
exportgraphics(fig,'pareto_feasible_2D.png','Resolution',220); close(fig);

% Parallel coordinates (normalised)
Z = normalizeObjectives(F);
labels = {'Performance','Cost (inv)','Reliability','B/C'};
fig=figure('Color','w','Position',[80 80 1000 560]); hold on; grid on;
for i=1:size(Z,1)
    c = [0.15 0.55 0.85]; lw = 0.7; a = 0.25;
    if i==k, c=[0 0 0]; lw=2.5; a=1; end
    plot(1:4,Z(i,:),'Color',[c a],'LineWidth',lw);
end
xlim([1 4]); xticks(1:4); xticklabels(labels); ylim([0 1]);
title('Normalised Objectives — Parallel Coordinates (knee bold)','FontSize',14);
exportgraphics(fig,'pareto_parallel.png','Resolution',220); close(fig);

% Anchors table (consistent 5×1 columns)
[~,iBestPerf]=max(Perf); [~,iBestRel]=max(Rel); [~,iBestBC]=max(BC); [~,iBestCost]=min(Cost);
idxList = [k; iBestPerf; iBestRel; iBestBC; iBestCost];   % 5×1
T = table( ...
    idxList, ...
    Perf(idxList), ...
    Cost_k(idxList), ...
    Rel_kh(idxList), ...
    BC(idxList), ...
    'VariableNames', {'Index','Perf','Cost_k$','Rel_kh','BbyC'}, ...
    'RowNames', {'Knee','BestPerf','BestRel','BestB/C','Cheapest'} ...
);
writetable(T,'anchors_summary.csv','WriteRowNames',true);
end

function Z = normalizeObjectives(F)
% Convert 4 objectives (minimise) to 0..1 where higher is better.
Y = F; Y(:,[1 3 4]) = -Y(:,[1 3 4]);     % Perf, Rel, B/C -> maximise
mn = min(Y,[],1); mx = max(Y,[],1); rng = max(mx-mn,1e-12);
Z = (Y - mn)./rng; 
Z(:,2) = 1 - Z(:,2);                    % flip cost so higher=better for the plot
end

function plotKneeBreakdown(out)
D=out.All; z = repair_to_feasible(out.ParetoSel(out.kneeIndex,:), D, out.cfg);
F=out.F; Perf=-F(:,1); Cost=F(:,2)/1e3; Rel=-F(:,3)/1e3; BC=-F(:,4); k=out.kneeIndex;

J=size(D.sysIdx,1); namesSys = strings(1,J); costS=zeros(1,J); mtbfS=zeros(1,J); kS=zeros(1,J);
for j=1:J
    rg=D.sysIdx(j,1):D.sysIdx(j,2);
    namesSys(j) = string(D.sysNames{D.sysIdx(j,1)});
    mask = z(rg);
    costS(j)=sum(D.Cost(rg(mask)))/1e3; 
    mtbfS(j)=numel(D.MTBF(rg(mask)))/sum(1./D.MTBF(rg(mask)))/1e3; % harmonic, kh
    kS(j)=sum(mask);
end

fig=figure('Color','w','Position',[80 80 1100 420]); 
t=tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

nexttile; 
bar(categorical(namesSys),costS); ylabel('Cost (k$)'); title('Knee cost by subsystem');
grid on;

nexttile;
bar(categorical(namesSys),mtbfS); ylabel('Harmonic MTBF (kh)'); title('Knee reliability by subsystem');
grid on;
exportgraphics(fig,'knee_subsystem_breakdown.png','Resolution',220); close(fig);

% Lollipop of knee sensors by NDCI
idx=find(z); nd=D.NDCI(idx); [nd,ord]=sort(nd,'descend'); idx=idx(ord);
lab = string(D.names(idx));
col = grp2idx(string(D.sysNames(idx)));
fig=figure('Color','w','Position',[80 80 1000 680]); 
stem(nd,'filled','LineWidth',1.6); grid on;
set(gca,'XTick',1:numel(idx),'XTickLabel',lab,'XTickLabelRotation',90);
ylabel('NDCI (0–1)'); title('Knee sensors ranked by NDCI (colour = subsystem)');
hold on; scatter(1:numel(idx), nd, 60, col, 'filled','MarkerEdgeColor',[.2 .2 .2]);
exportgraphics(fig,'knee_lollipop_ndci.png','Resolution',220); close(fig);

% Small text file with knee headline numbers
fid=fopen('knee_headline.txt','w');
fprintf(fid,'KNEE — Perf=%.3f  Cost=%.1fk$  Rel=%.1fkh  B/C=%.2g  k=%d\n', ...
        Perf(k), Cost(k), Rel(k), BC(k), sum(z));
for j=1:J, fprintf(fid,'  %-6s: %d\n', namesSys(j), kS(j)); end
fclose(fid);
end

function plotCountsWindows(out)
D=out.All; Z=out.ParetoSel;
J=size(D.sysIdx,1); N=size(Z,1);
K=zeros(N,J);
for i=1:N
    z=repair_to_feasible(Z(i,:),D,out.cfg);
    for j=1:J, rg=D.sysIdx(j,1):D.sysIdx(j,2); K(i,j)=sum(z(rg)); end
end
mn=D.kMin; mx=D.kMax; sys=arrayfun(@(j)string(D.sysNames{D.sysIdx(j,1)}),1:J);

fig=figure('Color','w','Position',[80 80 900 500]); 
tiledlayout(1,J,'TileSpacing','compact','Padding','compact');
for j=1:J
    nexttile; scatter(1:N,K(:,j),18,'filled'); hold on; yline(mn(j),'r--'); yline(mx(j),'r--');
    title(sys(j)); xlabel('Solution #'); ylabel('Count'); grid on; ylim([min(mn)-1 max(mx)+2]);
end
exportgraphics(fig,'counts_vs_windows.png','Resolution',220); close(fig);
end

function metrics = paretoQuality(out)
Z = normalizeObjectives(out.F);
% Monte-Carlo hypervolume wrt origin (0,0,0,0) in 4D
M = 20000; U = rand(M,4);  dom = false(M,1);
for i=1:size(Z,1), dom = dom | all(bsxfun(@le, U, Z(i,:)),2); end
HV = mean(dom);                 % 0..1
% Spread Δ on first two axes (Perf vs inv-Cost)
P = sortrows(Z,1); f = P(:,1:2); d = sqrt(sum(diff(f).^2,2)); 
Delta = (sum(abs(d - mean(d)))/(numel(d)*mean(d)+eps));
metrics = struct('Hypervolume',HV,'Spread',Delta);
T = struct2table(metrics); writetable(T,'pareto_metrics.csv');
fprintf('Pareto metrics — Hypervolume: %.3f  |  Spread Δ: %.3f\n', HV, Delta);
end

function exportTables(out)
F=out.F; S=out.ParetoSel; D=out.All; Perf=-F(:,1); Cost=F(:,2)/1e3; Rel=-F(:,3)/1e3; BC=-F(:,4); k=out.kneeIndex;
[~,iBestPerf]=max(Perf); [~,iBestRel]=max(Rel); [~,iBestBC]=max(BC); [~,iBestCost]=min(Cost);
ix=unique([k,iBestPerf,iBestRel,iBestBC,iBestCost]);
rows=[];
for t=ix(:)'
    z = repair_to_feasible(S(t,:), D, out.cfg);
    sysCnt=zeros(1,size(D.sysIdx,1));
    for j=1:size(D.sysIdx,1), rg=D.sysIdx(j,1):D.sysIdx(j,2); sysCnt(j)=sum(z(rg)); end
    rows=[rows; Perf(t),Cost(t),Rel(t),BC(t),sum(z),sysCnt];
end
vn=["Perf","Cost_k$","Rel_kh","BbyC","kTotal","kEngine","kFuel","kElec","kECS"]; vn=vn(1:5+size(D.sysIdx,1));
Tsum=array2table(rows,'VariableNames',cellstr(vn)); writetable(Tsum,'pareto_summary_feasible.xlsx','Sheet','summary');

z=repair_to_feasible(S(k,:), D, out.cfg); idx=find(z);
Knee=table(string(D.names(idx))','VariableNames',{'Sensor'});
Knee.Subsystem = string(D.sysNames(idx))';
Knee.Cost      = D.Cost(idx)'; Knee.MTBF=D.MTBF(idx)'; Knee.Weight=D.Weight(idx)';
Knee.NDCI      = D.NDCI(idx)'; Knee.U=D.U(idx)';
nr=height(Knee); 
Knee.Perf      = repmat(Perf(k),nr,1);
Knee.CostTotal = repmat(Cost(k),nr,1);
Knee.RelTotal  = repmat(Rel(k),nr,1);
Knee.BenefitCost=repmat(BC(k),nr,1);
writetable(Knee,'knee_suite_feasible.xlsx','Sheet','knee');
end

function enhanceParetoPlots(out, savePrefix)
% ENHANCEPARETOPLOTS  add legend, sensor names, and rich datatips
%
% Required fields in out:
%   out.F(:,1:4)  -> [Performance, Cost, Reliability, BenefitCost]
%   out.Solutions -> 1xN struct with either:
%       .names  (cellstr of sensor names)   OR
%       .SelMask (logical mask) with out.All.Names available
%       and per-subsystem counts (nEngine,nFuel,nElec,nECS) OR .counts struct
%   out.kneeIndex -> index of knee
% Optional:
%   out.All.Names (cellstr)
%
% Saves: <savePrefix>_2D.png, <savePrefix>_3D.png

% ---- gather arrays ------------------------------------------------------
F   = out.F;                            % [Perf, Cost, Rel, BC]
Perf= F(:,1); Cost=F(:,2); Rel=F(:,3);  BC = F(:,4);

N   = numel(out.Solutions);
namesCell = cell(N,1);
nE = zeros(N,1); nF = zeros(N,1); nEl = zeros(N,1); nEC = zeros(N,1);

for i=1:N
    s = out.Solutions(i);
    % sensor names
    if isfield(s,'names') && ~isempty(s.names)
        namesCell{i} = strjoin(s.names, ', ');
    elseif isfield(s,'Sensors') && ~isempty(s.Sensors)
        namesCell{i} = strjoin(s.Sensors, ', ');
    elseif isfield(s,'SelMask') && isfield(out,'All') && isfield(out.All,'Names')
        idx = find(s.SelMask);
        namesCell{i} = strjoin(out.All.Names(idx), ', ');
    else
        namesCell{i} = '(names unavailable)';
    end
    % counts
    if isfield(s,'nEngine'),   nE(i)  = s.nEngine; end
    if isfield(s,'nFuel'),     nF(i)  = s.nFuel;   end
    if isfield(s,'nElec'),     nEl(i) = s.nElec;   end
    if isfield(s,'nECS'),      nEC(i) = s.nECS;    end
    if isfield(s,'counts') && isstruct(s.counts)
        if nE(i)==0 && isfield(s.counts,'Engine'), nE(i)=s.counts.Engine; end
        if nF(i)==0 && isfield(s.counts,'Fuel'),   nF(i)=s.counts.Fuel;   end
        if nEl(i)==0 && isfield(s.counts,'Elec'),  nEl(i)=s.counts.Elec;  end
        if nEC(i)==0 && isfield(s.counts,'ECS'),   nEC(i)=s.counts.ECS;   end
    end
end

% dominant subsystem (for legend/markers)
[~,domIdx] = max([nE nF nEl nEC],[],2);
domNames = categorical(domIdx,1:4,{'Engine','Fuel','Elec','ECS'});

% choose marker per subsystem
mk = {'o','s','^','d'};
markers = mk(domIdx);

% ---- helper to attach rich datatips ------------------------------------
    function attachTips(hScatter)
        % suite id
        r1 = dataTipTextRow('Suite', (1:N).');
        r2 = dataTipTextRow('Performance', Perf);
        r3 = dataTipTextRow('Cost (k$)', Cost);
        r4 = dataTipTextRow('Reliability (kh)', Rel);
        r5 = dataTipTextRow('Benefit/Cost', BC);
        r6 = dataTipTextRow('Dominant', cellstr(domNames));
        r7 = dataTipTextRow('Counts [E,Fu,El,ECS]', [nE nF nEl nEC]);
        % Sensor list (one long line)
        r8 = dataTipTextRow('Sensors', namesCell);
        hScatter.DataTipTemplate.DataTipRows = [r1 r2 r3 r4 r5 r6 r7 r8];
    end

% ---- 2D: Performance vs Cost (colour = Reliability) --------------------
figure('Color','w','Position',[100 80 900 700]); hold on; grid on;
% draw by subsystem w/ different markers (and keep colour for Rel)
lbl = {'Engine','Fuel','Elec','ECS'};
hLeg = gobjects(1,4);
for g=1:4
    pick = domIdx==g;
    h = scatter(Perf(pick), Cost(pick), 55, Rel(pick), markers{g}, ...
        'filled','MarkerFaceAlpha',0.9,'MarkerEdgeColor',[0 0 0 0.15]);
    if any(pick)
        hLeg(g) = h;
    end
end
colormap(parula); cb = colorbar; cb.Label.String = 'Reliability (kh) (↑)';
xlabel('Performance (↑)'); ylabel('Cost (k$) (↓)');
title('Performance vs Cost');
% knee + anchors
k  = out.kneeIndex;
[~,iBestPerf] = max(Perf);  [~,iBestRel] = max(Rel);
[~,iBestBC]   = max(BC);    [~,iBestCost] = min(Cost);
plot(Perf(k),Cost(k),'kd','MarkerFaceColor','k','MarkerSize',8);
plot(Perf(iBestPerf),Cost(iBestPerf),'gp','MarkerFaceColor','g','MarkerSize',10);
plot(Perf(iBestRel),Cost(iBestRel),'bp','MarkerFaceColor','b','MarkerSize',10);
plot(Perf(iBestBC),Cost(iBestBC),'mp','MarkerFaceColor','m','MarkerSize',10);
plot(Perf(iBestCost),Cost(iBestCost),'cp','MarkerFaceColor','c','MarkerSize',10);
% direct labels for knee + anchors
txt = @(i,lab) text(Perf(i),Cost(i), sprintf('  %s: %s',lab, shortName(namesCell{i})), ...
    'FontSize',9,'FontWeight','bold','Interpreter','none',...
    'HorizontalAlignment','left','VerticalAlignment','bottom');
txt(k,'Knee'); txt(iBestPerf,'BestPerf'); txt(iBestRel,'BestRel');
txt(iBestBC,'BestB/C'); txt(iBestCost,'MinCost');
% legend for markers (subsystems)
legend(hLeg, lbl, 'Location','northwest'); legend boxoff;
% attach datatips
attachTips(findobj(gca,'Type','Scatter'));
exportgraphics(gcf, sprintf('%s_2D.png', savePrefix), 'Resolution', 200);

% ---- 3D: Performance–Cost–Reliability (colour = Benefit/Cost) ----------
figure('Color','w','Position',[80 60 1000 760]);
ax = axes; hold(ax,'on'); grid(ax,'on'); view(ax, -30, 20);
for g=1:4
    pick = domIdx==g;
    h = scatter3(Perf(pick), Cost(pick), Rel(pick), 65, BC(pick), markers{g}, ...
        'filled','MarkerFaceAlpha',0.95,'MarkerEdgeColor',[0 0 0 0.1]);
    if any(pick)
        hLeg(g) = h;
    end
end
colormap(ax, parula); cb = colorbar; cb.Label.String = 'Benefit/Cost (↑)';
xlabel('Performance (↑)'); ylabel('Cost (k$) (↓)'); zlabel('Reliability (kh) (↑)');
title('Airline Pareto — feasible');
% knee + anchors
plot3(Perf(k),Cost(k),Rel(k),'kd','MarkerFaceColor','k','MarkerSize',8);
plot3(Perf(iBestPerf),Cost(iBestPerf),Rel(iBestPerf),'gp','MarkerFaceColor','g','MarkerSize',10);
plot3(Perf(iBestRel),Cost(iBestRel),Rel(iBestRel),'bp','MarkerFaceColor','b','MarkerSize',10);
plot3(Perf(iBestBC),Cost(iBestBC),Rel(iBestBC),'mp','MarkerFaceColor','m','MarkerSize',10);
plot3(Perf(iBestCost),Cost(iBestCost),Rel(iBestCost),'cp','MarkerFaceColor','c','MarkerSize',10);
% legend (markers = dominant subsystem)
legend(hLeg, lbl, 'Location','northeastoutside'); legend boxoff;
% datatips
attachTips(findobj(ax,'Type','Scatter'));
exportgraphics(gcf, sprintf('%s_3D.png', savePrefix), 'Resolution', 200);

end % enhanceParetoPlots

% --- helper: shorten very long sensor lists for inline labels ------------
function s = shortName(longS)
    maxChars = 40;
    if numel(longS) <= maxChars
        s = longS;
    else
        s = [longS(1:maxChars-3) '...'];
    end
end
