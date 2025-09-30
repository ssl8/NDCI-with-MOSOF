function results = analysis_pipeline_nested_pro(varargin)
% ANALYSIS_PIPELINE_NESTED_PRO
% Rigorous nested CV pipeline that:
%  - Compares NDCI vs mRMR feature rankings
%  - Performs inner-CV classifier search (Bag, ECOC-SVM[RBF], subspace kNN, kernel NB; optional RUSBoost)
%  - Selects classifier AND k by policy: 'oneSE' or 'threshold' (TargetFrac)
%  - Computes isolation metrics (balanced acc, macro-F1, macro-recall, MCC)
%  - Skips detection confusion if detection is trivially perfect
%  - Produces clean, publication-ready figures and CSVs
%
% Compatible with Statistics and Machine Learning Toolbox.

%% ------------------------ Options ------------------------

opt = parseOptions(varargin{:});
rng(opt.Seed);

% Backwards‑compatible support for optional positional filename
if ~isfield(opt,'PositionalExcel')
    opt.PositionalExcel = [];
end
if ~isempty(opt.PositionalExcel)
    opt.excelFile = opt.PositionalExcel;
end


excelFile = char(opt.excelFile);
if ~isfile(excelFile), error('Input Excel file not found: %s', excelFile); end
fprintf('--- Data loaded (%s) ---\n', excelFile);

%% -------------------- Sensor groups & FM -------------------
sensorGroups.Engine = { ...
    'W_S1','W_S2','W_S21','W_S24','W_S3','W_S4','W_S45','W_S5','W_S7', ...
    'h_S1','h_S2','h_S21','h_S24','h_S3','h_S4','h_S45','h_S5','h_S7','h_S17', ...
    'Tt_S1','Tt_S2','Tt_S21','Tt_S24','Tt_S3','Tt_S4','Tt_S45','Tt_S5','Tt_S7','Tt_S17', ...
    'Pt_S1','Pt_S21','Pt_S3','Pt_S4','Pt_S45','Pt_S5','FanTrq','LPCTrq','HPCTrq','HPTTrq','LPTTrq','Thrust', ...
    'h_Fan','Tt_Fan','Pt_Fan','Wf', ...
    'W_BfBleed','W_AfBleed','W_BfCDP','W_AfCDP','h_BfBleed','h_AfBleed','h_BfCDP','h_AfCDP', ...
    'Tt_BfBleed','Tt_AfBleed','Tt_BfCDP','Tt_AfCDP','Pt_BfBleed','Pt_AfBleed','Pt_BfCDP','Pt_AfCDP','Tsfc'};
sensorGroups.Fuel = {'P1','P2','P3','P4','P5','P6','P7','Flow'};
sensorGroups.Elec = {'Power_GenOnly','Power_AC_Gen_load','Power_ACLamp','Power_TRU', ...
                     'AC_Fluoro_I','AC_Fluoro_V','AC_Instru_V','Eng_BleedValve_I','Eng_BleedValve_V'};
sensorGroups.ECS  = {'PVOT','ThiPHX','ThoPHX','TiC','ToC','ThiSHX','ThoSHX','ThiRHX','ThoRHX', ...
                     'ThiCHX','ThoCHX','TciRHX','TcoRHX','TiT','ToT'};

FM.Elec   = compose("FM%d",1:7);
FM.Fuel   = compose("FM%d",8:13);
FM.Engine = compose("FM%d",14:20);
FM.ECS    = compose("FM%d",21:25);

%% -------------------- Load & constant-prune -----------------
T = readtable(excelFile);
T.Properties.VariableNames = matlab.lang.makeValidName(T.Properties.VariableNames);

% Remove near-constant globally. Redundancy handled per subsystem during CV.
exists = unique([sensorGroups.Engine, sensorGroups.Fuel, sensorGroups.Elec, sensorGroups.ECS],'stable');
exists = intersect(exists, T.Properties.VariableNames, 'stable');
v = var(T{:, exists}, 0, 1, 'omitnan');
const = exists(v < 1e-6);
if ~isempty(const)
    fprintf('Remove near‑constant sensors: %s\n', strjoin(const, ','));
end
T(:, const) = [];
fn = fieldnames(sensorGroups);
for i = 1:numel(fn)
    sensorGroups.(fn{i}) = intersect(sensorGroups.(fn{i}), T.Properties.VariableNames, 'stable');
end
fprintf('--- Preprocessing complete (constants removed; redundancy handled per subsystem) ---\n');

%% -------------------- Classifier catalog -------------------
clfCatalog = buildClassifierCatalog(opt);

%% -------------------- Nested CV per subsystem ---------------
subs = fieldnames(sensorGroups);
results = struct(); manifest = struct();

for i = 1:numel(subs)
    subName = subs{i}; sensors = sensorGroups.(subName);
    if isempty(sensors), continue; end
    switch subName
        case 'Engine', subFM = FM.Engine;
        case 'Fuel',   subFM = FM.Fuel;
        case 'Elec',   subFM = FM.Elec;
        case 'ECS',    subFM = FM.ECS;
    end
    Tsub = T(ismember(T.FaultMode, subFM) | strcmp(T.FaultMode,'Healthy'), :);
    fprintf('\n--- %s ---\nRows: %d, Candidate sensors: %d\n', subName, height(Tsub), numel(sensors));

    % Illustrative full-data ranking (not used for eval)
    [ndciAll, SP, S, U, fDataAll, labelsAll] = calculateNDCI(Tsub, sensors, opt);
    [~, idxN] = sort(ndciAll,'descend');
    [sMAll, ~] = rank_mRMR_or_ANOVA(fDataAll, labelsAll, sensors);
    saveRankingPlot(subName, sensors, ndciAll, SP, S, U, sMAll);

    % Nested CV: model selection + k selection + evaluation
    [foldRes, stepN, stepM, compTbl] = nestedCV(Tsub, sensors, subName, subFM, opt, clfCatalog);
    results.(subName) = foldRes;

    % Stepwise overlay and confusion (only if results exist)
    if ~isempty(foldRes.iso.allTrue)
        plotStepwiseCombined(subName, stepN, stepM);
        saveConfusion(subName, foldRes.iso.method, foldRes.iso.allTrue, foldRes.iso.allPred);
    else
        fprintf('No valid isolation results for %s — skipping plots.\n', subName);
    end

    % Detection confusion skipped if perfect
    if ~foldRes.det.perfect && ~isempty(foldRes.det.allTrue)
        saveConfusion(subName, [foldRes.det.method ' (detection)'], foldRes.det.allTrue, foldRes.det.allPred);
    end

    % Persist classifier comparison
    if ~isempty(compTbl)
        writetable(compTbl, sprintf('classifier_comp_%s.csv', subName));
    end

    % Manifest summary
    manifest.(subName) = rmfield(foldRes, {'iso','det'});
    manifest.(subName).Isolation = foldRes.iso.summary;
    manifest.(subName).Detection = foldRes.det.summary;
end

% Final bar comparison across subsystems (isolation, NDCI means)
saveFinalComparison(results, subs);

save('run_manifest_nested_pro.mat','results','manifest','opt');

end % main


%% ========================== Options ==========================
function opt = parseOptions(varargin)
% Robust options parser: supports optional positional filename
% and standard name–value pairs. Always returns a struct with
% 'PositionalExcel' (possibly empty).

p = inputParser; 
p.CaseSensitive = false;

% Recognized parameter names
known = {'excelFile','Repeats','OuterFolds','InnerFolds','CorrThr','Seed', ...
         'UseParallel','SavePRCurves','SeverityMode','RedundancyMetric', ...
         'Classifiers','KRule','TargetFrac'};

% Start with default struct and include PositionalExcel by default
opt = struct('PositionalExcel',[]);

% If first arg is a filename (not a known name), treat as positional file
if ~isempty(varargin) && (ischar(varargin{1}) || isstring(varargin{1}))
    first = char(varargin{1});
    if ~any(strcmpi(first, known))
        opt.PositionalExcel = first;
        varargin = varargin(2:end);   % consume positional filename
    end
end

% Define defaults for name–value pairs
addParameter(p,'excelFile','DataB.xlsx',@(s)ischar(s)||isstring(s));
addParameter(p,'Repeats',10,@(x)isnumeric(x)&&x>=1);
addParameter(p,'OuterFolds',5,@(x)isnumeric(x)&&x>=2);
addParameter(p,'InnerFolds',3,@(x)isnumeric(x)&&x>=2);
addParameter(p,'CorrThr',0.995,@(x)isnumeric(x)&&x>0&&x<1);
addParameter(p,'Seed',42,@isnumeric);
addParameter(p,'UseParallel',false,@(x)islogical(x)||ismember(x,[0 1]));
addParameter(p,'SavePRCurves',false,@(x)islogical(x)||ismember(x,[0 1]));
addParameter(p,'SeverityMode','ramp',@(s)any(strcmpi(s,{'ramp','none','tau'})));
addParameter(p,'RedundancyMetric','pearson',@(s)any(strcmpi(s,{'pearson','spearman'})));
addParameter(p,'Classifiers','auto',@(c)ischar(c)||isstring(c)||iscellstr(c));
addParameter(p,'KRule','oneSE',@(s)any(strcmpi(s,{'oneSE','threshold'})));
addParameter(p,'TargetFrac',0.95,@(x)isnumeric(x)&&x>0&&x<=1);

% Parse remaining args
parse(p, varargin{:});
res = p.Results;

% Merge parsed results into opt (preserve PositionalExcel if set above)
fn = fieldnames(res);
for k = 1:numel(fn)
    opt.(fn{k}) = res.(fn{k});
end
end



%% ======================== Ranking ===========================
function [ndci, SPn, Sn, Un, fData, labels, baseline] = calculateNDCI(Tsub, sensors, opt)
sev = Tsub.Severity; X = Tsub{:, sensors};
h = (sev==0); f = (sev>0);
if any(h)
    baseline = mean(X(h,:),1,'omitnan');
    r = max(X(h,:),[],1,'omitnan') - min(X(h,:),[],1,'omitnan');
else
    baseline = median(X,1,'omitnan');
    r = max(X,[],1,'omitnan') - min(X,[],1,'omitnan');
end
r(~isfinite(r) | r<eps) = eps;

Xf = X(f,:); labels = categorical(Tsub.FaultMode(f)); sevF = sev(f);
if isempty(Xf)
    z = zeros(1, numel(sensors)); ndci=z; SPn=z; Sn=z; Un=z; fData = array2table([], 'VariableNames', sensors); return;
end

SP_raw = mean(abs(Xf - baseline)./r, 1, 'omitnan');  % separation

switch lower(opt.SeverityMode)
    case 'ramp'
        denom = max(1 - sevF, eps);  % higher when severity small
        S_raw = mean(abs(Xf - baseline)./denom, 1, 'omitnan');
    case 'tau'
        S_raw = nan(1,size(Xf,2));
        for j=1:size(Xf,2)
            S_raw(j) = abs(corr(Xf(:,j), sevF, 'type','Kendall','rows','pairwise'));
        end
    otherwise
        S_raw = zeros(1,size(Xf,2));
end

Z = zscoreSafe(Xf);
switch lower(opt.RedundancyMetric)
    case 'spearman', C = corr(Z, 'type','Spearman', 'rows','pairwise');
    otherwise,      C = corrcoef(Z,'Rows','pairwise');
end
if any(isnan(C),'all'), C(1:size(C,1)+1:end)=1; C(~isfinite(C))=0; end
D  = 1 - abs(C); D(1:size(D,1)+1:end) = NaN;
U_raw = nanmean(D,2)';  % uniqueness

SPn = rescaleSafe(SP_raw); Sn = rescaleSafe(S_raw); Un = rescaleSafe(U_raw);
ndci = (SPn + Sn + Un)/3;

fData = array2table(Xf, 'VariableNames', sensors);
end

function [ranked, scores] = rank_mRMR_or_ANOVA(dataTbl, labels, sensors)
X = table2array(dataTbl);
ranked = sensors; scores = zeros(1, numel(sensors));
try
    if exist('fscmrmr','file') == 2
        [idx,scr] = fscmrmr(X,labels); ranked = sensors(idx); scores = scr(idx); return;
    end
catch, end
F = nan(1,size(X,2));
for j=1:size(X,2)
    try, [~,tbl] = anova1(X(:,j), labels,'off'); F(j) = tbl{2,5}; catch, F(j) = 0; end
end
[~,idx] = sort(F,'descend'); ranked = sensors(idx); scores = F(idx);
end


%% =================== Redundancy pruning ====================
function [sensorsOut, keepMask, fDataOut] = pruneRedundantSensors(sensorsIn, fDataIn, score, thr)
X = table2array(fDataIn); keep = true(1,numel(sensorsIn));
while true
    idx = find(keep); if numel(idx) < 2, break; end
    C = corrcoef(X(:,idx), 'Rows','pairwise'); A = abs(C); A(1:size(A,1)+1:end) = 0;
    [ii,jj] = find(A > thr, 1);
    if isempty(ii), break; end
    a = idx(ii); b = idx(jj);
    if score(a) >= score(b), keep(b) = false; else, keep(a) = false; end
end
keepMask  = keep; sensorsOut = sensorsIn(keep); fDataOut = fDataIn(:, keep);
end


%% =================== Nested CV & selection =================
function [foldRes, stepN, stepM, compTbl] = nestedCV(Tsub, sensorsAll, subName, subFM, opt, catalog)
yAll  = categorical(Tsub.FaultMode);
Kout  = min(opt.OuterFolds, max(2, min(countcats(yAll))));
outer = cvpartition(yAll,'KFold',Kout);

det = initDet(); iso = initIso();
stepN = {}; stepM = {}; compRows = [];

for r = 1:opt.Repeats
    rng(opt.Seed + r);
    outer = repartition(outer);

    for fo = 1:outer.NumTestSets
        trMask = training(outer,fo); teMask = test(outer,fo);
        Train = Tsub(trMask,:); Test = Tsub(teMask,:);

        % NDCI on TRAIN, produce TRAIN-fault table and labels
        [ndci,~,~,~,Ftr,ytr] = calculateNDCI(Train, sensorsAll, opt);
        if height(Ftr) < 2 || numel(categories(ytr)) < 2
            fprintf('   - Skip fold r%d f%d (insufficient training faults: %d / classes: %d)\n', r, fo, height(Ftr), numel(categories(ytr))); 
            continue;
        end

        % Remove NaN-scored sensors; prune redundancy on TRAIN faults
        valid = ~isnan(ndci);
        sensors = sensorsAll(valid);
        ndci    = ndci(valid); 
        Ftr     = Ftr(:, valid);
        if isempty(sensors), fprintf('   - Skip r%d f%d (no valid sensors after NaN filtering)\n', r, fo); continue; end
        [sensors, keepMask, Ftr] = pruneRedundantSensors(sensors, Ftr, ndci, opt.CorrThr);
        ndci = ndci(keepMask);
        if isempty(sensors), fprintf('   - Skip r%d f%d (all sensors pruned by redundancy)\n', r, fo); continue; end

        % Rankings on TRAIN only
        [~, idxN] = sort(ndci,'descend'); sN = sensors(idxN);
        [sM, ~]   = rank_mRMR_onFaultTbl(Ftr, ytr, sensors);

        % Inner CV – detection (Healthy vs Fault) model+K search
        [histD_N, kD_N, clsD_N, peakD_N, tblD_N] = innerDetCV(Train, sN, opt, catalog);
        [histD_M, kD_M, clsD_M, peakD_M, tblD_M] = innerDetCV(Train, sM, opt, catalog);
        stepN{end+1} = histD_N; stepM{end+1} = histD_M;

        % Inner CV – isolation (multi-class faults) model+K search
        [histI_N, kI_N, clsI_N, peakI_N, tblI_N] = innerIsoCV(Ftr, ytr, sN, opt, catalog);
        [histI_M, kI_M, clsI_M, peakI_M, tblI_M] = innerIsoCV(Ftr, ytr, sM, opt, catalog);

        % Add to comparison records
        compRows = [compRows; addCompRows(subName, r, fo, 'NDCI', tblD_N, tblI_N); ...
                              addCompRows(subName, r, fo, 'mRMR', tblD_M, tblI_M)];

        % ---- Train on TRAIN and evaluate on TEST ----
        % Detection
        yDetTrain = categorical(Train.Severity > 0, [false true], {'Healthy','Fault'});
        yDetTest  = categorical(Test.Severity  > 0, [false true], {'Healthy','Fault'});

        XDetTrain_N = tableCols(Train, sN, kD_N); 
        XDetTrain_M = tableCols(Train, sM, kD_M);
        mdlDet_N = trainByName(clsD_N, XDetTrain_N, yDetTrain);
        mdlDet_M = trainByName(clsD_M, XDetTrain_M, yDetTrain);

        XDetTest_N = tableCols(Test, sN, kD_N);
        XDetTest_M = tableCols(Test, sM, kD_M);
        ypredDn = safePredict(mdlDet_N, XDetTest_N, yDetTest);
        ypredDm = safePredict(mdlDet_M, XDetTest_M, yDetTest);

        det = addDet(det, yDetTest, ypredDn, 'NDCI', clsD_N, kD_N, peakD_N);
        det = addDet(det, yDetTest, ypredDm, 'mRMR', clsD_M, kD_M, peakD_M);

        % Isolation (fault modes) — TEST faults only
        TestF = Test(Test.Severity > 0, :);
        if ~isempty(TestF)
            yIsoTest = categorical(TestF.FaultMode);

            XIsoTrain_N = tableCols(Ftr, sN, kI_N);
            XIsoTrain_M = tableCols(Ftr, sM, kI_M);
            mdlIso_N = trainByName(clsI_N, XIsoTrain_N, ytr);
            mdlIso_M = trainByName(clsI_M, XIsoTrain_M, ytr);

            XIsoTest_N  = tableCols(TestF, sN, kI_N);
            XIsoTest_M  = tableCols(TestF, sM, kI_M);
            ypredIn = safePredict(mdlIso_N, XIsoTest_N, yIsoTest);
            ypredIm = safePredict(mdlIso_M, XIsoTest_M, yIsoTest);

            iso = addIso(iso, yIsoTest, ypredIn, 'NDCI', clsI_N, kI_N, peakI_N);
            iso = addIso(iso, yIsoTest, ypredIm, 'mRMR', clsI_M, kI_M, peakI_M);
        end
    end
end

% Summaries
foldRes.det = finalizeDet(det);
foldRes.iso = finalizeIso(iso);
compTbl = []; 
if ~isempty(compRows), compTbl = struct2table(compRows); end

end


%% =================== Inner searches ========================
function [hist, kSel, clsSel, peak, compTbl] = innerDetCV(Train, ranked, opt, catalog)
% Binary detection Healthy vs Fault
yDet = categorical(Train.Severity > 0, [false true], {'Healthy','Fault'});
if numel(categories(yDet)) < 2
    L = numel(ranked); hist = nan(1,L); kSel = 1; clsSel = 'bag'; peak = NaN; compTbl = []; return;
end
[inner, ~] = makeInnerCV(yDet, opt.InnerFolds);
L = numel(ranked);
histPer = nan(numel(catalog), L); rec = [];

for c = 1:numel(catalog)
    row = nan(1,L);
    for k = 1:L
        feat = ranked(1:k); X = tableCols(Train, feat, k);
        acc = nan(1, inner.NumTestSets);
        for fi = 1:inner.NumTestSets
            tri = training(inner,fi); tei = test(inner,fi);
            yt = yDet(tri); ye = yDet(tei);
            if numel(categories(yt)) < 2, continue; end
            mdl = safeTrain(catalog(c), X(tri,:), yt);
            pr  = safePredict(mdl, X(tei,:), ye);
            acc(fi) = balancedAccBinary(ye, pr);
        end
        row(k) = nanmean(acc);
    end
    histPer(c,:) = row;
    rec = [rec; collectComp('det', catalog(c).Name, row)];
end

[peak, idxC] = max(max(histPer,[],2,'omitnan'));
clsSel = catalog(idxC).Name; hist = histPer(idxC,:);
kSel = selectK_generic(hist, opt);
compTbl = struct2table(rec);
end

function [hist, kSel, clsSel, peak, compTbl] = innerIsoCV(Ftr, ytr, ranked, opt, catalog)
[inner, ~] = makeInnerCV(ytr, opt.InnerFolds);
L = numel(ranked);
histPer = nan(numel(catalog), L); rec = [];

for c = 1:numel(catalog)
    row = nan(1,L);
    for k = 1:L
        feat = ranked(1:k); X = tableCols(Ftr, feat, k);
        acc = nan(1, inner.NumTestSets);
        for fi = 1:inner.NumTestSets
            tri = training(inner,fi); tei = test(inner,fi);
            yt = removecats(ytr(tri)); ye = ytr(tei);
            if numel(categories(yt)) < 2 || size(X(tri,:),1) < 2, continue; end
            mdl = safeTrain(catalog(c), X(tri,:), yt);
            pr  = safePredict(mdl, X(tei,:), ye);
            acc(fi) = balancedAccMulti(ye, pr);
        end
        row(k) = nanmean(acc);
    end
    histPer(c,:) = row;
    rec = [rec; collectComp('iso', catalog(c).Name, row)];
end

[peak, idxC] = max(max(histPer,[],2,'omitnan'));
clsSel = catalog(idxC).Name; hist = histPer(idxC,:);
kSel = selectK_generic(hist, opt);
compTbl = struct2table(rec);
end

function [inner, Ksafe] = makeInnerCV(y, K)
y = removecats(categorical(y));
if numel(y) < 4 || numel(categories(y)) < 2
    inner = cvpartition(numel(y), 'KFold', 2); Ksafe = 2; return;
end
minPerClass = min(countcats(y));
Ksafe = min([K, minPerClass, numel(y)-1]); Ksafe = max(Ksafe,2);
try
    inner = cvpartition(y, 'KFold', Ksafe);
catch
    inner = cvpartition(numel(y), 'KFold', 2);
    Ksafe = 2;
end
end


%% ================ Catalog / Train / Predict =================
function catalog = buildClassifierCatalog(opt)
% Build a robust classifier catalog. Supports:
% 'bag' (Bagged Trees), 'svmRBF' (ECOC SVM with RBF),
% 'subKNN' (kNN with standardization), 'nbKernel' (Kernel Naive Bayes),
% 'rusBoost' (optional). You can pass 'auto' to enable the default set.

list = opt.Classifiers;
if ischar(list) || isstring(list), list = string(list); end

% Normalise to lowercase for matching
list = lower(list);

% Expand 'auto' to a robust default set (all lowercase)
if numel(list) == 1 && list == "auto"
    list = ["bag","svmrbf","subknn","nbkernel"];
end

catalog = struct('Name',{},'TrainFcn',{});
for s = 1:numel(list)
    nm = lower(char(list(s)));  % ensure lowercase here
    switch nm
        case 'bag'
            % Bagged trees; include class-imbalance cost
            catalog(end+1) = entry('bag', @(X,y) ...
                fitcensemble(X,y,'Method','Bag', ...
                    'Learners',templateTree('Reproducible',true), ...
                    'Cost',classCost(y)));

        case 'svmrbf'
            % ECOC SVM with RBF kernel; classCost handled inside
            catalog(end+1) = entry('svmRBF', @(X,y) trainSVMauto(X,y,'rbf'));

        case 'subknn'
            % kNN with fixed K to avoid 'auto' errors; include cost
            catalog(end+1) = entry('subKNN', @(X,y) ...
                fitcknn(X,y,'NumNeighbors',5,'Standardize',true, ...
                    'Cost',classCost(y)));

        case 'nbkernel'
            % Kernel Naive Bayes; include cost
            catalog(end+1) = entry('nbKernel', @(X,y) ...
                fitcnb(X,y,'DistributionNames','kernel','Support','unbounded', ...
                    'Cost',classCost(y)));

        case 'rusboost'
            % Optional imbalanced-boosting; auto-fallback handled inside
            catalog(end+1) = entry('rusBoost', @(X,y) trainRUSBoostSilenced(X,y));

        otherwise
            warning('Unknown classifier "%s" – skipping.', nm);
    end
end
end

function cost = classCost(y)
% Inverse-frequency misclassification cost matrix for categorical y.
% Heavier penalty for minority classes.
y = removecats(categorical(y));
n = countcats(y);
C = numel(n);
if C < 2
    cost = 0;              % unused, but keep API consistent
    return;
end
w = max(n) ./ n;           % weight for each present class
cost = ones(C) - eye(C);   % off-diagonals = 1 initially
for i = 1:C
    cost(i,:) = cost(i,:) * w(i);  % scale row i by its inverse frequency
    cost(i,i) = 0;
end
end

function e = entry(name, f), e = struct('Name',name,'TrainFcn',f); end

function mdl = trainByName(name, X, y)
switch lower(name)
    case 'bag'
        mdl = fitcensemble(X,y,'Method','Bag','Learners',templateTree('Reproducible',true),'Cost',classCost(y));
    case 'svmrbf'
        mdl = trainSVMauto(X,y,'rbf'); % cost handled inside
    case 'subknn'
        mdl = fitcknn(X,y,'NumNeighbors',5,'Standardize',true,'Cost',classCost(y));
    case 'nbkernel'
        mdl = fitcnb(X,y,'DistributionNames','kernel','Support','unbounded','Cost',classCost(y));
    case 'rusboost'
        mdl = trainRUSBoostSilenced(X,y);
    otherwise
        mdl = fitcensemble(X,y,'Method','Bag','Learners',templateTree('Reproducible',true),'Cost',classCost(y));
end
end

function mdl = safeTrain(entry, X, y)
try
    mdl = entry.TrainFcn(X, y);
catch ME
    warning('Training "%s" failed (%s). Falling back to Bag.', entry.Name, ME.message);
    mdl = fitcensemble(X,y,'Method','Bag','Learners',templateTree('Reproducible',true),'Cost',classCost(y));
end
end

function pr = safePredict(mdl, X, yRef)
try
    pr = predict(mdl, X);
catch
    u = categories(yRef); if isempty(u), pr = yRef; else, pr = repmat(u(1), size(yRef)); end
end
end

function mdl = trainSVMauto(X, y, kernel)
if numel(categories(categorical(y))) > 2
    t = templateSVM('KernelFunction',kernel,'KernelScale','auto','Standardize',true,'BoxConstraint',1);
    mdl = fitcecoc(X,y,'Learners',t,'Coding','onevsone','Cost',classCost(y));
else
    mdl = fitcsvm(X,y,'KernelFunction',kernel,'KernelScale','auto','Standardize',true,'BoxConstraint',1,'Cost',classCost(y));
end
end

function mdl = trainRUSBoostSilenced(X,y)
ws = warning; warning('off','all');
try
    mdl = fitcensemble(X,y,'Method','RUSBoost','Learners',templateTree('Reproducible',true),'NumLearningCycles',200,'Cost',classCost(y));
catch
    mdl = fitcensemble(X,y,'Method','Bag','Learners',templateTree('Reproducible',true),'Cost',classCost(y));
end
warning(ws);
end


%% ========================= Metrics ==========================
function a = balancedAccBinary(y, p)
tp = sum(p=='Fault'   & y=='Fault');
tn = sum(p=='Healthy' & y=='Healthy');
fn = sum(p=='Healthy' & y=='Fault');
fp = sum(p=='Fault'   & y=='Healthy');
tpr = safeDiv(tp, tp+fn); tnr = safeDiv(tn, tn+fp);
a = 0.5*(tpr + tnr);
end

function a = balancedAccMulti(y, p)
cats = categories(y); if isempty(cats), a = NaN; return; end
rc = zeros(numel(cats),1);
for i=1:numel(cats)
    c = cats{i};
    tp = sum(p==c & y==c);
    fn = sum(p~=c & y==c);
    rc(i) = safeDiv(tp, tp+fn);
end
a = mean(rc,'omitnan');
end

function f1 = F1macro(y,p)
cats = categories(y); if isempty(cats), f1 = NaN; return; end
ff = nan(numel(cats),1);
for i=1:numel(cats)
    c = cats{i};
    tp = sum(p==c & y==c);
    fp = sum(p==c & y~=c);
    fn = sum(p~=c & y==c);
    prec = safeDiv(tp, tp+fp); rec = safeDiv(tp, tp+fn);
    ff(i) = 2*safeDiv(prec*rec, (prec+rec));
end
f1 = mean(ff,'omitnan');
end

function m = MCCmulti(y,p)
U = confusionmat(y,p); n = sum(U,'all');
row = sum(U,2); col = sum(U,1);
c = trace(U);
m = (n*c - sum(row.*col)) / sqrt((n^2 - sum(col.^2))*(n^2 - sum(row.^2)));
if ~isfinite(m), m = NaN; end
end

function r = safeDiv(a,b), if b<=0, r = NaN; else, r = a/b; end, end

function k = selectK_generic(hist, opt)
switch lower(opt.KRule)
    case 'threshold'
        mu = max(hist); k0 = find(hist==mu,1,'first');
        thr = opt.TargetFrac * mu;
        k = find(hist >= thr, 1, 'first'); if isempty(k), k = k0; end
    otherwise
        mu = max(hist); k0 = find(hist==mu,1,'first');
        se = std(hist,'omitnan')/sqrt(max(1,numel(hist)));
        thr = mu - se; k = find(hist >= thr, 1, 'first'); if isempty(k), k = k0; end
end
end


%% ===================== Accumulators =========================
function d = initDet()
d.N.allTrue = categorical([]); d.N.allPred = categorical([]); d.N.acc = []; d.N.cls = strings(0,1); d.N.k = []; d.N.peak = [];
d.M.allTrue = categorical([]); d.M.allPred = categorical([]); d.M.acc = []; d.M.cls = strings(0,1); d.M.k = []; d.M.peak = [];
end
function i = initIso()
i.N.allTrue = categorical([]); i.N.allPred = categorical([]);
i.M.allTrue = categorical([]); i.M.allPred = categorical([]);
i.N.acc = []; i.M.acc = []; i.N.f1 = []; i.M.f1 = []; i.N.sen = []; i.M.sen = []; i.N.mcc = []; i.M.mcc = [];
i.N.cls = strings(0,1); i.M.cls = strings(0,1); i.N.k = []; i.M.k = []; i.N.peak = []; i.M.peak = [];
end

function d = addDet(d, y, p, method, cls, k, peak)
a = balancedAccBinary(y,p);
if strcmpi(method,'NDCI')
    d.N.allTrue = [d.N.allTrue; y]; d.N.allPred = [d.N.allPred; p];
    d.N.acc = [d.N.acc; a]; d.N.cls = [d.N.cls; string(cls)]; d.N.k = [d.N.k; k]; d.N.peak = [d.N.peak; peak];
else
    d.M.allTrue = [d.M.allTrue; y]; d.M.allPred = [d.M.allPred; p];
    d.M.acc = [d.M.acc; a]; d.M.cls = [d.M.cls; string(cls)]; d.M.k = [d.M.k; k]; d.M.peak = [d.M.peak; peak];
end
end

function i = addIso(i, y, p, method, cls, k, peak)
a = balancedAccMulti(y,p); f1 = F1macro(y,p); sen = a; mcc = MCCmulti(y,p);
if strcmpi(method,'NDCI')
    i.N.allTrue=[i.N.allTrue; y]; i.N.allPred=[i.N.allPred; p];
    i.N.acc=[i.N.acc; a]; i.N.f1=[i.N.f1; f1]; i.N.sen=[i.N.sen; sen]; i.N.mcc=[i.N.mcc; mcc];
    i.N.cls=[i.N.cls; string(cls)]; i.N.k=[i.N.k; k]; i.N.peak=[i.N.peak; peak];
else
    i.M.allTrue=[i.M.allTrue; y]; i.M.allPred=[i.M.allPred; p];
    i.M.acc=[i.M.acc; a]; i.M.f1=[i.M.f1; f1]; i.M.sen=[i.M.sen; sen]; i.M.mcc=[i.M.mcc; mcc];
    i.M.cls=[i.M.cls; string(cls)]; i.M.k=[i.M.k; k]; i.M.peak=[i.M.peak; peak];
end
end

function out = finalizeDet(d)
out.meanAccN = mean(d.N.acc,'omitnan'); out.stdAccN = std(d.N.acc,'omitnan');
out.meanAccM = mean(d.M.acc,'omitnan'); out.stdAccM = std(d.M.acc,'omitnan');
if out.meanAccN >= out.meanAccM
    out.method = 'NDCI'; out.allPred = d.N.allPred; out.allTrue = d.N.allTrue;
else
    out.method = 'mRMR'; out.allPred = d.M.allPred; out.allTrue = d.M.allTrue;
end
out.perfect = ~isempty(out.allTrue) && all(out.allPred==out.allTrue);
out.summary = struct('NDCI_acc_mean',out.meanAccN,'mRMR_acc_mean',out.meanAccM, ...
                     'NDCI_k_mode',modeOrFirst(d.N.k),'mRMR_k_mode',modeOrFirst(d.M.k), ...
                     'NDCI_cls_mode',modeOrFirstStr(d.N.cls),'mRMR_cls_mode',modeOrFirstStr(d.M.cls));
end

function out = finalizeIso(i)
out.meanAccN = mean(i.N.acc,'omitnan'); out.stdAccN = std(i.N.acc,'omitnan');
out.meanAccM = mean(i.M.acc,'omitnan'); out.stdAccM = std(i.M.acc,'omitnan');
out.meanF1N = mean(i.N.f1,'omitnan'); out.meanF1M = mean(i.M.f1,'omitnan');
out.meanSenN= mean(i.N.sen,'omitnan'); out.meanSenM= mean(i.M.sen,'omitnan');
out.meanMCCN= mean(i.N.mcc,'omitnan'); out.meanMCCM= mean(i.M.mcc,'omitnan');
if out.meanAccN >= out.meanAccM
    out.method = 'NDCI'; out.allPred = i.N.allPred; out.allTrue = i.N.allTrue;
else
    out.method = 'mRMR'; out.allPred = i.M.allPred; out.allTrue = i.M.allTrue;
end
out.summary = struct( ...
    'NDCI_acc_mean',out.meanAccN,'mRMR_acc_mean',out.meanAccM, ...
    'NDCI_F1_mean',out.meanF1N,'mRMR_F1_mean',out.meanF1M, ...
    'NDCI_Sen_mean',out.meanSenN,'mRMR_Sen_mean',out.meanSenM, ...
    'NDCI_MCC_mean',out.meanMCCN,'mRMR_MCC_mean',out.meanMCCM, ...
    'NDCI_k_mode',modeOrFirst(i.N.k),'mRMR_k_mode',modeOrFirst(i.M.k), ...
    'NDCI_cls_mode',modeOrFirstStr(i.N.cls),'mRMR_cls_mode',modeOrFirstStr(i.M.cls));
end

function x = modeOrFirst(v), if isempty(v), x = NaN; else, try, x = mode(v); catch, x = v(1); end, end, end
function s = modeOrFirstStr(v), if isempty(v), s=""; else, try, s = mode(v); catch, s = v(1); end, end, end


%% ======================== Plots & I/O =======================
function saveRankingPlot(sub, sensors, ndci, SP, S, U, ranked_mRMR)
fig = figure('Visible','off','Position',[100 100 1100 450]);
tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');

[~, ordN] = sort(ndci,'descend');
nexttile;
barh([SP(ordN)' S(ordN)' U(ordN)'],'stacked'); grid on;
set(gca,'YTick',1:numel(sensors),'YTickLabel', sensors(ordN));
title(sprintf('%s — NDCI components (SP,S,U)', sub));
xlabel('Normalised contribution'); legend({'SP','S','U'},'Location','southoutside','Orientation','horizontal');

nexttile;
barh(categorical(ranked_mRMR(end:-1:1)), 1:numel(ranked_mRMR));
title(sprintf('%s — mRMR/ANOVA ranking (best at top)', sub));
xlabel('Rank order'); grid on;

exportgraphics(fig, sprintf('ranking_%s.png', sub));
close(fig);
end

function plotStepwiseCombined(sub, stepHistN, stepHistM)
L = max(cellfun(@numel, stepHistN));
HN = NaN(numel(stepHistN), L); HM = NaN(numel(stepHistM), L);
for i=1:numel(stepHistN), HN(i,1:numel(stepHistN{i})) = stepHistN{i}; end
for i=1:numel(stepHistM), HM(i,1:numel(stepHistM{i})) = stepHistM{i}; end
muN = nanmean(HN,1); sdN = nanstd(HN,0,1);
muM = nanmean(HM,1); sdM = nanstd(HM,0,1);

x = 1:L;
fig = figure('Visible','off','Position',[120 120 820 520]); hold on; grid on;
fill([x fliplr(x)], [(muN-sdN)*100 fliplr((muN+sdN)*100)], [0.8 0.9 1.0], 'EdgeColor','none','FaceAlpha',0.35);
fill([x fliplr(x)], [(muM-sdM)*100 fliplr((muM+sdM)*100)], [1.0 0.9 0.8], 'EdgeColor','none','FaceAlpha',0.35);
plot(x, muN*100, '-o', 'LineWidth',2,'MarkerSize',5);
plot(x, muM*100, '-s', 'LineWidth',2,'MarkerSize',5);
ylabel('Accuracy (%)'); xlabel('Sensors');
title(sprintf('Stepwise Accuracy — %s (nested CV mean ± std)', sub));
legend({'NDCI ±1σ','mRMR ±1σ','NDCI mean','mRMR mean'},'Location','SouthEast');
ylim([0 110]);
exportgraphics(fig, sprintf('stepwise_%s_combined.png', sub));
close(fig);
end

function saveConfusion(sub, method, yTrue, yPred)
fig = figure('Visible','off');
confusionchart(yTrue, yPred, ...
    'Title', sprintf('%s – %s (nested, aggregated)', sub, method), ...
    'RowSummary','row-normalized','ColumnSummary','column-normalized');
exportgraphics(fig, sprintf('confusion_isolation_%s_%s.png', sub, regexprep(method,'\\W','')));
close(fig);
end

function saveFinalComparison(results, subs)
fig = figure('Visible','off','Position',[100 100 900 600]);
accN = zeros(1,numel(subs)); lbl = strings(1,numel(subs));
for k=1:numel(subs)
    if isfield(results, subs{k}) && isfield(results.(subs{k}),'iso') && ~isempty(results.(subs{k}).iso.allTrue)
        accN(k) = results.(subs{k}).iso.meanAccN * 100;
        lbl(k)  = sprintf('%s', subs{k});
    end
end
bar(accN); ylim([0 110]); grid on; xticklabels(lbl);
ylabel('Isolation Balanced Accuracy (%)'); title('Final Sensor Suite Performance (NDCI, nested)');
text(1:numel(subs), accN, compose('%.1f%%',accN), ...
     'HorizontalAlignment','center','VerticalAlignment','bottom');
exportgraphics(fig, 'final_comparison_nested_pro.png');
close(fig);
end


%% ===================== Small helpers ========================
function Z = zscoreSafe(X)
mu = mean(X,1,'omitnan'); sg = std(X,0,1,'omitnan'); sg(~isfinite(sg)|sg<eps)=1; Z = (X - mu)./sg;
end
function y = rescaleSafe(x)
% Rescale vector x to [0,1], omitting NaNs and guarding degenerate ranges.
a = min(x,[],'omitnan');
b = max(x,[],'omitnan');
if ~isfinite(a) || ~isfinite(b)
    y = zeros(size(x)); return;
end
d = b - a;
if d < eps
    y = zeros(size(x));
else
    y = (x - a) ./ d;
end
% Replace any non-finite outputs with 0
y(~isfinite(y)) = 0;
end

function rec = collectComp(stage, clfName, rowK)
[peak, k] = max(rowK); rec = struct('stage',stage,'classifier',string(clfName),'peakAcc',peak,'kAtPeak',k);
end
function rows = addCompRows(sub, r, fo, method, tblDet, tblIso)
rows = [];
if ~isempty(tblDet)
    for i=1:height(tblDet)
        rows = [rows; struct('subsystem',string(sub),'repeat',r,'fold',fo,'method',string(method), ...
               'task',"det",'classifier',tblDet.classifier(i),'peakAcc',tblDet.peakAcc(i),'kAtPeak',tblDet.kAtPeak(i))];
    end
end
if ~isempty(tblIso)
    for i=1:height(tblIso)
        rows = [rows; struct('subsystem',string(sub),'repeat',r,'fold',fo,'method',string(method), ...
               'task',"iso",'classifier',tblIso.classifier(i),'peakAcc',tblIso.peakAcc(i),'kAtPeak',tblIso.kAtPeak(i))];
    end
end
end
function X = tableCols(T, ranked, k)
% Robust column extraction by names for first k items
if isstring(ranked), ranked = cellstr(ranked); end
if iscell(ranked), ranked = ranked(:)'; else, ranked = {ranked}; end
want = ranked(1:min(k, numel(ranked)));
want = intersect(want, T.Properties.VariableNames, 'stable');
if isempty(want), X = zeros(height(T),0); else, X = T{:, want}; end
end
function [ranked, scores] = rank_mRMR_onFaultTbl(Ftr, ytr, sensors)
X = table2array(Ftr); ranked = sensors; scores = zeros(1, numel(sensors));
try
    if exist('fscmrmr','file') == 2
        [idx, scr] = fscmrmr(X, ytr); ranked = sensors(idx); scores = scr(idx); return;
    end
catch, end
F = nan(1, size(X,2));
for j=1:size(X,2)
    try, [~,tbl] = anova1(X(:,j), ytr, 'off'); F(j) = tbl{2,5}; catch, F(j) = 0; end
end
[~, idx] = sort(F,'descend'); ranked = sensors(idx); scores = F(idx);
end
