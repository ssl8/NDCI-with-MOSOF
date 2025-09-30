function results = analysis_pipeline_nested(varargin)
% ANALYSIS_PIPELINE_NESTED
% Nested CV comparison of NDCI vs mRMR…

%% -------- options --------
p = inputParser;
p.addOptional('excelFile','DataB.xlsx',@(s)ischar(s)||isstring(s));
p.addParameter('Repeats',10,@(x)isnumeric(x)&&x>=1);
p.addParameter('OuterFolds',5,@(x)isnumeric(x)&&x>=2);
p.addParameter('InnerFolds',3,@(x)isnumeric(x)&&x>=2);
p.addParameter('CorrThr',0.995,@(x)isnumeric(x)&&x>0&&x<1);
p.addParameter('Seed',42,@isnumeric);
p.parse(varargin{:});
opt = p.Results;

rng(opt.Seed);
excelFile = char(opt.excelFile);
if ~isfile(excelFile), error('Input Excel file not found: %s', excelFile); end

%% -------- groups & faults (same names you’ve been using) --------
sensorGroups.Engine = { ...
    'W_S1','W_S2','W_S21','W_S24','W_S3','W_S4','W_S45','W_S5','W_S7', ...
    'h_S1','h_S2','h_S21','h_S24','h_S3','h_S4','h_S45','h_S5','h_S7','h_S17', ...
    'Tt_S1','Tt_S2','Tt_S21','Tt_S24','Tt_S3','Tt_S4','Tt_S45','Tt_S5','Tt_S7','Tt_S17', ...
    'Pt_S1','Pt_S21','Pt_S3','Pt_S4','Pt_S45','Pt_S5','FanTrq','LPCTrq','HPCTrq','HPTTrq','LPTTrq','Thrust', ...
    'h_Fan','Tt_Fan','Pt_Fan','Wf','FS_Motor_Torque','Power_FSPump', ...
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

%% -------- load & constant-prune only --------
T = readtable(excelFile);
T.Properties.VariableNames = matlab.lang.makeValidName(T.Properties.VariableNames);
fprintf('--- Data Loaded and Sanitized (%s) ---\n', excelFile);

[Tclean, sensorGroups] = preprocessConstantsOnly(T, sensorGroups);
fprintf('--- Preprocessing (constants only; redundancy nested per subsystem) ---\n');

%% -------- nested CV per subsystem --------
subs = fieldnames(sensorGroups);
results = struct(); manifest = struct();

for i = 1:numel(subs)
    sub = subs{i}; sensorsAll = sensorGroups.(sub);
    if isempty(sensorsAll), continue; end
    switch sub
        case 'Engine', subFM = FM.Engine;
        case 'Fuel',   subFM = FM.Fuel;
        case 'Elec',   subFM = FM.Elec;
        case 'ECS',    subFM = FM.ECS;
    end

    Tsub = Tclean(ismember(Tclean.FaultMode, subFM) | strcmp(Tclean.FaultMode,'Healthy'), :);
    fprintf('\n--- %s ---\n', sub);
    fprintf('Rows: %d, Candidate sensors: %d\n', height(Tsub), numel(sensorsAll));

    % Overall (illustrative) ranking plot on full Tsub (not used for eval)
    [ndciAll, SP, S, U, fDataAll, labelsAll] = calculateNDCI(Tsub, sensorsAll);
    [~, idxN] = sort(ndciAll,'descend');
    [sMAll, ~] = rank_mRMR_or_ANOVA(fDataAll, labelsAll, sensorsAll);
    saveRankingPlotNested(sub, sensorsAll, ndciAll, SP, S, U, sMAll);

    % Nested evaluation
    [foldRes, stepHistN, stepHistM] = nestedEvaluate(Tsub, sensorsAll, opt);
    results.(sub) = foldRes;

    % Stepwise mean ± std across folds (align lengths)
    saveStepwiseNested(sub, stepHistN, stepHistM);

    % Aggregated confusion (choose better)
    if foldRes.meanAccN >= foldRes.meanAccM
        saveConfusionNested(sub, 'NDCI', foldRes.allTrue, foldRes.allPredN);
    else
        saveConfusionNested(sub, 'mRMR', foldRes.allTrue, foldRes.allPredM);
    end

    % Manifest info
    manifest.(sub) = rmfield(foldRes, {'allTrue','allPredN','allPredM'});
end

% Final bar comparison (NDCI means)
saveFinalComparisonNested(results, subs);

save('run_manifest_nested.mat','results','manifest','opt');

end % main


%% ========================== Helpers ==========================

function [Tclean, groups] = preprocessConstantsOnly(T, groups)
    allS   = unique([groups.Engine, groups.Fuel, groups.Elec, groups.ECS], 'stable');
    existS = intersect(allS, T.Properties.VariableNames, 'stable');
    v = var(T{:, existS}, 0, 1, 'omitnan');
    const = existS(v < 1e-6);
    if ~isempty(const), fprintf('Remove const: %s\n', strjoin(const,',')); end
    fprintf('Remove red: (deferred to nested per-subsystem)\n');
    Tclean = T; Tclean(:, const) = [];
    fn = fieldnames(groups);
    for i=1:numel(fn)
        groups.(fn{i}) = intersect(groups.(fn{i}), Tclean.Properties.VariableNames, 'stable');
    end
end


function [ndci, SPn, Sn, Un, fData, labels, baseline] = calculateNDCI(Tsub, sensors)
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
        z = zeros(1, numel(sensors));
        ndci=z; SPn=z; Sn=z; Un=z; fData = array2table([], 'VariableNames', sensors); return;
    end

    SP_raw = mean(abs(Xf - baseline)./r, 1, 'omitnan');
    denom  = max(1 - sevF, eps);
    S_raw  = mean(abs(Xf - baseline)./denom, 1, 'omitnan');

    Z  = Xf; mu = mean(Z,1,'omitnan'); sg = std(Z,0,1,'omitnan'); sg(~isfinite(sg)|sg<eps)=1;
    Z  = (Z - mu)./sg;
    C  = corrcoef(Z,'Rows','pairwise');
    if any(isnan(C),'all'), C(1:size(C,1)+1:end)=1; C(~isfinite(C))=0; end
    D  = 1 - abs(C); D(1:size(D,1)+1:end)=NaN;
    U_raw = nanmean(D,2)';

    SPn = (SP_raw - min(SP_raw)) / max(max(SP_raw)-min(SP_raw), eps);
    Sn  = (S_raw  - min(S_raw )) / max(max(S_raw )-min(S_raw ), eps);
    Un  = (U_raw  - min(U_raw )) / max(max(U_raw )-min(U_raw ), eps);
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


function [foldRes, stepHistN, stepHistM] = nestedEvaluate(Tsub, sensorsAll, opt)
% Repeats × outer folds; per fold:
%  - compute NDCI on TRAIN (healthy baseline + fault-only features)
%  - prune redundant sensors on TRAIN (fault rows only)
%  - rank (NDCI & mRMR) on TRAIN faults
%  - inner CV on TRAIN faults to pick k*
%  - train on TRAIN faults, test on TEST faults

allTrue  = categorical([]); 
allPredN = categorical([]); 
allPredM = categorical([]);
kN_list = []; kM_list = []; 
accN = []; accM = [];
stepHistN = {}; stepHistM = {};

for r = 1:opt.Repeats
    rng(opt.Seed + r);
    % Stratify outer CV by FaultMode (includes "Healthy")
    yAll  = categorical(Tsub.FaultMode);
    Kout  = min(opt.OuterFolds, max(2, min(countcats(yAll))));
    outer = cvpartition(yAll,'KFold',Kout);

    for fo = 1:outer.NumTestSets
        trMask = training(outer,fo); 
        teMask = test(outer,fo);

        Train = Tsub(trMask,:);        % TRAIN: healthy + faults
        Test  = Tsub(teMask,:);        % TEST:  healthy + faults

        % ---- TRAIN-ONLY NDCI & pruning ----
        % calculateNDCI uses healthy rows (if present) for baseline
        % and returns fault-only table Ftr + ytr labels.
        [ndci, SP, S, U, Ftr, ytr] = calculateNDCI(Train, sensorsAll); % Ftr: TRAIN faults only
        if height(Ftr) < 2 || numel(categories(ytr)) < 2
            % Not enough training faults or classes — skip this fold
            continue;
        end

        % Remove NaN-scored sensors and prune redundancy on TRAIN faults
        valid  = ~isnan(ndci);
        sensors = sensorsAll(valid);
        ndci    = ndci(valid);  SP = SP(valid); S = S(valid); U = U(valid);
        Ftr     = Ftr(:, valid);

        [sensors, keepMask, Ftr] = pruneRedundantSensors(sensors, Ftr, ndci, opt.CorrThr);
        ndci = ndci(keepMask); SP = SP(keepMask); S = S(keepMask); U = U(keepMask);

        % Rankings on TRAIN only
        [~, idxN] = sort(ndci,'descend'); sN = sensors(idxN);
        [sM, ~]   = rank_mRMR_or_ANOVA(Ftr, ytr, sensors);

        % ---- INNER CV on TRAIN faults to pick k* ----
        histN = innerHist(Ftr, ytr, sN, opt.InnerFolds);
        histM = innerHist(Ftr, ytr, sM, opt.InnerFolds);

        if all(isnan(histN)), histN(:) = 0; end
        if all(isnan(histM)), histM(:) = 0; end

        stepHistN{end+1} = histN; 
        stepHistM{end+1} = histM;

        kN = selectK(histN); 
        kM = selectK(histM);

        % ---- TRAIN on TRAIN faults ----
        cost = classCost(ytr);
        XtrN = Ftr{:, sN(1:kN)};
        XtrM = Ftr{:, sM(1:kM)};
        mdlN = fitcensemble(XtrN, ytr, 'Method','Bag', ...
               'Learners', templateTree('Reproducible',true), 'Cost', cost);
        mdlM = fitcensemble(XtrM, ytr, 'Method','Bag', ...
               'Learners', templateTree('Reproducible',true), 'Cost', cost);

        % ---- TEST on TEST faults only ----
        TestF = Test(Test.Severity > 0, :);
        if isempty(TestF)     % no faults in this held-out fold → skip
            continue;
        end
        yte   = categorical(TestF.FaultMode);
        XteN  = TestF{:, sN(1:kN)};
        XteM  = TestF{:, sM(1:kM)};
        predN = predict(mdlN, XteN);
        predM = predict(mdlM, XteM);

        % ---- accumulate ----
        allTrue  = [allTrue;  yte];
        allPredN = [allPredN; predN];
        allPredM = [allPredM; predM];
        accN(end+1) = mean(predN == yte);
        accM(end+1) = mean(predM == yte);
        kN_list(end+1) = kN; 
        kM_list(end+1) = kM;
    end
end

% Summaries
foldRes.meanAccN = mean(accN);   foldRes.stdAccN = std(accN);
foldRes.meanAccM = mean(accM);   foldRes.stdAccM = std(accM);
foldRes.kN_mean  = mean(kN_list); foldRes.kN_mode = mode(kN_list);
foldRes.kM_mean  = mean(kM_list); foldRes.kM_mode = mode(kM_list);
foldRes.allTrue  = allTrue; 
foldRes.allPredN = allPredN; 
foldRes.allPredM = allPredM;
end



function hist = innerHist(Ftr, ytr, rankedSensors, K)
    % Robust inner CV:
    % - removes unused categories
    % - adapts K so each fold has >=1 sample per class when possible
    % - falls back to non-stratified 2-fold if needed

    if nargin < 4, K = 3; end
    ytr = removecats(categorical(ytr));        % <-- drop absent classes
    C   = numel(categories(ytr));
    N   = numel(ytr);

    % If we cannot form a meaningful CV, return NaNs (caller will still pick k by max)
    if C < 2 || N < 4
        hist = nan(1, numel(rankedSensors));
        return;
    end

    % Pick a fold count that keeps at least one sample per class
    minPerClass = min(countcats(ytr));
    Ksafe = min([K, minPerClass, N-1]);        % cannot exceed N-1
    if Ksafe < 2
        try
            inner = cvpartition(N, 'KFold', 2);        % non-stratified fallback
            stratified = false;
        catch
            hist = nan(1, numel(rankedSensors)); return;
        end
    else
        try
            inner = cvpartition(ytr, 'KFold', Ksafe);   % stratified
            stratified = true;
        catch
            inner = cvpartition(N, 'KFold', 2);        % last resort
            stratified = false;
        end
    end

    Kmax = numel(rankedSensors);
    hist = zeros(1, Kmax);

    for k = 1:Kmax
        feat = rankedSensors(1:k);
        X    = Ftr{:, feat};
        acc  = nan(1, inner.NumTestSets);
        for fi = 1:inner.NumTestSets
            tri = training(inner, fi);
            tei = test(inner, fi);

            % When using stratified cv on categorical, tri/tei align with ytr
            yt = removecats(ytr(tri));                 % drop unused *in this fold*
            ye = ytr(tei);
            Xt = X(tri,:); Xe = X(tei,:);

            if numel(categories(yt)) < 2 || size(Xt,1) < 2
                continue;                               % skip this split
            end

            cost = classCost(yt);                       % << safe size
            mdl  = fitcensemble(Xt, yt, 'Method','Bag', ...
                   'Learners', templateTree('Reproducible', true), ...
                   'Cost', cost);
            acc(fi) = mean(predict(mdl, Xe) == ye);
        end
        hist(k) = nanmean(acc);                         % NaN-safe average
    end
end


function k = selectK(hist)
    [mu, k0] = max(hist);
    se = std(hist)/sqrt(max(1,numel(hist)));   % crude one-SE rule
    thr = mu - se;
    k = find(hist >= thr, 1, 'first'); if isempty(k), k = k0; end
end


function cost = classCost(y)
    y = removecats(categorical(y));                     % only present classes
    n = countcats(y);
    C = numel(n);
    if C < 2
        cost = 0;                                       % unused, but well-defined
        return;
    end
    w = max(n) ./ n;                                    % inverse-frequency weights
    cost = ones(C) - eye(C);
    for i = 1:C
        cost(i,:) = cost(i,:) * w(i);
        cost(i,i) = 0;
    end
end


%% ---------- Plotting & summaries ----------

function saveRankingPlotNested(sub, sensors, ndci, SP, S, U, ranked_mRMR)
    fig = figure('Visible','off','Position',[100 100 1100 450]);
    t = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');

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

    exportgraphics(fig, sprintf('ranking_nested_%s.png', sub));
    close(fig);
end


function saveStepwiseNested(sub, stepHistN, stepHistM)
    % Align lengths by padding with NaN, then plot mean±std (ignore NaN)
    L = max(cellfun(@numel, stepHistN));
    HN = NaN(numel(stepHistN), L); HM = NaN(numel(stepHistM), L);
    for i=1:numel(stepHistN), HN(i,1:numel(stepHistN{i})) = stepHistN{i}; end
    for i=1:numel(stepHistM), HM(i,1:numel(stepHistM{i})) = stepHistM{i}; end
    muN = nanmean(HN,1); sdN = nanstd(HN,0,1);
    muM = nanmean(HM,1); sdM = nanstd(HM,0,1);

    x = 1:L;
    fig = figure('Visible','off'); hold on; grid on;
    % shaded areas
    f = fill([x fliplr(x)], [(muN-sdN)*100 fliplr((muN+sdN)*100)], [0.8 0.9 1.0], 'EdgeColor','none','FaceAlpha',0.4);
    f2= fill([x fliplr(x)], [(muM-sdM)*100 fliplr((muM+sdM)*100)], [1.0 0.9 0.8], 'EdgeColor','none','FaceAlpha',0.4);
    plot(x, muN*100, '-o', 'LineWidth',2,'MarkerSize',5);
    plot(x, muM*100, '-s', 'LineWidth',2,'MarkerSize',5);
    ylabel('Accuracy (%)'); xlabel('Sensors');
    title(sprintf('Stepwise Accuracy — %s (nested CV mean ± std)', sub));
    legend({'NDCI ±1σ','mRMR ±1σ','NDCI mean','mRMR mean'},'Location','SouthEast');
    ylim([0 110]);
    exportgraphics(fig, sprintf('stepwise_nested_%s.png', sub));
    close(fig);
end


function saveConfusionNested(sub, method, trueLabels, preds)
    fig = figure('Visible','off');
    confusionchart(trueLabels, preds, ...
        'Title', sprintf('%s – %s (nested, aggregated)', sub, method), ...
        'RowSummary','row-normalized','ColumnSummary','column-normalized');
    exportgraphics(fig, sprintf('confusion_nested_%s_%s.png', method, sub));
    close(fig);
end


function saveFinalComparisonNested(results, subs)
    fig = figure('Visible','off','Position',[100 100 900 600]);
    accN = zeros(1,numel(subs)); lbl = strings(1,numel(subs));
    for k=1:numel(subs)
        if isfield(results, subs{k})
            accN(k) = results.(subs{k}).meanAccN * 100;
            lbl(k)  = sprintf('%s', subs{k});
        end
    end
    bar(accN); ylim([0 110]); grid on; xticklabels(lbl);
    ylabel('Accuracy (%)'); title('Final Sensor Suite Performance (NDCI, nested)');
    text(1:numel(subs), accN, compose('%.1f%%',accN), ...
         'HorizontalAlignment','center','VerticalAlignment','bottom');
    exportgraphics(fig, 'final_comparison_nested.png');
    close(fig);
end
