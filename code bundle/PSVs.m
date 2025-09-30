%% FMECA_Analysis.m
% This script performs a Failure Modes, Effects, and Criticality Analysis (FMECA)
% on an aircraft dataset stored in an Excel file named '3re.xlsx'. The file is 
% assumed to have:
%   - Columns 1 to 36: Sensor readings (with the first row containing sensor names).
%   - Column 37: Fault Mode (FM) identifiers (e.g., 'FM1', 'FM2', etc.).
%   - Column 38: Severity values (expressed as decimals, e.g., 0.2 means 20% degradation).
%
% The analysis includes:
%   1. Loading and preprocessing the data (including conversion of FM strings,
%      missing data handling, and z-score normalization).
%   2. Mapping sensors to subsystems and fault modes to descriptive names.
%   3. Calculating, for each fault mode, the frequency of occurrence, average severity,
%      and Risk Priority Number (RPN).
%   4. Visualizing the results with bar charts and sensor “fault signature” plots.
%
% The script is modular with separate functions for preprocessing, mapping, FMECA 
% calculation, and visualization.

%% Main Script

clear; clc; close all;

% Define Excel file name
filename = 'DataB.xlsx';

% Load and preprocess data
try
    [sensorDataNorm, sensorNames, faultModes, severities] = preprocessData(filename);
catch ME
    disp('Error loading or preprocessing data:');
    disp(ME.message);
    return;
end

% Define fault mode mappings and subsystem sensor mappings
[fmMapping, fmDescriptions, subsystemMapping] = defineFaultModeMappings();

% Perform FMECA analysis:
% For each fault mode (FM1 to FM25), calculate the frequency of occurrence 
% (percentage of samples where that FM appears), compute the average severity,
% and calculate a Risk Priority Number (RPN = Frequency x Average Severity x Detection,
% with default detection assumed as 1).
fmSummary = performFMECA(faultModes, severities, sensorDataNorm, fmMapping);

% Display summary table in the command window
disp('FMECA Summary Table:');
disp(fmSummary);

% Visualize FMECA results (frequency, severity, and RPN bar charts)
visualizeFMECA(fmSummary);

% Plot fault signatures for each subsystem (average sensor reading profiles)
plotFaultSignatures(sensorDataNorm, faultModes, sensorNames, fmMapping, fmDescriptions, subsystemMapping);



%%%%%

plotNDClikeSeverityProfilesGrouped(...
    sensorDataNorm, ...
    faultModes, ...
    severities, ...
    sensorNames, ...
    fmDescriptions, ...
    fmMapping, ...
    subsystemMapping);


%% Function Definitions

function [sensorDataNorm, sensorNames, faultModes, severities] = preprocessData(filename)
    % preprocessData loads the Excel file and preprocesses the data.
    %   [sensorDataNorm, sensorNames, faultModes, severities] = preprocessData(filename)
    %
    %   - Reads the file using readtable.
    %   - Extracts sensor names (columns 1 to 36).
    %   - Converts sensor readings to a numeric matrix.
    %   - Handles missing values by replacing them with the column mean.
    %   - Applies z-score normalization to the sensor data.
    %   - Converts fault mode strings (e.g., 'FM1') to numeric values.
    %   - Converts severity values to numeric if needed.
    
    % Read Excel file with variable names preserved
    try
        dataTable = readtable(filename, 'PreserveVariableNames', true);
    catch ME
        error('Failed to read the Excel file: %s', ME.message);
    end
    
    % Verify the file has at least 38 columns
    if width(dataTable) < 105
        error('The Excel file does not contain the expected 38 columns.');
    end
    
    % Extract sensor names from the first 36 columns (assumed to be in the header)
    sensorNames = dataTable.Properties.VariableNames(1:105);
    
    % Extract sensor data and convert to numeric matrix
    sensorData = table2array(dataTable(:, 1:105));
    
    % Handle missing data: replace NaNs with the column mean
    for i = 1:size(sensorData,2)
        col = sensorData(:,i);
        if any(isnan(col))
            col(isnan(col)) = nanmean(col);
            sensorData(:,i) = col;
        end
    end
    
    % Normalize sensor data using z-score normalization
    sensorDataNorm = zscore(sensorData);
    
    % Extract fault modes (column 37) and severity values (column 38)
    faultModes = table2array(dataTable(:, 107));
    severities = table2array(dataTable(:, 106));
    
    % Convert faultModes to numeric if they are stored as strings (e.g., 'FM1', 'FM2')
    if ~isnumeric(faultModes)
        % If faultModes is a cell array or string array, convert each element
        faultModesNumeric = zeros(length(faultModes), 1);
        for i = 1:length(faultModes)
            if iscell(faultModes)
                currentFM = faultModes{i};
            else
                currentFM = faultModes(i);
            end
            % Remove 'FM' (case insensitive) from the string and convert to number
            if ischar(currentFM) || isstring(currentFM)
                numStr = erase(string(currentFM), 'FM');
                faultModesNumeric(i) = str2double(numStr);
            else
                error('Unexpected type for faultModes.');
            end
        end
        faultModes = faultModesNumeric;
    end
    
    % Convert severities to numeric if necessary
    if ~isnumeric(severities)
        if iscell(severities)
            severities = cellfun(@str2double, severities);
        elseif isstring(severities)
            severities = str2double(severities);
        else
            error('Unexpected type for severities.');
        end
    end
    
    % Validate that faultModes and severities are now numeric
    if ~isnumeric(faultModes) || ~isnumeric(severities)
        error('Fault Modes and Severities must be numeric.');
    end
end

function [fmMapping, fmDescriptions, subsystemMapping] = defineFaultModeMappings()
    % defineFaultModeMappings creates mappings for fault modes to subsystems and
    % assigns descriptive names for each fault mode.
    %
    % Outputs:
    %   fmMapping - A containers.Map that maps fault mode numbers (1-25) to subsystem names.
    %   fmDescriptions - A containers.Map that maps fault mode numbers to descriptive strings.
    %   subsystemMapping - A structure that maps subsystem names to the sensor indices.
    
    % Define real descriptive names for fault modes FM1 to FM25
    fmDescriptions = containers.Map('KeyType', 'double', 'ValueType', 'char');
    fmDescriptions(1)  = 'AC Motor Fault (FS Motor)';
    fmDescriptions(2)  = 'FS Nozzle Switch Open';
    fmDescriptions(3)  = 'FS Valve Switch Open';
    fmDescriptions(4)  = 'Engine Bleed Valve Stuck Open';
    fmDescriptions(5)  = 'ACS TCV Failure Closed';
    fmDescriptions(6)  = 'AC Lamp InstrSw Open';
    fmDescriptions(7)  = 'AC Motor Switch Open';
    fmDescriptions(8)  = 'Pump External Leakage (DPV1)';
    fmDescriptions(9)  = 'Internal Pump Leakage (DPV2)';
    fmDescriptions(10) = 'FOHE Clogging (DPV3)';
    fmDescriptions(11) = 'FOHE Leakage (DPV4)';
    fmDescriptions(12) = 'Fuel Nozzle Clogging (DPV5)';
    fmDescriptions(13) = 'Reduced Pump RPM';
    fmDescriptions(14) = 'LPT Blade Broken';
    fmDescriptions(15) = 'LPC Fouling';
    fmDescriptions(16) = 'HPC Fouling';
    fmDescriptions(17) = 'Fan FOD';
    fmDescriptions(18) = 'HPC Blade Broken';
    fmDescriptions(19) = 'HPC Seizure';
    fmDescriptions(20) = 'HPC Stall';
    fmDescriptions(21) = 'Primary Heat Exchanger (PHX) Fouling';
    fmDescriptions(22) = 'Secondary Heat Exchanger (SHX) Fouling';
    fmDescriptions(23) = 'SHX Blockage of Cold Mass Flow';
    fmDescriptions(24) = 'Air Cycle Machine Efficiency';
    fmDescriptions(25) = 'RAM Mass Flow Blockage';
    
    % Map fault modes to subsystems:
    % FM1-FM7  -> Elec
    % FM8-FM13 -> Fuel
    % FM14-FM20 -> Engine
    % FM21-FM25 -> ECS
    fmMapping = containers.Map('KeyType', 'double', 'ValueType', 'char');
    for fm = 1:25
        if fm >= 1 && fm <= 7
            fmMapping(fm) = 'Elec';
        elseif fm >= 8 && fm <= 13
            fmMapping(fm) = 'Fuel';
        elseif fm >= 14 && fm <= 20
            fmMapping(fm) = 'Engine';
        elseif fm >= 21 && fm <= 25
            fmMapping(fm) = 'ECS';
        end
    end
    
    % Define sensor indices for each subsystem:
    % ECS: Sensors 1 to 9
    % Engine: Sensors 10 to 16
    % Elec: Sensors 17 to 26
    % Fuel: Sensors 27 to 36
    subsystemMapping = struct();
    subsystemMapping.ECS = 80:105;
    subsystemMapping.Engine = 9:79;
    subsystemMapping.Elec = 88:98;
    subsystemMapping.Fuel = 1:8;


end

function fmSummary = performFMECA(faultModes, severities, sensorData, fmMapping)
    % performFMECA calculates the frequency, average severity, and RPN for each fault mode.
    %
    % Inputs:
    %   faultModes - Vector containing the fault mode number for each sample.
    %   severities - Vector containing the severity (in decimals) for each sample.
    %   sensorData - Normalized sensor data matrix (used here if further analysis is needed).
    %   fmMapping  - Map from fault mode numbers to the associated subsystem.
    %
    % Output:
    %   fmSummary  - A table summarizing, for each fault mode, the fault mode label,
    %                associated subsystem, frequency of occurrence (as a percentage),
    %                average severity (as a percentage), and the computed RPN.
    
    numFaultModes = 25;
    totalSamples = length(faultModes);
    
    % Initialize storage arrays
    fmNumbers = cell(numFaultModes, 1);
    subsystems = cell(numFaultModes, 1);
    frequencies = zeros(numFaultModes, 1);
    avgSeverities = zeros(numFaultModes, 1);
    RPNs = zeros(numFaultModes, 1);
    
    defaultDetection = 1; % Assumed default detection rating
    
    for fm = 1:numFaultModes
        % Create fault mode label (e.g., 'FM1', 'FM2', etc.)
        fmNumbers{fm} = ['FM' num2str(fm)];
        
        % Determine the subsystem from mapping
        if isKey(fmMapping, fm)
            subsystems{fm} = fmMapping(fm);
        else
            subsystems{fm} = 'Unknown';
        end
        
        % Identify all samples with the current fault mode
        indices = find(faultModes == fm);
        
        % Compute frequency as the percentage of total samples
        frequencies(fm) = (length(indices) / totalSamples) * 100;
        
        % Compute average severity (convert decimal to percentage)
        if ~isempty(indices)
            avgSeverities(fm) = mean(severities(indices)) * 100;
        else
            avgSeverities(fm) = 0;
        end
        
        % Calculate Risk Priority Number (RPN)
        RPNs(fm) = frequencies(fm) * avgSeverities(fm) * defaultDetection;
    end
    
    % Create a summary table with the results
    fmSummary = table(fmNumbers, subsystems, frequencies, avgSeverities, RPNs, ...
        'VariableNames', {'FaultMode', 'Subsystem', 'Frequency_Percent', 'AvgSeverity_Percent', 'RPN'});
end

function visualizeFMECA(fmSummary)
    % visualizeFMECA generates bar charts for fault mode frequency, average severity, and RPN.
    %
    % Input:
    %   fmSummary - Table containing FMECA analysis results.
    
    faultLabels = fmSummary.FaultMode;
    
    % Bar chart for frequency
    figure;
    bar(fmSummary.Frequency_Percent);
    set(gca, 'XTick', 1:length(faultLabels), 'XTickLabel', faultLabels);
    xlabel('Fault Mode');
    ylabel('Frequency (%)');
    title('Fault Mode Frequency');
    grid on;
    
    % Bar chart for average severity
    figure;
    bar(fmSummary.AvgSeverity_Percent);
    set(gca, 'XTick', 1:length(faultLabels), 'XTickLabel', faultLabels);
    xlabel('Fault Mode');
    ylabel('Average Severity (%)');
    title('Fault Mode Average Severity');
    grid on;
    
    % Bar chart for RPN
    figure;
    bar(fmSummary.RPN);
    set(gca, 'XTick', 1:length(faultLabels), 'XTickLabel', faultLabels);
    xlabel('Fault Mode');
    ylabel('Risk Priority Number (RPN)');
    title('Fault Mode RPN');
    grid on;
end

function plotFaultSignatures(sensorData, faultModes, sensorNames, fmMapping, fmDescriptions, subsystemMapping)
    % plotFaultSignatures generates plots of the average sensor reading profiles
    % (fault signatures) for each subsystem.
    %
    % For each subsystem, the function finds all fault modes that belong to that
    % subsystem and plots the average (normalized) sensor readings of the sensors
    % associated with the subsystem when that fault mode occurs.
    %
    % Inputs:
    %   sensorData      - Normalized sensor data matrix.
    %   faultModes      - Vector of fault mode numbers for each sample.
    %   sensorNames     - Cell array of sensor names (for columns 1-36).
    %   fmMapping       - Map from fault mode numbers to subsystem names.
    %   fmDescriptions  - Map from fault mode numbers to descriptive fault names.
    %   subsystemMapping - Structure mapping subsystem names to sensor indices.
    
    subsystemsList = fieldnames(subsystemMapping);
    
    % Loop through each subsystem (ECS, Engine, Elec, Fuel)
    for s = 1:length(subsystemsList)
        subsystem = subsystemsList{s};
        sensorIndices = subsystemMapping.(subsystem);
        
        % Identify fault modes corresponding to the current subsystem
        faultModesInSubsystem = [];
        for fm = 1:25
            if isKey(fmMapping, fm) && strcmp(fmMapping(fm), subsystem)
                faultModesInSubsystem = [faultModesInSubsystem, fm]; %#ok<AGROW>
            end
        end
        
        if isempty(faultModesInSubsystem)
            continue;
        end
        
        figure;
        hold on;
        legendEntries = {};
        % For each fault mode in this subsystem, calculate the average sensor profile
        for fm = faultModesInSubsystem
            indices = find(faultModes == fm);
            if isempty(indices)
                continue;
            end
            avgProfile = mean(sensorData(indices, sensorIndices), 1);
            plot(avgProfile, 'LineWidth', 2);
            % Prepare legend entry with fault mode number and its description
            if isKey(fmDescriptions, fm)
                legendEntries{end+1} = ['FM' num2str(fm) ': ' fmDescriptions(fm)];
            else
                legendEntries{end+1} = ['FM' num2str(fm)];
            end
        end
        hold off;
        % Set x-axis ticks and labels to sensor names for the current subsystem
        set(gca, 'XTick', 1:length(sensorIndices), 'XTickLabel', sensorNames(sensorIndices), 'XTickLabelRotation', 45);
        xlabel('Sensors');
        ylabel('Normalized Reading');
        title(['Fault Signatures for ' subsystem ' Subsystem']);
        legend(legendEntries, 'Location', 'best');
        grid on;
    end
end



%%%%


function plotNDClikeSeverityProfilesGrouped(sensorData, faultModes, severities, sensorNames, fmDescriptions, fmMapping, subsystemMapping)
    % plotNDClikeSeverityProfilesGrouped
    %   - X: all sensors
    %   - Y: normalized reading
    %   - One curve per severity level (1.0:–0.1:0.1)
    %   - Sensors grouped by subsystem with separators + labels

    severityLevels = 1:-0.05:0.05;
    colors        = lines(numel(severityLevels));

    % Prepare subsystem boundaries
    sysNames = fieldnames(subsystemMapping);
    % Make sure each group of indices is sorted
    bounds = zeros(numel(sysNames),1);
    for k=1:numel(sysNames)
        idxs = sort(subsystemMapping.(sysNames{k}));
        bounds(k) = max(idxs);
    end

    uniqueFMs = unique(faultModes);
    for i = 1:numel(uniqueFMs)
        fm = uniqueFMs(i);
        idxFM = (faultModes == fm);
        if ~any(idxFM), continue; end

        fmData = sensorData(idxFM, :);
        fmSev  = severities(idxFM);

        figure('Name', sprintf('FM%d',fm),'NumberTitle','off');
        hold on;

        % plot each severity level (with some tolerance)
        tol = 0.1;
        for s = 1:numel(severityLevels)
            sevVal = severityLevels(s);
            sel = abs(fmSev - sevVal) < tol;
            if any(sel)
                avgProf = mean(fmData(sel,:), 1);
                plot(1:numel(sensorNames), avgProf, 'LineWidth', 2, 'Color', colors(s,:));
            end
        end

        % draw vertical separators between subsystems
        for b = 1:numel(bounds)-1
            xline(bounds(b)+0.5, '--k', 'LineWidth', 1);
        end

        hold off;

        % tidy up axes
        ax = gca;
        ax.XTick       = 1:numel(sensorNames);
        ax.XTickLabel  = sensorNames;
        ax.XTickLabelRotation = 45;
        xlabel('Sensor');
        ylabel('Normalized Reading');
        grid on;

        % add subsystem labels above the plot
        yl = ax.YLim;
        yText = yl(2) + 0.05*diff(yl);
        prev = 1;
        for k = 1:numel(sysNames)
            mid = (prev + bounds(k)) / 2;
            text(mid, yText, sysNames{k}, ...
                 'HorizontalAlignment','center', ...
                 'FontWeight','bold');
            prev = bounds(k) + 1;
        end

        % title
        if isKey(fmDescriptions, fm)
            title(sprintf('%s FAULTY RESULTS', upper(fmDescriptions(fm))));
        elseif isKey(fmMapping, fm)
            title(sprintf('%s FAULTY RESULTS', upper(fmMapping(fm))));
        else
            title(sprintf('FM%d FAULTY RESULTS', fm));
        end

        % legend
        legendLabels = arrayfun(@(x) sprintf('%.1f',x), severityLevels, 'UniformOutput',false);
        legend(legendLabels, 'Location','bestoutside');

        % expand ylim slightly to make room for subsystem labels
        ax.YLim = [yl(1), yText + 0.05*diff(yl)];
    end
end
