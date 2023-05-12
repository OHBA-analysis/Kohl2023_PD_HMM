%% Overview
% Script does the following steps:
%   1) runs HMM inferences (loops accross No of States, and repetitions).
%   2) calculates state metrics
%   3) saves Gammas and input data for further analysis in Python.

clearvars;
addpath(genpath('/path/to/My_Scripts/'));
addpath(genpath('/path/to/My_Scripts/osl_old/'));
osl_startup;

%% Set Dirs and Parameters

% Specify across which HMM settings you want to loop
number_of_states = [8,10,12]; % Number of States
embedding_windows = [7]; % Length of Embedding
nrepeats = [1:10]; % Number of repetitions of HMM-inference for same parameters
sampling_rates = 250; % sampling rate of Data
sub_num = 67;

% Project Dir
Project_Dir = '/home/okohl/Documents/HMM_PD_V07/Data/';
dir_preproc = [Project_Dir,'/preproc/'];
dir_hmm_in = [Project_Dir,'/hmm_in/'];
dir_hmm_out = [Project_Dir,'hmm_out/'];
dir_spectra = [Project_Dir,'spectra/'];
dir_toPython = [Project_Dir,'spectra_toPython/'];
dir_plots = '/home/okohl/Documents/HMM_PD_V07/Results/State_Descriptions/';

% Locate File with HMM input data
outfile = fullfile(dir_hmm_in, 'ds250_embedded_hmm_data_flipped.mat');

% ---------------------------------------
%% HMM inference
% ---------------------------------------

load(outfile); % Load Data

% Prepare options structure
options = struct();
options.verbose = 1;

% These options specify the data and preprocessing that hmmmar might perform. Further options are discussed here
options.onpower = 0;
options.standardise = 1;
options.standardise_pc = options.standardise;
options.Fs = sampling_rates;

% Here we specify the HMM parameters
options.order = 0;              % The lag used, this is only relevant when using MAR observations
options.zeromean = 1;           % We do not want to model the mean, so zeromean is set on
options.covtype = 'full';       %  We want to model the full covariance matrix
options.pca = size(data,2)*2;   % The PCA dimensionality reduction is 42times the number of ROIs
options.useParallel = 1;

% These options specify parameters relevant for the Stochastic inference. 
% A detailed description of the Stochastic options and their usage can 
% be found here:
% https://github.com/OHBA-analysis/HMM-MAR/wiki/User-Guide#stochastic
options.BIGNinitbatch = 5;
options.BIGNbatch = 5;
options.BIGtol = 1e-7;
options.BIGcyc = 500;
options.BIGundertol_tostop = 5;
options.BIGdelay = 5; % 1 recommended in HMM mar wiki
options.BIGforgetrate = 0.7;
options.BIGbase_weights = 0.9;

% Run HMM
% The following loop performs the main HMM inference. 
% The HMM inference is repeated 10 times for each Number of States to later
% assess the robust ness across different runs and number of states.
% Note that this can be extremely time-consuming for large datasets. 
% For a quick exploration of results, nrepeats can be set to a smaller value or even 1
% and the number_of_States can be set to 8 (or 10 or 12)
% The full inference is run over 10 repeats fur HMMs extracting 8, 10, and 12 states.

for kk = number_of_states
    best_freeenergy = nan;
    options.K = kk;
    
    for iemb = embedding_windows
        options.embeddedlags = -iemb:iemb;
        for irep = nrepeats
            % Run the HMM, note we only store a subset of the outputs
            % more details can be found here: https://github.com/OHBA-analysis/HMM-MAR/wiki/User-Guide#estimation
            [hmm, Gamma, ~, vpath, ~, ~, ~, ~, fehist] = hmmmar(data,T,options);
            
            % Save the HMM outputs
            hmm_outfile =  [dir_hmm_out,'/ds',num2str(sampling_rates),'/K',num2str(kk),'/cat_ds',num2str(sampling_rates),'_TDE_HMM_K',num2str(options.K),'_emb',num2str(iemb),'_run',num2str(irep)];
            save( hmm_outfile ,'hmm','Gamma','vpath','T','options','fehist')

            clear hmm Gamma vpath fehist
        end
    end
end



% -----------------------------------------------------------------------
%% Get State Metrics

% Get Mean Lifetimes, IntervalTimes, Fracctional Occupancy and
% State Rate values.
% Make sure to check Fractional Occupancies to see whether HMM-States mix
% well, i.e., States do not model interindividual differences.
% ----------------------------------------------------------------------

% labels of the state metrics that will be calculated
labels = {'mean_lifetimes','mean_intervaltimes','std_lifetimes',...
    'std_intervaltimes','fractional_occupancy','state_rates','switching_rate'};

% load HMM input vars
outfile = fullfile(dir_hmm_in, 'ds250_embedded_hmm_data_flipped.mat');
load(outfile,'R'); % Load Data

% --- loop over different repetitions ---
for irep = nrepeats
    
    % --- loop over different sampling rates ---
    for iSamp = 1:length(sampling_rates)
        ds = sampling_rates(iSamp);

        % ----  Loop over different embeddings ----
        for iemb = 1:length(embedding_windows)
            emb = embedding_windows(iemb);

            % ---- Loop over different number of states ----
            for iState = 1:length(number_of_states)
                K = number_of_states(iState);

                % -------  Load output from TDE-HMM estimation ---------
                hmm_outfile = [dir_hmm_out,'/ds',num2str(sampling_rates),'/K',num2str(K),'/cat_ds',num2str(sampling_rates),'_TDE_HMM_K',num2str(K),'_emb',num2str(emb),'_run',num2str(irep)];   
                load( hmm_outfile ,'hmm','Gamma','T')

                % account for delay embedding in state gammas
                pad_options = struct;
                pad_options.embeddedlags = -emb:emb;
                Gamma = padGamma(Gamma, T, pad_options);

                if size(Gamma,1) ~= sum(T)
                    warning('The size of data and Gamma do not match');
                end

                % create fake Vpath based on Gamma
                [~,fake_vpath] = max(Gamma');
                
                
                % ----- Get State Metrics per Subject ---------

                % Preallocate Vars
                metric.mean_lifetimes = zeros(size(R,1),K);
                metric.mean_intervaltimes = zeros(size(R,1),K);
                metric.std_lifetimes = zeros(size(R,1),K);
                metric.std_intervaltimes = zeros(size(R,1),K);
                metric.state_rates = zeros(size(R,1),K);
                metric.fractional_occupancy = zeros(size(R,1),K);
                metric.switching_rate = zeros(size(R,1),1);

                % Loop accross Participants
                for iSub = 1:length(R)  
                    
                    % get Ts of Sub
                    T_sub = R(iSub,2)-(R(iSub,1)-1);
                    
                    
                    % ---- Lifetimes and interval times ----

                    % Get state lifetimes and interval times in samples
                    lifetimes = getStateLifeTimes(Gamma(R(iSub,1):R(iSub,2),:),T_sub,hmm.train);
                    intervaltimes = getStateIntervalTimes(Gamma(R(iSub,1):R(iSub,2),:),T_sub,hmm.train);

                    % Tranform state durations from samples in seconds
                    lifetimes_ms{iSub,:} = cellfun(@(x) x*(1/ds),lifetimes,'UniformOutput',false); % time in s spend per in state for each visit
                    intervaltimes_ms{iSub,:} = cellfun(@(x) x*(1/ds),intervaltimes,'UniformOutput',false); % time in s between visits to state

                    % calculate mean lifetimes
                    metric.mean_lifetimes(iSub,:) = cellfun(@(x) mean(x),lifetimes_ms{iSub,:});
                    metric.mean_intervaltimes(iSub,:) = cellfun(@(x) mean(x),intervaltimes_ms{iSub,:});

                    
                    % ---- Fractional occupancy -----
                    fractionalOccupancy_tmp = getFractionalOccupancy(Gamma(R(iSub,1):R(iSub,2),:),T_sub,hmm.train); % proportion of time spent in each state over whole trial
                    metric.fractional_occupancy(iSub,:) = mean(fractionalOccupancy_tmp,1);

                    
                    % ---- State Rates (because fractional occipancy takes ----
                    
                    % ----      length of events into account)      ----
                    stateOnsets = getStateOnsets(fake_vpath(R(iSub,1):R(iSub,2))',T_sub,ds,K);
                    stateOnsets_perSec = cellfun(@(x) length(x)/(T_sub(1)/ds),stateOnsets); % Hard Coded because all Ts have the same length of 10sec
                    metric.state_rates(iSub,:) = mean(stateOnsets_perSec,1); 

                    % ---- Switching Rate -----
                    metric.switching_rate(iSub,:) = mean(getSwitchingRate(Gamma(R(iSub,1):R(iSub,2),:),T_sub,hmm.train)); % Simple mean ok because all segmants same length           
                
                end

                
                % ---- Save Each State Metric in Separate File ----

                metric_out = fullfile(Project_Dir,['StateMetrics/ds',num2str(ds),'/K',num2str(K),'/run',num2str(irep),'/']);

                for i = 1:length(labels)
                    out = metric.(labels{i});

                    outfile = fullfile(metric_out, [labels{i},'.mat']);
                    save( outfile, 'out');
                end
                clear lifetimes_ms intervaltimes_ms
            end
        end
    end

clear metric StateOnsets lifetimes_ms lifetimes intervaltimes_ms intervaltimes vpath_padded Gamma

end
disp('State Metrics Calculated')


% --------------------------------------------------
%% Save Data for State Spectra Calculation in Python
%  This data will be fed into GLM-Spectrograms to
%  calculate state specific power and coherence.
% -------------------------------------------------

% load HMM input vars
outfile = fullfile(dir_hmm_in, 'ds250_embedded_hmm_data_flipped.mat');
load(outfile,'data','R'); % Load Data
%data = data';

% --- loop over different repetitions ---
for irep = nrepeats
    
    % --- loop over different sampling rates ---
    for iSamp = 1:length(sampling_rates)
        ds = sampling_rates(iSamp);

        % ----  Loop over different embeddings ----
        for iemb = 1:length(embedding_windows)
            emb = embedding_windows(iemb);

            % ---- Loop over different number of states ----
            for iState = 1:length(number_of_states)
                K = number_of_states(iState);

                % -------  Load output from TDE-HMM estimation ---------
                hmm_outfile = [dir_hmm_out,'/ds',num2str(sampling_rates),'/K',num2str(K),'/cat_ds',num2str(sampling_rates),'_TDE_HMM_K',num2str(K),'_emb',num2str(emb),'_run',num2str(irep)];   
                load( hmm_outfile ,'Gamma','T')

                % ------- Prepare Gamma for exporting --------
                % account for delay embedding in state gammas
                pad_options = struct;
                pad_options.embeddedlags = -emb:emb;
                Gamma = padGamma(Gamma, T, pad_options);
                
                if size(Gamma,1) ~= size(data,1)
                    warning('The size of data and Gamma do not match');
                end
                
                % ----- Loop picking subject's data and Gamma and stores them -----
                sub_num = size(R,1);
                for ind = 1:sub_num
                    disp(ind);

                    subj_data = data(R(ind,1):R(ind,2),:);
                    subj_Gamma = Gamma(R(ind,1):R(ind,2),:);
                    
                    mt_outfile = fullfile( [dir_toPython,'/ds',num2str(sampling_rates),'/K',num2str(K),'/run',num2str(irep),'/Subject',num2str(ind),'_HMMout.mat']);
                    save( mt_outfile ,'subj_Gamma','subj_data') 
                end
            end
        end
    end
end


