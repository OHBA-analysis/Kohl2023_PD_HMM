%% Script Matches states between HMMs with lowest Free Energy per No of States
%
% 1) For each HMM w lowest FE, Autocovaraince Matrices for all HMM States are extracted
% 2) Distance Matrix for each 10 and 12 State HMM with lowest FE relative to HMM 
%    inferring States and lowest FE are calculated. Distance calculated as
%    correlation distance (1-corr).
%    Importantly, distance matrices are zero-padded so that they have
%    square shape. This important for Hungarian Alogrythm.
% 3) Distance Matrix is fed into Hungarian algorythm that finds optimal
%    State to State allocations of the HMMs (lowest distance). States of
%    the 10 and 12 State HMM that were matched to padded rows are set to
%    zero so that they can be ignored in later analyses where their states are
%    matched to 8 State HMM.
% 4) For each HMM vector indicating how HMM states need to be reshuffeled
%    to optimally match reference HMM state order, are stored in assig
%    matrix (Hmms x states).
%

clearvars;
addpath(genpath('/home/okohl/Documents/rest_PD/My_Scripts/'));
addpath(genpath('/home/okohl/Documents/Toolboxes/osl_old/'));
osl_startup;

% Parameters of HMMs
number_of_states = [8,10,12]; % Number of States
embedding_windows = [7]; % Length of Embedding
sampling_rates = 250; % sampling rate of Data
refRun = [7,2,2]; % HMM runs with lowest free energy for each K

% Project Dir
Project_Dir = '/home/okohl/Documents/HMM_PD_V07/Data/';
dir_hmm_in = [Project_Dir,'hmm_in/'];
dir_hmm_out = [Project_Dir,'hmm_out/'];
dir_AutoCov_out = [Project_Dir,'StateMatching/'];

%% --- Get autocovariance matrices of states for each HMM run ---

Cov_all = {};

% --- loop over different sampling rates ---
for iSamp = 1:length(sampling_rates)
    ds = sampling_rates(iSamp);

    % ----  Loop over different embeddings ----
    for iemb = 1:length(embedding_windows)
        emb = embedding_windows(iemb);

        % ---- Loop over different number of states ----
        for iState = 1:length(number_of_states)
            K = number_of_states(iState);
            iRef = refRun(iState);
                
            % ----- Load Autocovariance Matrix of reference runs -----
            CovMat_file = [dir_AutoCov_out,'/ds',num2str(sampling_rates),'/K',num2str(K),'/run',num2str(iRef),'/AutoCov.mat'];
            Cov = load( CovMat_file ,'Cs');
            Cov = Cov.Cs;
                
             % ----- Save AutoCov Matrices in one Matrix ---
             Cov_all{iState} = Cov; % (Roi x lag) x (Roi x lag) x States x RevRun 
        end
    end
end


%% --- For  Calculate similarity between States of Reference HMMs ---

% Set reference refHMM
refCov_all = Cov_all{1};

% ---- Loop over different number of states ----
for nStates = 2:length(number_of_states)
    Cov_all_in = Cov_all{nStates};
    
    distMat = zeros([number_of_states(nStates),number_of_states(nStates)]);
    for k1 = 1:number_of_states(1)
        for k2 = 1:number_of_states(nStates)
                Cov1 = refCov_all(:,:,k1);
                Cov2 = Cov_all_in(:,:,k2);
                
                distMat(k1,k2) = 1-corr(Cov1(:),Cov2(:));
        end
        
        % --- Get Optimal state assignment ---
        % Apply Hungaryian algorithm to similarity matrix
        % to get optimal assignment of respective run to reference run.
        % 10K and 12 K states matched to dummy rows will be set to 0
        % because they are not most similar to States of 8K HMM.
        
        [assig,cost] = munkres(distMat);
        assig(:,9:end) = 0; % Set where States were assigned to dummy var to zero
                
        % Save best State Matching
        outfile = [dir_AutoCov_out,'/ds',num2str(sampling_rates),'/K',num2str(number_of_states(nStates)),'_to_8K_RefHMM_matching.mat'];
        save(outfile,'distMat','assig','cost')
    end
end

