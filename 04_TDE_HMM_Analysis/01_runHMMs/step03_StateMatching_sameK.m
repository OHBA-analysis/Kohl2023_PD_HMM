% Script Matches states from different HMMs with same parameter selection
%
% 1) For each HMM, Autocovaraince Matrices for all HMM States are extracted
% 2) Distance Matrix for each HMM relative to HMM inference with lowest FE
%    are calculated. Distance calculated as correlation distance (1-corr)
% 3) Distance Matrix is fed into Hungarian algorythm that finds optimal
%    State to State allocations of the HMMs (lowest distance).
% 4) For each HMM vector indicating how HMM states need to be reshuffeled
%    to optimally match reference HMM state order, are stored in assig
%    matrix (Hmms x states).
%

clearvars;
addpath(genpath('/path/to/My_Scripts/'));
addpath(genpath('/path/to/osl/'));
osl_startup;

% Parameters of HMMs
number_of_states = [12]; % Number of States
embedding_windows = [7]; % Length of Embedding
sampling_rates = 250; % sampling rate of Data
nrepeats = 1:10; % Number of repetitions
revRun = [2]; % Number of HMM inference with lowest FE for respective number of States

% Project Dir
Project_Dir = '/path/to/Data/';
dir_hmm_in = [Project_Dir,'hmm_in/'];
dir_hmm_out = [Project_Dir,'hmm_out/'];
dir_AutoCov_out = [Project_Dir,'StateMatching/'];

%% --- Get autocovariance matrices of states for each HMM run ---

% --- loop over different sampling rates ---
for iSamp = 1:length(sampling_rates)
    ds = sampling_rates(iSamp);

    % ----  Loop over different embeddings ----
    for iemb = 1:length(embedding_windows)
        emb = embedding_windows(iemb);

        % ---- Loop over different number of states ----
        for iState = 1:length(number_of_states)
            K = number_of_states(iState);
            
            for irep = 1:length(nrepeats)
                
                % ----- Load HMM -----
                hmm_outfile = [dir_hmm_out,'/ds',num2str(sampling_rates),'/K',num2str(K),'/cat_ds',num2str(sampling_rates),'_TDE_HMM_K',num2str(K),'_emb',num2str(emb),'_run',num2str(irep)];   
                load( hmm_outfile ,'hmm'); 
                
                % ----- Get Covariance Matrix per state
                Cs = [];
                for k = 1:K
                    C = getAutoCovMat(hmm,k);
                    Cs(:,:,k) = C;                   
                end
                
                % Save autocovariance Matrices
                outfile = [dir_AutoCov_out,'/ds',num2str(sampling_rates),'/K',num2str(K),'/run',num2str(irep),'/AutoCov.mat'];
                save(outfile,'Cs')
            end
        end
    end
end


%% --- Calculate similarity matrices of all runs with run w lowest FE ---

% --- loop over different sampling rates ---
for iSamp = 1:length(sampling_rates)
    ds = sampling_rates(iSamp);

    % ----  Loop over different embeddings ----
    for iemb = 1:length(embedding_windows)
        emb = embedding_windows(iemb);

        % ---- Loop over different number of states ----
        for iState = 1:length(number_of_states)
            K = number_of_states(iState);
            
            RefCov_file = [dir_AutoCov_out,'/ds',num2str(sampling_rates),'/K',num2str(K),'/run',num2str(revRun(iState)),'/AutoCov.mat'];
            RefCov = load(RefCov_file);
            RefCov = RefCov.Cs;
            
            assig_all = [];
            cost_all = [];
            for irep = 1:length(nrepeats)
                
                Cov_file = [dir_AutoCov_out,'/ds',num2str(sampling_rates),'/K',num2str(K),'/run',num2str(irep),'/AutoCov.mat'];
                Cov = load(Cov_file);
                Cov = Cov.Cs;
                
                % --- calculate similarity matrix (Correlation distance) ---
                % Munkres gives optimal column assignmet for rows
                % thus have reference States on rows, so that Munkres finds
                % assignments for reference state for all matches
                
                distMat = zeros([K,K]);
                for k1 = 1:K
                    for k2 = 1:K
                        Cov1 = RefCov(:,:,k1);
                        Cov2 = Cov(:,:,k2);
                        distMat(k1,k2) = 1-corr(Cov1(:),Cov2(:));
                    end
                end
                
                % Save Similarity Mat
                outfile = [dir_AutoCov_out,'/ds',num2str(sampling_rates),'/K',num2str(K),'/run',num2str(irep),'/DistMat.mat'];
                save(outfile,'distMat')
                
                
                % --- Get Optimal state assignment ---
                
                % Apply Hungaryian algorithm to similarity matrix
                % to get optimal assignment of respective run to reference
                % run
                
                [assig,cost] = munkres(distMat);
                assig_all(irep,:) = assig;
                cost_all(irep,1) = cost;
                
            end
            
            outfile = [dir_AutoCov_out,'/ds',num2str(sampling_rates),'/K',num2str(K),'/StateMatching.mat'];
            save(outfile,'assig_all','cost_all');
        end
    end
end