%% Overview
%
% Free Energy of each HMM run is extracted.
% This may take some time.
% Free Energy values are used to identify HMM-inference with best fit.
% Main Analysis in Paper are run on the HMM with 8 States on lowest FE.

clearvars;
addpath(genpath('/home/okohl/Documents/rest_PD/My_Scripts/'));
addpath(genpath('/home/okohl/Documents/Toolboxes/osl_old/'));
osl_startup;

% ---- Set Up ----

% Parameters of HMMs
number_of_states = [8]; % Number of States
embedding_windows = [7]; % Length of Embedding
sampling_rates = 250; % sampling rate of Data
nrepeats = 1:10;

% Project Dir
Project_Dir_old = '/home/okohl/Documents/HMM_PD_V07/Data/'; % Comment
Project_Dir = '/home/okohl/Documents/HMM_PD_V07/Data/'; % Set to 3
dir_hmm_in = [Project_Dir_old,'hmm_in/']; % rm olde_
dir_hmm_out = [Project_Dir,'hmm_out/'];
dir_plots = '/home/okohl/Documents/HMM_PD_V07/Results/State_Descriptions/'; % rm4

% Load Data
outfile = fullfile(dir_hmm_in, 'ds250_embedded_hmm_data_flipped.mat');
load(outfile); % Load Data

fe_all = zeros(length(number_of_states),length(nrepeats));


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
                load( hmm_outfile ,'T','hmm','Gamma'); 
                
                fe = hmmfe(data,T,hmm,Gamma);
                fe_all(iState,irep) = fe(end);               
            end
            
            % ---- Save Free Engery Values ----
            save([dir_hmm_out,'fe_State',num2str(number_of_states(iState)),'.mat'],'fe_all')
            
        end
    end
end

