%% Overview
%
% Here we load in our source parcellated MEG data from the preprocessing stage,
% perform some normalisation and compute an Time-Delay-Embedded HMM.

clear all
restoredefaultpath

%% get started

addpath(genpath('path/to/osl/'));
osl_startup;

% load function allowing to get locations of participant files
addpath(genpath('path/to/My_Scripts/'));

% define output directory
outdir = '/path/to/outdir';

% define subjects and task to analyse
subjects = [2:19 51:54 56:59 61:71 101:107 109 110 116:118 151 153:157 159:170];%[1 2 5:9 12:18 51:55 57 59 61 63:65 68:71 101 103:107 109 110 112 116 118 151 153 154 156 159:161 164:169];
task = 'rest';
fsample = 250;

%% Data preparation/normalisation for HMM

% We'll need to collect the data, T and epoch information
data = [];                          % HMM ready dataset
T = [];                             % Length of continuous good segments
R = [];                             % Indices of single run within data
B = cell(length(subjects),1);       % Indices of bad samples per session
trl = cell(length(subjects),1);     % epoch info per segment
runlen = zeros(length(subjects),1); % Length of run per good segment

% main normalisation loop
ind = 1;
for iSub = subjects
    
    % get path to participant data
    param = getSubjParamPD_new(iSub);
    
    fprintf('Processing %s\n',param.subjName);

    %------------------------------
    % continuous file
    D = spm_eeg_load(fullfile([param.path.preproc,param.preproc.rest]));
    D = D.montage('switch',5);
    
    % Filter Data
    D = osl_filter(D,[1 45],'prefix','filt_');
    
    % Downsample Data
    D = D.montage('switch',0);
    D = spm_eeg_downsample(struct('D',D,'fsample_new',fsample));    
    parcelated_montage = D.montage('getnumber');
    D = D.montage('switch',parcelated_montage);
   
    runlen(ind) = size(D,2);

    %-------------------------------
	% get data and orthogonalise
	dat = D(:,:,1);
	dat = ROInets.remove_source_leakage(dat, 'symmetric');

	%-------------------------------
    % Get badsamples
    runlen(ind) = size(dat,2);
    bs = ~good_samples( D );
    
    clear D;

    % find single good samples - bug when we have consecutive bad segments
    xx = find(diff(diff(bs)) == 2)+1;
    if ~isempty(xx)
        bs(xx) = 1;
    end

    % store bad samples
    B{ind}=find(bs);

    % indices of good samples
    good_inds=setdiff(1:runlen(ind),B{ind});

    % remove bad samples,
    dat = dat(:,good_inds);

    if any(bs)

        t_good = ~bs;
        db = find(diff([0; t_good(:); 0]));
        onset = db(1:2:end);
        offset = db(2:2:end);
        t = offset-onset;

        % sanity check
        if size(dat,2) ~= sum(t)
            disp('Mismatch between Data and T!!');
        end
    else
        t = size(dat,2);
    end
    
    
    % ----- Remove Outliers  -----
    
    % Devide data in 10sec segments and calculate Variance and Curtosis for
    % each segment. Run GESD outlier detection of Variance and Curtosis.
    
    dat_in = dat;
    leakage_corr = false;
    seg_length = 10;
    
    [corrData, t, dropInfo] = outlierRM(dat_in, leakage_corr, seg_length, fsample);  
 

    %--------------------------------
    % Concatenate Data acorss Participants

    offset = sum(T);
    R = cat(1,R,[offset+1 offset+size(corrData,2)]);
    T = cat(1,T,t');
    data = cat(1,data,corrData');
    
    ind = ind + 1;
end

% ------ Save HMM-ready data -----

% Define HMM folder and save HMM-ready data
hmm_folder = fullfile(outdir,'hmm_in/');
outfile = fullfile( hmm_folder, 'ds250_embedded_hmm_data' );

save( outfile, 'data', 'R', 'T', 'B', 'runlen', '-v7.3' );



%% resolve dipol sign ambiguity
% The sign ambiguity in the beamforming process means that data from the same 
% parcel from different sessions may have arbitrarily opposite signs. Across 
% a group-level dataset this can lead to suppression between group-level phase 
% relations between nodes. To reduce this effect we applied the sign-flipping 
% algorithm described in Vidaurre et al. (2017). (From Quinn et al., 2018)

run_flip = true;
if run_flip
    options_signflip = [];
    options_signflip.maxlag = 4; 
    options_signflip.verbose = 0;
    options_sf.noruns = 20;
    options_signflip.nbatch = 3;
    T2 = R(:,2)-(R(:,1)-1);
    
    % run sign-flipping
    flips = findflip(data,T2',options_signflip);
    data = flipdata(data,T2',flips);
  
    % Define HMM folder and save HMM-ready data
    outfile = fullfile(hmm_folder, 'ds250_embedded_hmm_data_flipped.mat' );

    save( outfile, 'data', 'R', 'T', 'B', 'runlen', '-v7.3' );
end


%% Additional cleaning and leakage correction

% ------ Script removing outliers from HMM ready data ------

% Steps (for each subject individually):
% 1) Devide Data into 10 Sec time windows
% 2) Calculate Kurtosis and STD for each window
% 3) Generalized extreme Studentized deviate test to identfy outlier
%    windows for respective subject
% 4) remove outlier windows and put merge remaining windows again.
% 
% If a large percentag of a subject's data is discarded considering
% revisting the preprocessing for this subject.

function [corrData, T, drop_info] = outlierRM(dat_in, leakage_corr, seg_length, fsample)

    % Function igores Segments within Subjects in T and cerates new T based
    % on segments that are created in this function.

    % Future: Look for way to account for T from bad sample removing. In
    % case this will not remove to many samples.

    % Number of Parcels
    nParc = size(dat_in,1);
    
    % ---- Devide signal into 10 second periods ----
    x = permute(dat_in,[1,2]); % Sort Dimensions
    lengSegs = seg_length*fsample; % Set WindowLength
    nsegs = floor(size(x,2)/lengSegs); % Caluclate number of 10sec segments for Sub
    x = x(:,1:nsegs*lengSegs); % Get rid of samples in last interval that is shorter than 10 sec. HERE WE REMOVE DATA!
    x = reshape(x,nParc,lengSegs,nsegs); % bring data into (Roi x Segment Length x Segment number) shape
    
    % ----- Check Identify outlierw ----
    drops = gesd(std(reshape(x,nParc*lengSegs,size(x,3)))); % check variance of segments
    drops = drops + gesd(kurtosis(reshape(x,nParc*lengSegs,size(x,3))));  % check kurtosis of segments
    drops =  drops > 0; % Bolean indicating segments to drop
    
    fprintf('\tdropping %d/%d segments\n', sum(drops), length(drops));

    % --- the actual dropping + leakage correction take place ---
    data = [];
    for jj = 1:nsegs
        if drops(jj) ~=  1
            
            % Leakage correction
            if leakage_corr
                tmp = ROInets.remove_source_leakage(x(:,:,jj),'symmetric');
            else
                tmp = x(:,:,jj);
            end
            
            data = [data tmp]; % concatenate data
        end
    end
   
    % --- Collect data for output ---
     T = repmat(lengSegs,1,nsegs-sum(drops));
    drop_info.dropped_seg =  sum(drops);
    drop_info.all_seg = length(drops);
    corrData = data;
    
end

