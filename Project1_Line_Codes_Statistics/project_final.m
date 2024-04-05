%% initialize workspace
clear all;
close all;
clc ;

%% Required Variables
A = 4; % Amplitude of the signal
num_bits = 100; % Number of bits per waveform
num_waveforms = 500; % Number of waveforms in the ensemble
num_of_plots = 1; % Number of waveforms to draw for each line code.
num_samples = 700; %Number of Samples in a waveform in one realization
Fs=100;  %Fs= 1/DAC Time, DAC Time = 0.01 seconds

%% Calling the created functions
% Polar Non Return to Zero
% Ensemble Generation
PNRZ_ensemble = Polar_NRZ(num_waveforms, num_bits, A)';
% Stat. Mean
PNRZ_statMean = statMean(PNRZ_ensemble, num_waveforms, num_samples);
% Time Mean
PNRZ_timeMean = timeMean(PNRZ_ensemble, num_waveforms, num_samples);
% Stat. Autocorrelation
PNRZ_statACF = statAutoCorrelation(PNRZ_ensemble);
% Time Autocorrelation
PNRZ_timeACF = timeAutoCorrelation(PNRZ_ensemble);
% PSD
plotPSD(PNRZ_statACF, Fs, 1);

% Unipolar Signaling
UP_ensemble = Uni_Polar(num_waveforms, num_bits, A)';
% Stat. Mean
UP_statMean = statMean(UP_ensemble, num_waveforms, num_samples);
% Time Mean
UP_timeMean = timeMean(UP_ensemble, num_waveforms, num_samples);
% Stat. Autocorrelation
UP_statACF = statAutoCorrelation(UP_ensemble);
% Time Autocorrelation
UP_timeACF = timeAutoCorrelation(UP_ensemble);
% PSD
plotPSD(UP_statACF, Fs, 2);

% Polar Return to Zero
PRZ_ensemble = Polar_RZ(num_waveforms, num_bits, A)';
% Stat. Mean
PRZ_statMean = statMean(PRZ_ensemble, num_waveforms, num_samples);
% Time Mean
PRZ_timeMean = timeMean(PRZ_ensemble, num_waveforms, num_samples);
% Stat. Autocorrelation
PRZ_statACF = statAutoCorrelation(PRZ_ensemble);
% Time Autocorrelation
PRZ_timeACF = timeAutoCorrelation(PRZ_ensemble);
% PSD
plotPSD(PRZ_statACF, Fs, 3);

% Plotting Line codes Waveforms
figure;
subplot(3, 1, 1);
plot(PNRZ_ensemble(1, :));
xlabel('time');
ylabel('Amplitude');
title("PNRZ Waveform");
ylim([-5, 5]); % Limit y-axis
subplot(3, 1, 2);
plot(UP_ensemble(1, :));
xlabel('time');
ylabel('Amplitude');
title("Unipolar Waveform");
ylim([-5, 5]); % Limit y-axis
subplot(3, 1, 3);
plot(PRZ_ensemble(1, :));
xlabel('time');
ylabel('Amplitude');
title("PRZ Waveform");
ylim([-5, 5]); % Limit y-axis

% Plotting STAT MEAN OF THE 3 line codes
Samples = 1:700;
figure;
subplot(3, 1, 1);
plot(Samples, PNRZ_statMean);
xlabel('Samples');
ylabel('Magnitude');
ylim([-4, 4]); % Limit y-axis
title("PNRZ Statistical Mean");

subplot(3, 1, 2);
plot(Samples, UP_statMean);
xlabel('Samples');
ylabel('Magnitude');
ylim([-4, 4]); % Limit y-axis
title("Uni-polar Statistical Mean");

subplot(3, 1, 3);
plot(Samples, PRZ_statMean);
xlabel('Samples');
ylabel('Magnitude');
ylim([-4, 4]); % Limit y-axis
title("PRZ Statistical Mean");

% Plotting TIME MEAN OF THE 3 line codes
waveforms = 1:500;
figure;
subplot(3, 1, 1);
plot(waveforms, PNRZ_timeMean);
ylabel('Magnitude');
ylim([-4, 4]); % Limit y-axis
title("PNRZ Time Mean");

subplot(3, 1, 2);
plot(waveforms, UP_timeMean);
ylabel('Magnitude');
ylim([-4, 4]); % Limit y-axis
title("Uni-polar Time Mean");

subplot(3, 1, 3);
plot(waveforms, PRZ_timeMean);
ylabel('Magnitude');
ylim([-4, 4]); % Limit y-axis
title("PRZ Time Mean");

% Stat. AutoCorrelation Plotting
figure;
subplot(3, 1, 1);
plotACF(PNRZ_statACF,'PNRZ Statistical AutoCorrelation');
subplot(3, 1, 2);
plotACF(UP_statACF,'Uni-Polar Statistical AutoCorrelation');
subplot(3, 1, 3);
plotACF(PRZ_statACF,'PRZ Statistical AutoCorrelation');

% Time Autocorrelation Plotting
figure;
subplot(3, 1, 1);
plotACF(PNRZ_timeACF,'PNRZ Time AutoCorrelation');
subplot(3, 1, 2);
plotACF(UP_timeACF,'Uni-Polar Time AutoCorrelation');
subplot(3, 1, 3);
plotACF(PRZ_timeACF,'PRZ Time AutoCorrelation');



%% Line Codes Functions Implementation
% Polar NRZ Function
function ensemble_shifted = Polar_NRZ(num_waveforms, num_bits , A)
% Initialize ensemble matrix
ensemble = zeros((num_bits+1) * 7, num_waveforms);
ensemble_shifted = zeros(num_bits*7, num_waveforms);

% Generate each waveform in the ensemble
for i = 1:num_waveforms
    % Generate random binary data for each waveform
    Data = randi([0, 1], 1, num_bits+1);
    % Modulate the binary data for polar NRZ signaling
    Tx = ((2 * Data) - 1) * A;
    % Repeat the waveform to match the duration of 70ms per bit
    Tx_2 = repmat(Tx, 7, 1);
    % Reshape the repeated waveform into a column vector
    Tx_out = reshape(Tx_2, size(Tx_2, 1) * size(Tx_2, 2), 1);
    ensemble(:, i) = Tx_out;

    %*** RANDOM DELAY  ***%
    % Generate Random initial shift
    initial_shift = randi([0, 6]);
    % Apply the time shifts
    Tx_out_shifted = Tx_out(initial_shift+1:(initial_shift+700));
    % Store the shifted waveform in the shifted ensemble matrix
    ensemble_shifted(:, i) = Tx_out_shifted;
end
end

% Uni-Polar Function
function ensemble_shifted = Uni_Polar(num_waveforms, num_bits , A)
% Initialize ensemble matrix
ensemble = zeros((num_bits+1) * 7, num_waveforms);
ensemble_shifted = zeros(num_bits*7, num_waveforms);

% Generate each waveform in the ensemble
for i = 1:num_waveforms
    % Generate random binary data for each waveform
    Data = randi([0, 1], 1, num_bits+1);
    % Modulate the binary data for Uni-Polar signaling
    Tx = Data*A;
    % Repeat the waveform to match the duration of 70ms per bit
    Tx_2 = repmat(Tx, 7, 1);
    % Reshape the repeated waveform into a column vector
    Tx_out = reshape(Tx_2, size(Tx_2, 1) * size(Tx_2, 2), 1);
    ensemble(:, i) = Tx_out;

    %*** RANDOM DELAY  ***%
    % Generate Random initial shift
    initial_shift = randi([0, 6]);
    % Apply the time shifts
    Tx_out_shifted = Tx_out(initial_shift+1:(initial_shift+700));
    % Store the shifted waveform in the shifted ensemble matrix
    ensemble_shifted(:, i) = Tx_out_shifted;
end   
end

% Polar-RZ Function
function ensemble_shifted = Polar_RZ(num_waveforms, num_bits , A)
% Initialize ensemble matrix
ensemble = zeros((num_bits+1) * 7, num_waveforms);
ensemble_shifted = zeros(num_bits*7, num_waveforms);

% Generate each waveform in the ensemble
for i = 1:num_waveforms
    % Generate random binary data for each waveform
    Data = randi([0, 1], 1, num_bits+1);
    % Modulate the binary data for Polar-RZ signaling
    Tx = ((2 * Data) - 1) * A;
    % Repeat the waveform to match the duration of 70ms per bit
    % 40ms bit value and 30ms for return to zero
    Tx_2 = repmat(Tx, 4, 1);
    % Reshape the repeated waveform into a column vector
    Tx_out = reshape(Tx_2, size(Tx_2, 1) * size(Tx_2, 2), 1);
    % Create zeros array to stuff the ensemble with it
    zeros_array = zeros((num_bits+1)*3, 1);

    % Stuffing the zeros into the array to have the final ensemble
    Tx_RZ = zeros((num_bits+1)*7,1);
    for j = 1:(num_bits+1)
        start_idx = (j-1)*7+1;
        end_idx = j*7;
        Tx_RZ(start_idx:end_idx) = [Tx_out((j - 1) * 4 + 1 : j * 4); zeros_array((j - 1) * 3 + 1 : j * 3)];
    end
    ensemble(:,i) = Tx_RZ; % ensemble with extra bit
 
    %*** RANDOM DELAY  ***%
    % Generate Random initial shift
    initial_shift = randi([0, 6]);
    % Apply the time shifts
    Tx_RZ_shifted = Tx_RZ(initial_shift+1:(initial_shift+700));
    % Store the shifted waveform in the shifted ensemble matrix
    ensemble_shifted(:, i) = Tx_RZ_shifted;
end
end

% Function to plot waveforms of different linecodes
function plotWaveforms(num_of_plots, ensemble, titleStr)
 % Plot the waveforms in a subplot
   figure;
   for i = 1:num_of_plots
    subplot(num_of_plots, 1, i);
    plot(ensemble(i, :));
    xlabel('time');
    ylabel('Amplitude');
    ylim([-5, 5]); % Limit y-axis plot to zero
    title(titleStr+ " waveform " + num2str(i));
   end
end

% Function to Calculate time mean
function time_mean = timeMean(data,num_waveforms,num_bits)
% MEANACROSSTIME calculates the mean across time for each statistic
%   Inputs:
%       data: Ensemble (each row is a statistic, each column is a time point)
%       num_waveforms
%       num_bits
%   Output:
%       time_mean: row Vector containing the mean across time for each waveform

% --------preallocation of mean for speed -----%
time_mean_i= zeros(num_waveforms,1);
    for i=1:num_waveforms
            time_mean_i(i)=sum(data(i,:));
    end  
        %---Saving the divsion over total num_bits at end is more efficient%
    time_mean = ( time_mean_i * (1/(num_bits)) )'; %transpose to get row vector instead of column;
end

% Function to calculate statistical mean
function statstical_mean = statMean(data, num_waveforms, num_bits)
% --------preallocation of mean for speed -----%
statstical_mean_i= zeros(num_bits,1);
    for i=1:num_bits
            statstical_mean_i(i)=sum(data(:,i));
    end
    %---Saving the divsion over total num_waveforms at end is more efficient%
   
  statstical_mean = statstical_mean_i *(1/num_waveforms);
end

% Function to Calculate statistical Autocorrelation
function R_x = statAutoCorrelation(data)

  [num_waveforms, num_samples] = size(data);
%------------ preallocation for speed -----------%
  ensemble_autocorr  = zeros(1,num_samples);  
  for waveform= 1:num_waveforms
    for sample=1:num_samples 
     ensemble_autocorr(1,sample) = ensemble_autocorr(1,sample) + data(waveform, 1) * data(waveform, sample);
    end
   end
  ensemble_autocorr = ensemble_autocorr/num_waveforms;
  R_x =[fliplr(ensemble_autocorr(1,2:end)), ensemble_autocorr];
 end

% Function to calculate Time autocorrelation
function Rx_time = timeAutoCorrelation(data)
    [num_waveforms, num_bits] = size(data);
    %------------ preallocation for speed -----------%
    ensemble_time_autocorr = zeros(1, num_bits);
    first_realization = data(1, :);
    for bit = 1 : num_bits
        if bit == 1
            shifted_bits = first_realization(1:end);
        else
            shifted_bits = [first_realization(bit:end), data( 1:(bit-1) )];
        end
        ensemble_time_autocorr(bit) = sum(first_realization .* shifted_bits);
    end
    ensemble_time_autocorr = ensemble_time_autocorr / num_bits;
    Rx_time = [fliplr(ensemble_time_autocorr(2:end)), ensemble_time_autocorr];
end

% Function to Plot Autocorrelation
function plotACF(ACF,titleStr)
% Plot the autocorrelation function
lag_indices = -699:699; % Generate lag indices
plot(lag_indices, ACF);
xlabel('Lag');
ylabel('Autocorrelation');
title(titleStr);
ylim([0, max(ACF)]); % Limit y-axis plot to zero
xlim([lag_indices(670), lag_indices(730)]); % Limit x-axis plot to +-30
end


% Function to Plot the PSD of the ensemble (FFT of the Autocorrelation?)
function plotPSD(R_x, fs,n)
    % Function to plot the Power Spectral Density (PSD) from the statistical autocorrelation function
    % Inputs:
    %   R_x - Autocorrelation function
    %   fs - Sampling frequency
    %   n index for which plot is (Polar NRZ, UniPolar, Polar RZ)
    
    %Get the Length of the Autocorrelation Function
    N = length(R_x);
    % Compute the Fourier Transform of the autocorrelation function and Normalize it
    PSD = fftshift(fft(R_x))/100;
    % Take the absolute value to ensure no complex numbers are present
    PSD = abs(PSD);
    % Create a frequency vector centered around zero
    f = (-N/2:N/2-1)*(fs/N);
    % Plot the PSD
    %Using a switch statment, if n=1 then this is the PSD of The Polar NRZ
    %if n=2, PSD of UniPolar and if n=3 , PSD of Polar RZ
switch n
    case 1
    figure;
    plot(f, PSD);
    title('Polar NRZ PSD');
    xlabel('Frequency (Hz)');
    ylabel('Power');
    grid on;
    % Determine the bandwidth, freq till 1st null
    postivie_indicies= (f>0);
    f_positive =f(postivie_indicies);
    postivie_PSD=PSD(postivie_indicies);
    [min_PSD, min_index] = min(postivie_PSD);
    BW=f_positive(min_index);
    disp(['Bandwidth of the The Polar NRZ : ', num2str(BW), ' Hz']);
    case 2
    figure;
    plot(f, PSD);
    title('Uni Polar PSD');
    xlabel('Frequency (Hz)');
    ylabel('Power');
    grid on;
    % Determine the bandwidth, freq till 1st null
    postivie_indicies= (f>0);
    f_positive =f(postivie_indicies);
    postivie_PSD=PSD(postivie_indicies);
    [min_PSD, min_index] = min(postivie_PSD);
    BW=f_positive(min_index);
    disp(['Bandwidth of the The UniPolar : ', num2str(BW), ' Hz']);
    case 3
    figure;
    plot(f, PSD);
    title('Polar RZ PSD');
    xlabel('Frequency (Hz)');
    ylabel('Power');
    grid on;
    % Determine the bandwidth, freq till 1st null
    postivie_indicies= (f>0);
    f_positive =f(postivie_indicies);
    postivie_PSD=PSD(postivie_indicies);
    [min_PSD, min_index] = min(postivie_PSD);
    BW=f_positive(min_index);
    disp(['Bandwidth of the The Polar RZ : ', num2str(BW), ' Hz']);
    otherwise  %just default case not supposed to happen
     figure;
    plot(f, PSD);
    title('Power Spectral Density (PSD) from Autocorrelation');
    xlabel('Frequency (Hz)');
    ylabel('Power');
    grid on;
     % Determine the bandwidth, freq till 1st null
    postivie_indicies= (f>0);
    f_positive =f(postivie_indicies);
    postivie_PSD=PSD(postivie_indicies);
    [min_PSD, min_index] = min(postivie_PSD);
    BW=f_positive(min_index);
    disp(['Bandwidth of the The Polar NRZ : ', num2str(BW), ' Hz']);
end
end

