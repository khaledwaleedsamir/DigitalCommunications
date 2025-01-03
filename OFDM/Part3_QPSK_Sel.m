clear;
clc;
%..........................................................%
%                                                          %
%  No Coding QPSK over Frequency selective Fading channel  %   
%                                                          %
%..........................................................%

%% ************************* Parameters ************************* %%
numBits = 262144;               % Number of bits
Eb = 1;                          % Bit energy
bits_per_symbol = 2;             % bits per symbol for QPSK
interleaver_size = [16, 16];     % Interleaver size
IFFT_size = 128;                 % Number of points for FFT and IFFT
cp_length = IFFT_size/4;         % Cyclic prefix length 25% FFT length
EbN0_dB = -5:1:20;               % Eb/N0 range in dB (SNR)
BER = zeros(size(EbN0_dB,2),1);  % Array to store BER at different SNR values
% Generate random binary data
binary_data = randi([0 1], numBits, 1);

%% ************************* TRANSMITTER ************************* %%
% Initalize the ofdm_tx_signal
ofdm_tx_signal = [];
for j = 1 : 256 : numBits
    % Interleaver
    Interleaved_Data = Interleaver(binary_data(j:j+255),interleaver_size);
    % Mapper
    QPSK = QPSK_Gray_Mapper(Interleaved_Data)';
    % Perform the 128-point IFFT
    IFFT = ifft(QPSK, IFFT_size);
    % Add cyclic extension
    data_with_cp = [IFFT(end - cp_length + 1:end), IFFT];
    ofdm_tx_signal = [ofdm_tx_signal data_with_cp];
end
% Number of subchannels
num_subchannels = 128;
% Length of each subchannel
subchannel_length = length(ofdm_tx_signal) / num_subchannels;
% Initialize a cell array to store subchannels
ofdm_tx_subchannels = cell(1, num_subchannels);
ofdm_channel_subchannels = cell(1, num_subchannels);
% Split the OFDM transmitted signal into subchannels using a loop
for k = 1:num_subchannels
    start_idx = (k-1) * subchannel_length + 1;
    end_idx = k * subchannel_length;
    ofdm_tx_subchannels{k} = ofdm_tx_signal(start_idx:end_idx);
end
for i = 1 : size(EbN0_dB,2)
    %% ********* Frequency selective fading Channel ********* %%
    binary_data_received = [];
    % Generate the real and imaginary parts of the channel impulse response
    h_real = sqrt(1/2) * randn(num_subchannels, subchannel_length);
    h_imag = sqrt(1/2) * randn(num_subchannels, subchannel_length);
    SNR_linear = 10^(EbN0_dB(i)/10);
    N0 = Eb/SNR_linear;    % noise power
    variance = N0/2; % variance of the Gaussian noise
    % Generate the real part of the noise vector
    n_c = 0 + (1/sqrt(128))*sqrt(variance) * randn(1,length(ofdm_tx_signal));
    % Generate the imaginary part of the noise vector
    n_s = 0 + (1/sqrt(128))*sqrt(variance) * randn(1,length(ofdm_tx_signal));
    % Combine the real and imaginary parts to form the complex noise vector
    n = n_c + 1j * n_s;
    % Split the OFDM transmitted signal into subchannels and apply channel effect with noise
    for k = 1:num_subchannels
    start_idx = (k-1) * subchannel_length + 1;
    end_idx = k * subchannel_length;
    % Split the transmitted signal into subchannels
    ofdm_tx_subchannels{k} = ofdm_tx_signal(start_idx:end_idx);
    % Form the complex channel impulse response for each subchannel
    h = h_real(k, :) + 1j * h_imag(k, :);
    % Apply the channel effect and add noise to each subchannel
    ofdm_channel_subchannels{k} = ifft(h .* fft(ofdm_tx_subchannels{k})) + n(start_idx:end_idx);
    end
    % Combine the received signals from all subchannels
    ofdm_channel_signal_combined = [ofdm_channel_subchannels{:}];

    %% *************************  RECEIVER ************************* %% 
    % Initialize cell arrays to store the received subchannels
    ofdm_rx_subchannels = cell(1, num_subchannels);
    %Loop through each subchannel to compensate for the channel gain
    for k = 1:num_subchannels
        start_idx = (k-1) * subchannel_length + 1;
        end_idx = k * subchannel_length;
        % Split the combined received signal into subchannels
        ofdm_channel_subchannel = ofdm_channel_signal_combined(start_idx:end_idx);
        % Form the complex channel impulse response for each subchannel
        h = h_real(k, :) + 1j * h_imag(k, :);
        % Compensate for the channel gain at the receiver
        ofdm_rx_subchannels{k} = ifft(fft(ofdm_channel_subchannel) ./ h);
    end
    % Combine the compensated signals from all subchannels
    ofdm_rx_signal = [ofdm_rx_subchannels{:}];
    % Remove Cyclic extension
    for j = 1:numel(data_with_cp):numel(ofdm_channel_signal_combined)
        data_without_cp = ofdm_rx_signal(cp_length +j : j+cp_length +IFFT_size-1);
        % Perform the 128-point FFT
        FFT = fft(data_without_cp , IFFT_size);
        % De-mapper
        QPSK_Recievedsignal = QPSK_Gray_Demapper(FFT');
        % De-interleaver
        QPSK_Recievedsignal = QPSK_Recievedsignal';
        DeInterleaved_Data1 = reshape(QPSK_Recievedsignal,interleaver_size);
        DeInterleaved_Data = reshape (DeInterleaved_Data1.',1,[]);
        binary_data_received = [binary_data_received DeInterleaved_Data];
    end
    % Calculating BER
    num_error_bits = sum(binary_data ~= binary_data_received');
    BER(i) = num_error_bits/numBits;
end
% Plotting BER
figure('Name' , 'BER of QPSK over frequency selective fading channel');
semilogy(EbN0_dB , BER, 'r', 'linewidth', 1);
hold on;
title('OFDM System: BER of QPSK over frequency selective fading channel');
xlabel('EB/No(dB)');  ylabel('BER');  grid on;
% Limit y-axis to 10^-3
ylim([1e-3, 1]);

%.......................................................................%
%                                                                       %
% Rate 1/3 repetition code QPSK over frequency selective fading channel %
%                                                                       %
%.......................................................................%

%% ************************* Parameters ************************* %%
numBits_coded = 1000*252;              % Number of bits
BER_coded = zeros(size(EbN0_dB,2),1);  % Array to store BER at different SNR values
% Generate random binary data
binary_data_2 = randi([0 1], numBits_coded, 1);
% Repeatition Coding
repetition_factor = 3;  
binary_data_coded = repelem (binary_data_2, repetition_factor);

%% ************************* TRANSMITTER ************************* %%
% Initalize the ofdm_tx_signal
ofdm_tx_signal_coded = []
for j = 1 : 252 : numBits_coded*repetition_factor
    % padding the coded data with zeros
    pad = zeros(4,1);
    coded_data = [binary_data_coded(j:j+251); pad];
    % Interleaver
    Interleaved_Data_coded = Interleaver(coded_data, interleaver_size);
    % Mapper (QPSK)
    QPSK_coded = QPSK_Gray_Mapper(Interleaved_Data_coded)';
    % Perform the 128-point IFFT
    IFFT_coded = ifft(QPSK_coded, IFFT_size);
    % Add cyclic prefix
    coded_data_with_cp = [IFFT_coded(end - cp_length + 1:end), IFFT_coded];
    ofdm_tx_signal_coded = [ofdm_tx_signal_coded coded_data_with_cp];
end
% Number of subchannels
num_subchannels = 128;
% Length of each subchannel
subchannel_length = length(ofdm_tx_signal_coded) / num_subchannels;
% Initialize a cell array to store subchannels
ofdm_tx_subchannels = cell(1, num_subchannels);
ofdm_channel_subchannels = cell(1, num_subchannels);
% Split the OFDM transmitted signal into subchannels using a loop
for k = 1:num_subchannels
    start_idx = (k-1) * subchannel_length + 1;
    end_idx = k * subchannel_length;
    ofdm_tx_subchannels{k} = ofdm_tx_signal_coded(start_idx:end_idx);
end
for i = 1 : size(EbN0_dB,2)
    %% ********* Frequency selective fading Channel ********* %%
    binary_data_coded_received = [];
    decoded_data = [];
    % Generate the real and imaginary parts of the channel impulse response
    h_real = sqrt(1/2) * randn(num_subchannels, subchannel_length);
    h_imag = sqrt(1/2) * randn(num_subchannels, subchannel_length);
    SNR_linear = 10^(EbN0_dB(i)/10);
    N0 = Eb/SNR_linear;    % noise power
    variance = N0/2; % variance of the Gaussian noise
    % Generate the real part of the noise vector
    n_c = 0 + (1/sqrt(128))*sqrt(variance) * randn(1,length(ofdm_tx_signal_coded));
    % Generate the imaginary part of the noise vector
    n_s = 0 + (1/sqrt(128))*sqrt(variance) * randn(1,length(ofdm_tx_signal_coded));
    % Combine the real and imaginary parts to form the complex noise vector
    n = n_c + 1j * n_s;
    % Split the OFDM transmitted signal into subchannels and apply channel effect with noise
    for k = 1:num_subchannels
    start_idx = (k-1) * subchannel_length + 1;
    end_idx = k * subchannel_length;
    % Split the transmitted signal into subchannels
    ofdm_tx_subchannels{k} = ofdm_tx_signal_coded(start_idx:end_idx);
    % Form the complex channel impulse response for each subchannel
    h = h_real(k, :) + 1j * h_imag(k, :);
    % Apply the channel effect and add noise to each subchannel
    ofdm_channel_subchannels{k} = ifft(h .* fft(ofdm_tx_subchannels{k})) + n(start_idx:end_idx);
    end
    % Combine the received signals from all subchannels
    ofdm_channel_signal_combined_coded = [ofdm_channel_subchannels{:}];
    
    %% *************************  RECEIVER ************************* %%
    % Initialize cell arrays to store the received subchannels
    ofdm_rx_subchannels = cell(1, num_subchannels);
    % Loop through each subchannel to compensate for the channel gain
    for k = 1:num_subchannels
        start_idx = (k-1) * subchannel_length + 1;
        end_idx = k * subchannel_length;
        % Split the combined received signal into subchannels
        ofdm_channel_subchannel = ofdm_channel_signal_combined_coded(start_idx:end_idx);
        % Form the complex channel impulse response for each subchannel
        h = h_real(k, :) + 1j * h_imag(k, :);
        % Compensate for the channel gain at the receiver
        ofdm_rx_subchannels{k} = ifft(fft(ofdm_channel_subchannel) ./ h);
    end
    % Combine the compensated signals from all subchannels
    ofdm_rx_signal_coded = [ofdm_rx_subchannels{:}];

    % Remove Cyclic extension
    for j = 1:numel(coded_data_with_cp):numel(ofdm_channel_signal_combined_coded)
        coded_data_without_cp = ofdm_rx_signal_coded(cp_length +j : j+cp_length +IFFT_size-1);
        % Perform the 128-point FFT
        FFT_coded = fft(coded_data_without_cp , IFFT_size);
        % De-mapper
        QPSK_Recievedsignal_coded = QPSK_Gray_Demapper(FFT_coded');
        % De-interleaver
        QPSK_Recievedsignal_coded = QPSK_Recievedsignal_coded';
        DeInterleaved_Data1_coded = reshape(QPSK_Recievedsignal_coded,interleaver_size);
        DeInterleaved_Data_coded = reshape (DeInterleaved_Data1_coded.',1,[]);
        binary_data_coded_received = [binary_data_coded_received DeInterleaved_Data_coded];
        % Remove padding
        Decoded_Data1 = reshape(DeInterleaved_Data_coded(1 : numel(DeInterleaved_Data_coded)-4), repetition_factor, []);
        % Hard decision decoding
        Decoded_Data2 = sum(Decoded_Data1, 1) >= 2;
        decoded_data = [decoded_data Decoded_Data2];
    end
    % Calculating BER
    num_error_bits_coded = sum(binary_data_2 ~= decoded_data');
    BER_coded(i) = num_error_bits_coded/numBits_coded;
end
% Plotting BER
semilogy(EbN0_dB , BER_coded, 'b', 'linewidth', 1);
legend('QPSK no coding', 'QPSK with rate 1/3 repetition code');


function Interleaved_Data = Interleaver(binary_data, interleaver_size)
numInterleaverBits = prod(interleaver_size);
% Perform interleaving
numBlocks = length(binary_data) / numInterleaverBits;
Interleaved_Data = zeros(length(binary_data), 1);
for i = 1:numBlocks
    startIndex = (i-1)*numInterleaverBits + 1;
    endIndex = i*numInterleaverBits;
    dataBlock = binary_data(startIndex:endIndex);
    Inter = reshape(dataBlock, interleaver_size);
    Interleaved_Block = reshape(Inter.', 1, []);
    Interleaved_Data(startIndex:endIndex) = Interleaved_Block;
end
end
% QPSK Gray Mapper
% Maps binary data to QPSK symbols using Gray code mapping.
% Inputs:
%   binary_data: Binary data to be mapped to QPSK symbols.
% Outputs:
%   QPSK_Data_Mapped: QPSK symbols mapped from binary data using Gray code.
% Notes:
%   - The QPSK constellation table used in this function is defined as:
%     QPSK_Table = [-1-1i, -1+1i, 1-1i, 1+1i];
function QPSK_Data_Mapped = QPSK_Gray_Mapper(binary_data)
% Reshape the binary data to have each 2 consecutive bits as 1 symbol
% (Each Row is a Symbol)
QPSK_Table = [-1-1i, -1+1i, 1-1i, 1+1i];
QPSK_Data = reshape(binary_data, 2, []).';
numRows = size(QPSK_Data, 1);
QPSK_Data_Mapped = zeros(numRows, 1);
for i = 1:numRows
    % Convert binary data to decimal
    decimalValue = bi2de(QPSK_Data(i, :), 'left-msb');
    
    % Use decimal value as index to access corresponding symbol from QPSK table
    % Add 1 because MATLAB indices start from 1
    QPSK_Data_Mapped(i) = QPSK_Table(decimalValue + 1);
end
end
% QPSK Gray Demapper
% Demaps received QPSK symbols to binary data using Gray code demapping.
% Inputs:
%   QPSK_Data: Received QPSK symbols to be demapped to binary data.
% Outputs:
%   QPSK_Data_Demapped: Demapped binary data from received QPSK symbols.
% Notes:
%   - The QPSK constellation table used in this function is defined as:
%     QPSK_Table = [-1-1i, -1+1i, 1-1i, 1+1i];
function QPSK_Data_Demapped = QPSK_Gray_Demapper(QPSK_Data)
numBits = size(QPSK_Data,1)*2;
QPSK_Data_Demapped = zeros(size(QPSK_Data, 1), 2);
QPSK_Table = [-1-1i, -1+1i, 1-1i, 1+1i];
% Iterate over each received symbol
for i = 1:size(QPSK_Data, 1)
    % Calculate distance to each QPSK constellation point
    distances = abs(QPSK_Data(i) - QPSK_Table);
        
    % Find index of closest constellation point
    [~, index] = min(distances);

    % Convert the index to binary representation
    binary_QPSK = de2bi(index - 1, 2, 'left-msb');
    
    % Store the binary representation in the demapped array
    QPSK_Data_Demapped(i, :) = binary_QPSK;
end
QPSK_Data_Demapped = reshape(QPSK_Data_Demapped.',1,numBits)';
end