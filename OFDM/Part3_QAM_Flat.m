clear;
clc;
%.....................................................%
%                                                     %
%  No Coding 16-QAM over Rayleigh flat fading Channel %   
%                                                     %
%.....................................................%

%% ************************* Parameters ************************* %%
numBits = 262144;               % Number of bits
Eb = 1;                          % Bit energy
bits_per_symbol = 4;             % bits per symbol for 16 QAM
interleaver_size = [32, 16];     % Interleaver size
IFFT_size = 128;                 % Number of points for FFT and IFFT
cp_length = IFFT_size/4;         % Cyclic prefix length 25% FFT length
EbN0_dB = -5:1:20;               % Eb/N0 range in dB (SNR)
BER = zeros(size(EbN0_dB,2),1);  % Array to store BER at different SNR values
% Generate random binary data
binary_data = randi([0 1], numBits, 1);

%% ************************* TRANSMITTER ************************* %%
% Initalize the ofdm_tx_signal
ofdm_tx_signal = [];
for j = 1 : 512 : numBits
    % Interleaver
    Interleaved_Data = Interleaver(binary_data(j:j+511),interleaver_size);
    % Mapper
    QAM16 = QAM16_Mapper(Interleaved_Data).';
    % Perform the 128-point IFFT
    IFFT = ifft(QAM16, IFFT_size);
    % Add cyclic extension
    data_with_cp = [IFFT(end - cp_length + 1:end), IFFT];
    ofdm_tx_signal = [ofdm_tx_signal data_with_cp];
end

for i = 1 : size(EbN0_dB,2)
    %% *************************  Rayleigh flat fading Channel ************************* %%
    binary_data_received = [];
    % Generate the real part of the channel impulse response
    h_r = 0 + sqrt(1/2) * randn(1,length(ofdm_tx_signal));
    % Generate the imaginary part of the channel impulse response
    h_i = 0 + sqrt(1/2) * randn(1,length(ofdm_tx_signal));
    % Combine the real and imaginary parts to form the complex channel impulse response
    h = h_r + 1j * h_i;
    SNR_linear = 10^(EbN0_dB(i)/10);
    N0 = Eb/SNR_linear;    % noise power
    variance = 2.5*N0/2; % variance of the noise
    % Generate the real part of the noise vector
    n_c = 0 + (1/sqrt(128))*sqrt(variance) * randn(1,length(ofdm_tx_signal));
    % Generate the imaginary part of the noise vector
    n_s = 0 + (1/sqrt(128))*sqrt(variance) * randn(1,length(ofdm_tx_signal));
    % Combine the real and imaginary parts to form the complex noise vector
    n = n_c + 1j * n_s;
    ofdm_channel_signal = h.*ofdm_tx_signal + n;

    %% *************************  RECEIVER ************************* %%
    % Compensate for the channel gain at the receiver
    ofdm_rx_signal = ofdm_channel_signal./h;
    % Remove Cyclic extension
    for j = 1:numel(data_with_cp):numel(ofdm_channel_signal)
        data_without_cp = ofdm_rx_signal(cp_length +j : j+cp_length +IFFT_size-1);
        % Perform the 128-point FFT
        FFT = fft(data_without_cp , IFFT_size);
        % De-mapper
        QAM16_Recievedsignal = QAM16_Demapper(FFT.');
        % De-interleaver
        QAM16_Recievedsignal = QAM16_Recievedsignal';
        % Reverse the interleaver size when de-interleaving
        DeInterleaved_Data1 = reshape(QAM16_Recievedsignal,interleaver_size(2),interleaver_size(1));
        DeInterleaved_Data = reshape (DeInterleaved_Data1.',1,[]);
        binary_data_received = [binary_data_received DeInterleaved_Data];
    end
    binary_data_received = binary_data_received';
    % Calculating BER
    num_error_bits = sum(binary_data ~= binary_data_received);
    BER(i) = num_error_bits/numBits;
end
% Plotting BER
figure('Name' , 'BER of 16-QAM over Rayleigh flat fading channel');
semilogy(EbN0_dB , BER, 'r', 'linewidth', 1);
hold on;
title('OFDM System: BER of 16-QAM over Rayleigh flat fading channel');
xlabel('EB/No(dB)');  ylabel('BER');  grid on;
% Limit y-axis to 10^-3
ylim([1e-3, 1]);

%...................................................................%
%                                                                   %
% Rate 1/3 repetition code 16-QAM over Rayleigh flat fading Channel %
%                                                                   %
%...................................................................%

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
ofdm_tx_signal_coded = [];
for j = 1 : 504 : numBits_coded*repetition_factor
    % padding the coded data with zeros
    pad = zeros(8,1);
    coded_data = [binary_data_coded(j:j+503); pad];
    % Interleaver
    Interleaved_Data_coded = Interleaver(coded_data, interleaver_size);
    % Mapper (QPSK)
    QAM16_coded = QAM16_Mapper(Interleaved_Data_coded).';
    % Perform the 128-point IFFT
    IFFT_coded = ifft(QAM16_coded, IFFT_size);
    % Add cyclic prefix
    coded_data_with_cp = [IFFT_coded(end - cp_length + 1:end), IFFT_coded];
    ofdm_tx_signal_coded = [ofdm_tx_signal_coded coded_data_with_cp];
end

for i = 1 : size(EbN0_dB,2)
    %% *************************  Rayleigh flat fading Channel ************************* %%
    binary_data_coded_received = [];
    decoded_data = [];
    % Generate the real part of the channel impulse response
    h_r = 0 + sqrt(1/2) * randn(1,length(ofdm_tx_signal_coded));
    % Generate the imaginary part of the channel impulse response
    h_i = 0 + sqrt(1/2) * randn(1,length(ofdm_tx_signal_coded));
    % Combine the real and imaginary parts to form the complex channel impulse response
    h = h_r + 1j * h_i;
    SNR_linear = 10^(EbN0_dB(i)/10);
    N0 = Eb/SNR_linear;    % noise power
    variance = 2.5*N0/2; % variance of the noise
    % Generate the real part of the noise vector
    n_c = 0 + (1/sqrt(128))*sqrt(variance) * randn(1,length(ofdm_tx_signal_coded));
    % Generate the imaginary part of the noise vector
    n_s = 0 + (1/sqrt(128))*sqrt(variance) * randn(1,length(ofdm_tx_signal_coded));
    % Combine the real and imaginary parts to form the complex noise vector
    n = n_c + 1j * n_s;
    ofdm_channel_signal_coded = h.*ofdm_tx_signal_coded + n;
    
    %% *************************  RECEIVER ************************* %%
    % Compensate for the channel gain at the receiver
    ofdm_rx_signal_coded = ofdm_channel_signal_coded./h;

    % Remove Cyclic extension
    for j = 1:numel(coded_data_with_cp):numel(ofdm_channel_signal_coded)
        coded_data_without_cp = ofdm_rx_signal_coded(cp_length +j : j+cp_length +IFFT_size-1);
        % Perform the 128-point FFT
        FFT_coded = fft(coded_data_without_cp , IFFT_size);
        % De-mapper
        QAM16_Recievedsignal_coded = QAM16_Demapper(FFT_coded.');
        % De-interleaver
        QAM16_Recievedsignal_coded = QAM16_Recievedsignal_coded';
        % Reverse the interleaver size when de-interleaving
        DeInterleaved_Data1_coded = reshape(QAM16_Recievedsignal_coded,interleaver_size(2),interleaver_size(1));
        DeInterleaved_Data_coded = reshape (DeInterleaved_Data1_coded.',1,[]);
        binary_data_coded_received = [binary_data_coded_received DeInterleaved_Data_coded];
        % Remove padding
        Decoded_Data1 = reshape(DeInterleaved_Data_coded(1 : numel(DeInterleaved_Data_coded)-8), repetition_factor, []);
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
legend('16-QAM no coding', '16-QAM with rate 1/3 repetition code');


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
% 16-QAM Mapper
% QAM16_Mapper - Maps binary data to 16-QAM symbols
% Syntax: 
%   QAM16_Data_Mapped = QAM16_Mapper(binary_data)
% Input:
%   binary_data: A binary vector containing the input data bits.
% Output:
%   QAM16_Data_Mapped: A column vector containing the mapped 16-QAM symbols.
function QAM16_Data_Mapped = QAM16_Mapper(binary_data)
% Reshape the binary data to have each 4 consecutive bits as 1 symbol
% (Each Row is a Symbol)
QAM_16_Table = [-3-3i, -3-1i, -3+3i, -3+1i, -1-3i, -1-1i, -1+3i, -1+1i, 3-3i, 3-1i, 3+3i, 3+1i, 1-3i, 1-1i, 1+3i, 1+1i];
QAM_16_Data = reshape(binary_data, 4, []).';
% Map each row in QAM_16_Data to a value from the table
numRows = size(QAM_16_Data, 1);
QAM16_Data_Mapped = zeros(numRows, 1);
for i = 1:numRows
    % Convert binary data to decimal
    decimalValue = bi2de(QAM_16_Data(i, :), 'left-msb');
    
    % Use decimal value as index to access corresponding symbol from QAM table
    % Add 1 because MATLAB indices start from 1
    QAM16_Data_Mapped(i) = QAM_16_Table(decimalValue + 1);
end
end

% 16-QAM Demapper
% QAM16_Demapper - Demaps received 16-QAM symbols to binary data
% Syntax: 
%   QAM16_Data_Demapped = QAM16_Demapper(QAM16_data)
% Input:
%   QAM16_data: A column vector containing the received 16-QAM symbols.
% Output:
%   QAM16_Data_Demapped: A column vector containing the demapped binary data.
function QAM16_Data_Demapped = QAM16_Demapper(QAM16_data)
numBits = size(QAM16_data,1)*4;
QAM16_Data_Demapped = zeros(size(QAM16_data, 1), 4);
QAM_16_Table = [-3-3i, -3-1i, -3+3i, -3+1i, -1-3i, -1-1i, -1+3i, -1+1i, 3-3i, 3-1i, 3+3i, 3+1i, 1-3i, 1-1i, 1+3i, 1+1i];
% Iterate over each received symbol
for i = 1:size(QAM16_data, 1)
    % Calculate distance to each 16-QAM constellation points
    distances = abs(QAM16_data(i) - QAM_16_Table);
        
    % Find index of closest constellation point
    [~, index] = min(distances);

    % Convert the index to binary representation
    binary_16QAM = de2bi(index - 1, 4, 'left-msb');
    
    % Store the binary representation in the demapped array
    QAM16_Data_Demapped(i, :) = binary_16QAM;
end
QAM16_Data_Demapped = reshape(QAM16_Data_Demapped.',1,numBits)';
end