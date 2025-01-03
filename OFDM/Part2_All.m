numBits = 1000000;                          % Number of bits to transmit
Eb = 1;                                    % Bit Energy
binary_data = randi([0 1], numBits, 1);    % Generate random binary data bitstream
EbN0_dB = -4:1:14;                         % Eb/N0 range in dB (SNR)

BER_BPSK = zeros(size(EbN0_dB,2),1);        % Array to store BER at different SNR values
BER_BPSK_Coded = zeros(size(EbN0_dB,2),1);  % Array to store BER at different SNR values
BER_QPSK = zeros(size(EbN0_dB,2),1);        % Array to store BER at different SNR values
BER_QPSK_Coded = zeros(size(EbN0_dB,2),1);  % Array to store BER at different SNR values
BER_16QAM = zeros(size(EbN0_dB,2),1);       % Array to store BER at different SNR values
BER_16QAM_Coded = zeros(size(EbN0_dB,2),1); % Array to store BER at different SNR values

% Applying rate 1/3 repetition code
binary_data_coded = repelem(binary_data, 3);

% BPSK Mapper
BPSK_data = sqrt(Eb)*(2*binary_data-1);
BPSK_data_coded = sqrt(Eb)*(2*binary_data_coded-1);
% QPSK Mapper
QPSK_data = QPSK_Gray_Mapper(binary_data);
QPSK_data_coded = QPSK_Gray_Mapper(binary_data_coded);
% 16QAM Mapper
QAM16_data = QAM16_Mapper(binary_data);
QAM16_data_coded = QAM16_Mapper(binary_data_coded);

for i = 1 : size(EbN0_dB,2)
    % Generate the real part of the channel impulse response
    h_r = 0 + sqrt(1/2) * randn(length(BPSK_data), 1);
    h_r_coded = 0 + sqrt(1/2) * randn(length(BPSK_data_coded), 1);
    % Generate the imaginary part of the channel impulse response
    h_i = 0 + sqrt(1/2) * randn(length(BPSK_data), 1);
    h_i_coded = 0 + sqrt(1/2) * randn(length(BPSK_data_coded), 1);
    % Combine the real and imaginary parts to form the complex channel impulse response
    h = h_r + 1j * h_i;
    h_coded = h_r_coded + 1j * h_i_coded;
    SNR_linear = 10^(EbN0_dB(i)/10);
    N0 = Eb/SNR_linear;    % noise power
    variance = N0/2; % variance of the Gaussian noise
    % Generate the real part of the noise vector
    n_c = 0 + sqrt(variance) * randn(length(BPSK_data), 1);
    n_c_coded = 0 + sqrt(variance) * randn(length(BPSK_data_coded), 1);
    % Generate the imaginary part of the noise vector
    n_s = 0 + sqrt(variance) * randn(length(BPSK_data), 1);
    n_s_coded = 0 + sqrt(variance) * randn(length(BPSK_data_coded), 1);
    % Combine the real and imaginary parts to form the complex noise vector
    n = n_c + 1j * n_s;
    n_coded = n_c_coded + 1j * n_s_coded;
    % Received data
    datareceived = h.*BPSK_data + n;
    datareceived_coded = h_coded.*BPSK_data_coded + n_coded;
    % Compensate for the channel gain at the receiver
    datareceived_equalized = datareceived./h;
    datareceived_equalized_coded = datareceived_coded./h_coded;
    % Demapping bitstream
    binary_data_demapped = real(datareceived_equalized) > 0;
    binary_data_demapped_coded = real(datareceived_equalized_coded) > 0;
    % Hard decision decoding of the rate 1/3 repetition code
    hard_decoding_reshape = reshape(binary_data_demapped_coded, 3, []).';
    hard_decoding_sum = sum(hard_decoding_reshape, 2);
    hard_decoding_binary_data = hard_decoding_sum >= 2;
    % Calculating BER
    num_error_bits = sum(binary_data ~= binary_data_demapped);
    num_error_bits_coded = sum(binary_data ~= hard_decoding_binary_data);
    BER_BPSK(i) = num_error_bits/numBits;
    BER_BPSK_Coded(i) = num_error_bits_coded/numBits;
end
for i = 1 : size(EbN0_dB,2)
    % Generate the real part of the channel impulse response
    h_r = 0 + sqrt(1/2) * randn(length(QPSK_data), 1);
    h_r_coded = 0 + sqrt(1/2) * randn(length(QPSK_data_coded), 1);
    % Generate the imaginary part of the channel impulse response
    h_i = 0 + sqrt(1/2) * randn(length(QPSK_data), 1);
    h_i_coded = 0 + sqrt(1/2) * randn(length(QPSK_data_coded), 1);
    % Combine the real and imaginary parts to form the complex channel impulse response
    h = h_r + 1j * h_i;
    h_coded = h_r_coded + 1j * h_i_coded;
    SNR_linear = 10^(EbN0_dB(i)/10);
    N0 = Eb/SNR_linear;    % noise power
    variance = N0/2; % variance of the Gaussian noise
    % Generate the real part of the noise vector
    n_c = 0 + sqrt(variance) * randn(length(QPSK_data), 1);
    n_c_coded = 0 + sqrt(variance) * randn(length(QPSK_data_coded), 1);
    % Generate the imaginary part of the noise vector
    n_s = 0 + sqrt(variance) * randn(length(QPSK_data), 1);
    n_s_coded = 0 + sqrt(variance) * randn(length(QPSK_data_coded), 1);
    % Combine the real and imaginary parts to form the complex noise vector
    n = n_c + 1j * n_s;
    n_coded = n_c_coded + 1j * n_s_coded;
    % Received data
    datareceived = h.*QPSK_data + n;
    datareceived_coded = h_coded.*QPSK_data_coded + n_coded;
    % Compensate for the channel gain at the receiver
    datareceived_equalized = datareceived./h;
    datareceived_equalized_coded = datareceived_coded./h_coded;
    % Demapping bitstream
    binary_data_demapped = QPSK_Gray_Demapper(datareceived_equalized);
    binary_data_demapped_coded = QPSK_Gray_Demapper(datareceived_equalized_coded);
    % Hard decision decoding of the rate 1/3 repetition code
    hard_decoding_reshape = reshape(binary_data_demapped_coded, 3, []).';
    hard_decoding_sum = sum(hard_decoding_reshape, 2);
    hard_decoding_binary_data = hard_decoding_sum >= 2;
    % Calculating BER
    num_error_bits = sum(binary_data ~= binary_data_demapped);
    num_error_bits_coded = sum(binary_data ~= hard_decoding_binary_data);
    BER_QPSK(i) = num_error_bits/numBits;
    BER_QPSK_Coded(i) = num_error_bits_coded/numBits;
end
for i = 1 : size(EbN0_dB,2)
    % Generate the real part of the channel impulse response
    h_r = 0 + sqrt(1/2) * randn(length(QAM16_data), 1);
    h_r_coded = 0 + sqrt(1/2) * randn(length(QAM16_data_coded), 1);
    % Generate the imaginary part of the channel impulse response
    h_i = 0 + sqrt(1/2) * randn(length(QAM16_data), 1);
    h_i_coded = 0 + sqrt(1/2) * randn(length(QAM16_data_coded), 1);
    % Combine the real and imaginary parts to form the complex channel impulse response
    h = h_r + 1j * h_i;
    h_coded = h_r_coded + 1j * h_i_coded;
    SNR_linear = 10^(EbN0_dB(i)/10);
    N0 = Eb/SNR_linear;    % noise power
    variance = 2.5*N0/2; % variance of the Gaussian noise
    % Generate the real part of the noise vector
    n_c = 0 + sqrt(variance) * randn(length(QAM16_data), 1);
    n_c_coded = 0 + sqrt(variance) * randn(length(QAM16_data_coded), 1);
    % Generate the imaginary part of the noise vector
    n_s = 0 + sqrt(variance) * randn(length(QAM16_data), 1);
    n_s_coded = 0 + sqrt(variance) * randn(length(QAM16_data_coded), 1);
    % Combine the real and imaginary parts to form the complex noise vector
    n = n_c + 1j * n_s;
    n_coded = n_c_coded + 1j * n_s_coded;
    % Received data
    datareceived = h.*QAM16_data + n;
    datareceived_coded = h_coded.*QAM16_data_coded + n_coded;
    % Compensate for the channel gain at the receiver
    datareceived_equalized = datareceived./h;
    datareceived_equalized_coded = datareceived_coded./h_coded;
    % Demapping bitstream
    binary_data_demapped = QAM16_Demapper(datareceived_equalized);
    binary_data_demapped_coded = QAM16_Demapper(datareceived_equalized_coded);
    % Hard decision decoding of the rate 1/3 repetition code
    hard_decoding_reshape = reshape(binary_data_demapped_coded, 3, []).';
    hard_decoding_sum = sum(hard_decoding_reshape, 2);
    hard_decoding_binary_data = hard_decoding_sum >= 2;
    % Calculating BER
    num_error_bits = sum(binary_data ~= binary_data_demapped);
    num_error_bits_coded = sum(binary_data ~= hard_decoding_binary_data);
    BER_16QAM(i) = num_error_bits/numBits;
    BER_16QAM_Coded(i) = num_error_bits_coded/numBits;
end
% Plotting BER
figure('Name' , 'BER over Rayleigh flat fading channel');
semilogy(EbN0_dB , BER_BPSK, 'r', 'linewidth', 1.5);
hold on;
semilogy(EbN0_dB , BER_QPSK, 'b--', 'linewidth', 1.5);
hold on;
semilogy(EbN0_dB , BER_16QAM, 'g', 'linewidth', 1.5);
hold on;
semilogy(EbN0_dB , BER_BPSK_Coded, 'm', 'linewidth', 1.5);
hold on;
semilogy(EbN0_dB , BER_QPSK_Coded, 'y--', 'linewidth', 1.5);
hold on;
semilogy(EbN0_dB , BER_16QAM_Coded, 'k', 'linewidth', 1.5);
hold off;
title('BER over Rayleigh flat fading channel');
xlabel('EB/No(dB)');  ylabel('BER');  grid on;
legend('BPSK no coding','QPSK no coding','16-QAM no coding','BPSK with rate 1/3 repetition code','QPSK with rate 1/3 repetition code','16-QAM with rate 1/3 repetition code');
% Limit y-axis to 10^-3
ylim([1e-3, 1]);



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