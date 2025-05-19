%% Define parameters
EbN0_dB = 0:2:30;                       % Eb/N0 range in dB
N = 128;                                % Subcarriers (same N for FFT/IFFT size)
numSymbols = 32*32*2;                   % Number of OFDM symbols
numBits = 2*2 * (N/2 - 1) * numSymbols; % number of bits to be multiple of subcarriers
BER_values = zeros(length(EbN0_dB), 1); % Store BER for each SNRR
% Generate the binary data
binary_data = randi([0 1], numBits, 1);

for j = 1:length(EbN0_dB)
    %% Transmitter
    % Generate 16-QAM symbols for ACO-OFDM
    dataSymbols_U = QAM16_Mapper(binary_data);
    % Reshape for OFDM symbols (ensuring correct size)
    dataSymbols_U = reshape(dataSymbols_U, N/2 - 1, []);
    % Construct the Hermitian symmetric OFDM frame
    X = zeros(N, size(dataSymbols_U, 2));
    % U_k for k=1 to N/2-1
    X(2:N/2, :) = dataSymbols_U;
    X(1, :) = 0;       % 0 DC
    X(N/2 + 1, :) = 0; % 0 Nyquist
    % Define indices for k = 2 to N/2 - 1
    data_indices = 2:N/2;
    % Their Hermitian counterparts: N - k + 2
    hermitian_indices = N - data_indices + 2;
    % Apply symmetry
    X(hermitian_indices, :) = conj(X(data_indices, :));
    % Apply IFFT to get real-valued time domain signal
    ofdm_signal = ifft(X, N);
    x_pos_tx = max(ofdm_signal, 0); % Positive part
    x_neg = min(ofdm_signal, 0);    % Negative part (still negative)
    x_neg_tx = -x_neg;         % Flip negative to positive
    % Concatenate for transmission
    %x_U_OFDM = [x_pos_tx; x_neg_tx];
    %% Channel
    SNR_linear = 10^(EbN0_dB(j)/10);
    % Noise power
    N0 = 1/SNR_linear;
    % Variance of the Gaussian noise
    variance = 2.5*N0/2; % 16-QAM carries 4 bits per symbol.
    % Generate the noise vector
    n_dsc = length(data_indices);
    n1 = sqrt(n_dsc/(N*N))*sqrt(variance).*randn(size(x_pos_tx));
    n2 = sqrt(n_dsc/(N*N))*sqrt(variance).*randn(size(x_neg_tx));
    % Assuming AWGN channel so noise is simply added
    %u_ofdm_channel_signal = x_U_OFDM + n;
    u_ofdm_frame1 = x_pos_tx + n1;
    u_ofdm_frame2 = x_neg_tx + n2;
    %% Receiver
    %recieved_signal = u_ofdm_channel_signal;
    %y_pos = recieved_signal(1:N, :); % First half (original positive)
    %y_neg_flipped = recieved_signal(N+1:end, :);  % Second half (flipped negative)
    % Re-flip the negative part to its original polarity
    %y_neg = -y_neg_flipped;
    % Reconstruct the full bipolar OFDM signal
    y_reconstructed = u_ofdm_frame1 -u_ofdm_frame2;  
    % Step 4: FFT to go to frequency domain
    Y = fft(y_reconstructed, N);
    % Reshape to a column vector for demodulation
    received_symbols_u = Y(data_indices, :);
    received_symbols_u = received_symbols_u(:);
    % QAM Demapping
    received_bits_u = QAM16_Demapper(received_symbols_u);
    BER_values(j) = sum(binary_data ~= received_bits_u)/ length(binary_data);
end
% Plot the BER curves
figure;
semilogy(EbN0_dB, BER_values, 'o-r', 'LineWidth', 2, 'DisplayName', 'U-OFDM (16-QAM)');
grid on;
% Add plot formatting
xlabel('Eb/N0 (dB)');
ylabel('Bit Error Rate (BER)');
title('BER vs Eb/N0 for U-OFDM (16-QAM)');
legend('Location', 'southwest');
ylim([1e-4 1]);

%% Required Fuctions
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