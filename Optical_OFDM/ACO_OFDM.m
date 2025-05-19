%% Define parameters
EbN0_dB = 0:2:30;                            % Eb/N0 range in dB
N = 128;                                     % Subcarriers (same N for FFT/IFFT size)
half_N = N/2;                                % Half subcarriers
numSymbols = 32*32*2;                        % Number of OFDM symbols
numBits = 2*2 * (N/2 - 1) * numSymbols;      % number of bits to be multiple of subcarriers
BER_values_odd = zeros(length(EbN0_dB), 1);  % Store BER for each SNR
BER_values_even = zeros(length(EbN0_dB), 1); % Store BER for each SNR

% Generate the binary data
binary_data = randi([0 1], numBits, 1);
for j = 1:length(EbN0_dB)
    %% Transmitter
    % Generate 16-QAM symbols for ACO-OFDM
    dataSymbols_ACO = QAM16_Mapper(binary_data);
    % Reshape for OFDM symbols (ensuring correct size)
    dataSymbols_ACO = reshape(dataSymbols_ACO, (N/4), []);
    % Initialize OFDM frame with zeros
    X_ACO_odd = zeros(N, size(dataSymbols_ACO, 2));
    X_ACO_even= zeros(N, size(dataSymbols_ACO, 2));
    % Note: MATLAB uses 1-based indexing, while in comm subcarrier indexing starts from 0.
    % So, comm subcarrier index 0 (DC) → MATLAB index 1
    %     comm subcarrier index 1 (1st odd) → MATLAB index 2
    %     comm subcarrier index 2 (1st even) → MATLAB index 3
    % This means to modulate odd subcarriers (in comm terms), we use even MATLAB indices (2, 4, 6, ...)
    % and for even comm subcarriers, we use odd MATLAB indices (3, 5, 7, ...).
    odd_indices = 2:2:(N/2);
    even_indices = 1:2:(N/2);
    % Load data on odd subcarriers only
    X_ACO_odd(odd_indices, :) = dataSymbols_ACO;
    % Load data onto even subcarriers
    X_ACO_even(even_indices, :) = dataSymbols_ACO;
    % Hermitian symmetry for real-valued IFFT
    hermitian_odd = N - odd_indices + 2;
    X_ACO_odd(hermitian_odd, :) = conj(X_ACO_odd(odd_indices, :));
    %hermitian_even = N - even_indices + 2;
    %X_ACO_even(hermitian_even, :) = conj(X_ACO_even(even_indices, :));
    % IFFT to generate real-valued time-domain signal
    ofdm_signal_ACO_odd = ifft(X_ACO_odd, N);
    ofdm_signal_ACO_even = ifft(X_ACO_even, N, 'symmetric');
    % Clipping at zero
    ofdm_signal_ACO_odd_clipped = max(ofdm_signal_ACO_odd, 0);
    ofdm_signal_ACO_even_clipped = max(ofdm_signal_ACO_even, 0);
    %% Channel
    SNR_linear = 10^(EbN0_dB(j)/10);
    % Noise power
    N0 = 1/SNR_linear;
    % Variance of the Gaussian noise
    variance = 2.5*N0/2; % 16-QAM carries 4 bits per symbol.
    % Generate the noise vector
    n_dsc = length(odd_indices);
    n = sqrt(n_dsc/(N*N))*sqrt(variance) * randn(size(ofdm_signal_ACO_odd_clipped));
    % Assuming AWGN channel so noise is simply added
    aco_ofdm_odd_channel_signal = ofdm_signal_ACO_odd_clipped + n;
    aco_ofdm_even_channel_signal = ofdm_signal_ACO_even_clipped + n;
    %% Receiver
    % Use the anti-symmetry to reconstruct the signal
    y1 = aco_ofdm_odd_channel_signal(1:half_N, :);
    y2 = aco_ofdm_odd_channel_signal(half_N+1:end, :);
    ofdm_signal_ACO_odd_reconstructed = [y1 - y2; -(y1 - y2)];
    % Convert to frequency domain
    Y_odd = fft(ofdm_signal_ACO_odd_reconstructed, N);
    Y_even = fft(aco_ofdm_even_channel_signal, N);
    % Obtain the received symbols
    % subcarriers will be halved so we multiply by 2 after demodulation
    received_symbols_odd = Y_odd(odd_indices, :);
    received_symbols_even = Y_even(even_indices, :);
    % Reshape to a column vector for demodulation
    received_symbols_odd = received_symbols_odd(:);
    received_symbols_even = received_symbols_even(:);
    % QAM Demapping
    received_bits_odd = QAM16_Demapper(received_symbols_odd);
    received_bits_even = QAM16_Demapper(received_symbols_even);
    BER_values_odd(j) = sum(binary_data ~= received_bits_odd) / length(binary_data);
    BER_values_even(j) = sum(binary_data ~= received_bits_even)/ length(binary_data);
end
% Plot the BER curves
figure;
semilogy(EbN0_dB, BER_values_odd, 'o-r', 'LineWidth', 2, 'DisplayName', 'ACO-OFDM (Odd subcarriers)');
hold on;
semilogy(EbN0_dB, BER_values_even, 's-b', 'LineWidth', 2, 'DisplayName', 'ACO-OFDM (Even subcarriers)');
grid on;
% Add plot formatting
xlabel('Eb/N0 (dB)');
ylabel('Bit Error Rate (BER)');
title('BER vs Eb/N0 for ACO-OFDM (Odd vs Even Subcarriers)');
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