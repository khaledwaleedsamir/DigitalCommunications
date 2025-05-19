%% Define parameters
SNR_dB = 0:2:30;                                         % Eb/N0 range in dB
N = 128;                                                 % Subcarriers (same N for FFT/IFFT size)
bias_dB = 0:3:12;                                        % DC Bias range in dB
numSymbols = 32*32*2;                                    % Number of OFDM symbols
numBits = 2*2*(N/2 - 1)*numSymbols;                      % number of bits to be multiple of subcarriers
BER_all_biases = zeros(length(bias_dB), length(SNR_dB)); % Store BER for all biases
% Generate the binary data
binary_data = randi([0 1], numBits, 1);
biased_symbols = cell(1, length(bias_dB));   % Store one biased symbol per bias level
clipped_symbols = cell(1, length(bias_dB));  % Store one clipped symbol per bias level
%% Transmitter
% Generate QPSK symbols
dataSymbols = QPSK_Gray_Mapper(binary_data);
% Reshape for OFDM symbols
dataSymbols = reshape(dataSymbols, N/2 - 1, []);
% Construct the Hermitian symmetric OFDM frame
X = zeros(N, size(dataSymbols, 2));
% U_k for k=1 to N/2-1
X(2:N/2, :) = dataSymbols;
X(1, :) = 0;       % 0 DC
X(N/2 + 1, :) = 0; % 0 Nyquist
% Define indices for k = 2 to N/2 - 1
data_indices = 2:N/2;
% Their Hermitian counterparts: N - k + 2
hermitian_indices = N - data_indices + 2;
% Apply symmetry
X(hermitian_indices, :) = conj(X(data_indices, :));
% Apply IFFT to get real-valued time domain signal
n_dsc = length(data_indices);
ofdm_signal = ifft(X, N);
sqrt(mean(abs(ofdm_signal(:)).^2));
ofdm_signal = ofdm_signal / sqrt(mean(abs(ofdm_signal(:)).^2));
sqrt(mean(abs(ofdm_signal(:)).^2));
original_OFDM_symbol = ofdm_signal(:,1); % Original unclipped
% Compute RMS using standard deviation
RMS = std(ofdm_signal(:));
% Compute useful power from original time-domain OFDM signal (unbiased)
P_useful = mean(abs(ofdm_signal(:)).^2);
power_efficiency = zeros(length(bias_dB), 1);
for i = 1:length(bias_dB)
    BER_values = zeros(1, length(SNR_dB)); % Store BER for each SNR value
    dc_bias = (10^(bias_dB(i)/20))*RMS;
    OFDM_Signal_Biased = ofdm_signal + dc_bias;
    biased_symbols{i} = OFDM_Signal_Biased(:,1);               % Before clipping
    % Clipping step (this will introduce clipping noise If the DC 
    % bias is high enough, the clipping noise can be neglected) 
    OFDM_Signal_Biased_clipped = max(OFDM_Signal_Biased, 0);
    clipped_symbols{i} = OFDM_Signal_Biased_clipped(:,1);      % After clipping
    % Compute total power after DC bias and clipping
    P_total = mean(abs(OFDM_Signal_Biased_clipped(:)).^2);
    % Calculate power efficiency
    power_efficiency(i) = (P_useful / P_total)*100;
    fprintf('=============================================== \n');
    fprintf('Iteration number %d \n',i);
    fprintf('bias_dB = %.4f \n',bias_dB(i));
    fprintf('RMS = %.4f \n', RMS);
    fprintf('DC Bias = %.4f \n',dc_bias);

    for j = 1:length(SNR_dB)
        %% Channel
        % Convert SNR from dB to linear
        SNR_linear = 10^(SNR_dB(j)/10);
        N0 = 1/SNR_linear;
        % Variance of the Gaussian noise
        variance = N0/2;
        % Generate the noise vector
        noise = sqrt(variance) * randn(size(OFDM_Signal_Biased_clipped));
        % Assuming AWGN channel so noise is simply added
        ofdm_channel_signal = OFDM_Signal_Biased_clipped + noise;
        %ofdm_channel_signal = awgn(OFDM_Signal_Biased_clipped, SNR_dB(j));
        %% Receiver
        OFDM_Signal_Received  = ofdm_channel_signal - dc_bias;
        Y = fft(OFDM_Signal_Received, N); % Convert to frequency domain
        Y_data = Y(2:N/2, :);  % Keep only the first half (excluding DC and Nyquist)
        binary_data_received = QPSK_Gray_Demapper(Y_data(:));
        BER_all_biases(i, j) = sum(binary_data ~= binary_data_received) / length(binary_data_received);
    end
end
colors = lines(length(bias_dB)); % Generate different colors for each 
figure;
semilogy(SNR_dB, BER_all_biases(1,:), '-o', 'LineWidth', 2, 'DisplayName', 'DC Bias = 0 dB', 'Color', colors(1, :));
hold on;
semilogy(SNR_dB, BER_all_biases(2,:), '-o', 'LineWidth', 2, 'DisplayName', 'DC Bias = 3 dB', 'Color', colors(2, :));
hold on;
semilogy(SNR_dB, BER_all_biases(3,:), '-o', 'LineWidth', 2, 'DisplayName', 'DC Bias = 6 dB', 'Color', colors(3, :));
hold on;
semilogy(SNR_dB, BER_all_biases(4,:), '-o', 'LineWidth', 2, 'DisplayName', 'DC Bias = 9 dB', 'Color', colors(4, :));
hold on;
semilogy(SNR_dB, BER_all_biases(5,:), '-o', 'LineWidth', 2, 'DisplayName', 'DC Bias = 12 dB', 'Color', colors(5, :));
grid on;
% Add plot formatting
xlabel('SNR (dB)');
ylabel('Bit Error Rate (BER)');
title('BER vs. SNR for DCO-OFDM with Different DC Bias Levels');
legend('Location', 'southwest');
ylim([1e-4 1]);

figure;
t = 1:N;  % Time samples
colors = lines(length(bias_dB));
for i = 1:length(bias_dB)
    subplot(length(bias_dB), 1, i);
    hold on;
    stem(t, real(biased_symbols{i}), 'r', 'DisplayName', 'Biased (Before Clipping)', 'LineWidth', 1);
    stem(t, real(clipped_symbols{i}), 'b', 'DisplayName', 'Clipped (After Clipping)', 'LineWidth', 1);
    title(sprintf('DC Bias = %d dB', bias_dB(i)));
    xlabel('Time Samples');
    ylabel('Amplitude');
    grid on;
    legend;
end
sgtitle('Effect of DC Bias on DCO-OFDM Time-Domain Signal (One OFDM Symbol)');

target_snr_idx = find(SNR_dB == 8); 
BER_vs_efficiency = BER_all_biases(:, target_snr_idx);

figure;
plot(power_efficiency, BER_vs_efficiency, '-x', 'LineWidth', 2);
xlabel('Power Efficiency (%)');
ylabel('Bit Error Rate (BER)');
title('BER vs Power Efficiency for DCO-OFDM');
grid on;
set(gca, 'FontSize', 12, 'FontWeight', 'bold');
grid on;
box on;

% Add DC bias labels to each marker
for i = 1:length(bias_dB)
    text(power_efficiency(i), BER_vs_efficiency(i), ...
        sprintf('@DC Bias = %.0f dB', bias_dB(i)), ...
        'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', ...
        'FontSize', 12);
end
% Create the figure
figure;
hold on;
grid on;
box on;

% Scatter plot with color mapped to DC bias
scatter(power_efficiency, BER_vs_efficiency, 80, bias_dB, 'filled');

% Smooth connecting curve (interpolated)
xx = linspace(min(power_efficiency), max(power_efficiency), 200);
yy = interp1(power_efficiency, BER_vs_efficiency, xx, 'pchip');
plot(xx, yy, '--', 'Color', [0.3 0.3 0.3], 'LineWidth', 1.5);

% Add DC bias labels at each point
for i = 1:length(bias_dB)
    text(power_efficiency(i), BER_vs_efficiency(i), ...
        sprintf('%.1f dB', bias_dB(i)), ...
        'VerticalAlignment', 'bottom', ...
        'HorizontalAlignment', 'right', ...
        'FontSize', 10, 'FontWeight', 'bold');
end

% Label axes and title
xlabel('Power Efficiency (%)', 'FontSize', 12);
ylabel('Bit Error Rate (BER)', 'FontSize', 12);
title('BER vs Power Efficiency for DCO-OFDM', 'FontSize', 14, 'FontWeight', 'bold');

% Optional: set y-axis to log scale if BER varies a lot
set(gca, 'YScale', 'log');

hold off;

%% Required Fuctions
% QPSK Gray Mapper
% Maps binary data to QPSK symbols using Gray code mapping.
% Inputs:
%   binary_data: Binary data to be mapped to QPSK symbols.
% Outputs:
%   QPSK_Data_Mapped: QPSK symbols mapped from binary data using Gray code.
% Notes:
%   - The QPSK constellation table used in this function is defined as:
%     QPSK_Table = [-1-1i, -1+1i, 1-1i, 1+1i]
function QPSK_Data_Mapped = QPSK_Gray_Mapper(binary_data)
% Reshape the binary data to have each 2 consecutive bits as 1 symbol
% (Each Row is a Symbol)
QPSK_Table = [-1-1i, -1+1i, 1-1i, 1+1i]/sqrt(2);
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
QPSK_Table = [-1-1i, -1+1i, 1-1i, 1+1i]/sqrt(2);
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