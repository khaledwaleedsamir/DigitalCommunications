%% ***********************************************************************
%  *        Variables initalization and binary data generation           *
%  ***********************************************************************
numBits = 384000; % Number of bits to transmit
EbN0_dB = -4:1:14; % Eb/N0 range in dB
binary_data = randi([0 1], numBits, 1); % Generate Binary Data

% Variables to Store BER for each Modulation Scheme
BER_BPSK = zeros(size(EbN0_dB,2),1);
BER_theoretical_BPSK = zeros(size(EbN0_dB,2),1);

BER_QPSK = zeros(size(EbN0_dB,2),1);
BER_QPSK_NoGray = zeros(size(EbN0_dB,2),1);
BER_theoretical_QPSK = zeros(size(EbN0_dB,2),1);

BER_8PSK = zeros(size(EbN0_dB,2),1);
BER_theoretical_8PSK = zeros(size(EbN0_dB,2),1);

BER_16QAM = zeros(size(EbN0_dB,2),1);
BER_theoretical_16QAM = zeros(size(EbN0_dB,2),1);

BER_BFSK = zeros(size(EbN0_dB,2),1);
BER_theoretical_BFSK = zeros(size(EbN0_dB,2),1);

%% ***********************************************************************
%  *                        Function Calls (Main)                        *
%  ***********************************************************************

% Map Binary Data to BPSK, QPSK, 8-PSK, 16-QAM
BPSK_Tx = BPSK_Mapper(binary_data);
QPSK_Tx = QPSK_Gray_Mapper(binary_data);
PSK8_Tx = PSK8_Mapper(binary_data);
QAM16_Tx = QAM16_Mapper(binary_data);
QPSK_NoGray_Tx = QPSK_NoGray_Mapper(binary_data);
BFSK_Tx = BFSK_Mapper(binary_data);

% iterate over all SNR values to get BER at each value
for i = 1 : size(EbN0_dB,2)
    % Display Iteration number in command window
    disp(['SNR Iteration number: ', num2str(i)]);
    % AWGN Channel Effect on sent Data In this model, the channel just adds 
    % noise to the transmitted signal.
    BPSK_Rx = Add_Noise(BPSK_Tx, EbN0_dB(i), 'BPSK');
    QPSK_Rx = Add_Noise(QPSK_Tx, EbN0_dB(i), 'QPSK');
    PSK8_Rx = Add_Noise(PSK8_Tx, EbN0_dB(i), '8PSK');
    QAM16_Rx = Add_Noise(QAM16_Tx, EbN0_dB(i), '16QAM');
    QPSK_NoGray_Rx = Add_Noise(QPSK_NoGray_Tx, EbN0_dB(i), 'QPSK');
    BFSK_Rx = Add_Noise(BFSK_Tx, EbN0_dB(i), 'BFSK');

    % Demap BPSK data to Binary Data
    binary_data_demapped_BPSK = BPSK_Demapper(BPSK_Rx);
    binary_data_demapped_QPSK = QPSK_Gray_Demapper(QPSK_Rx);
    binary_data_demapped_8PSK = PSK8_Demapper(PSK8_Rx);
    binary_data_demapped_16QAM = QAM16_Demapper(QAM16_Rx);
    binary_data_demapped_QPSK_NoGray = QPSK_NoGray_Demapper(QPSK_NoGray_Rx);
    binary_data_demapped_BFSK = BFSK_Demapper(BFSK_Rx);
    num_error_bits_BPSK = 0;
    num_error_bits_QPSK = 0;
    num_error_bits_8PSK = 0;
    num_error_bits_16QAM = 0;
    num_error_bits_QPSK_noGray = 0;
    num_error_bits_BFSK = 0;
    for j = 1:size(binary_data)
        if(binary_data_demapped_BPSK(j) ~= binary_data(j))
            num_error_bits_BPSK = num_error_bits_BPSK + 1;
        end
        if(binary_data_demapped_QPSK(j) ~= binary_data(j))
            num_error_bits_QPSK = num_error_bits_QPSK + 1;
        end
        if(binary_data_demapped_8PSK(j) ~= binary_data(j))
            num_error_bits_8PSK = num_error_bits_8PSK + 1;
        end
        if(binary_data_demapped_16QAM(j) ~= binary_data(j))
            num_error_bits_16QAM = num_error_bits_16QAM + 1;
        end
        if(binary_data_demapped_QPSK_NoGray(j) ~= binary_data(j))
            num_error_bits_QPSK_noGray = num_error_bits_QPSK_noGray + 1;
        end
        if(binary_data_demapped_BFSK(j) ~= binary_data(j))
            num_error_bits_BFSK = num_error_bits_BFSK + 1;
        end
    end

    BER_BPSK(i,1) = num_error_bits_BPSK/size(binary_data, 1);
    BER_QPSK(i,1) = num_error_bits_QPSK/size(binary_data, 1);
    BER_8PSK(i,1) = num_error_bits_8PSK/size(binary_data, 1);
    BER_16QAM(i,1) = num_error_bits_16QAM/size(binary_data ,1);
    BER_QPSK_NoGray(i,1) = num_error_bits_QPSK_noGray/size(binary_data,1);
    BER_BFSK(i,1) = num_error_bits_BFSK/size(binary_data,1);
   

    BER_theoretical_BPSK(i,1) = Calc_Theoretical_BER(EbN0_dB(i), 'BPSK');
    BER_theoretical_QPSK(i,1) = Calc_Theoretical_BER(EbN0_dB(i), 'QPSK');
    BER_theoretical_8PSK(i,1) = Calc_Theoretical_BER(EbN0_dB(i), '8PSK');
    BER_theoretical_16QAM(i,1) = Calc_Theoretical_BER(EbN0_dB(i), '16QAM');
    BER_theoretical_BFSK(i,1) = Calc_Theoretical_BER(EbN0_dB(i), 'BFSK');
end
%% ***********************************************************************
%  *                          Graphs Plotting                            *
%  ***********************************************************************

% Plot 1 BPSK
figure('Name' , 'BER of BPSK');
semilogy(EbN0_dB , BER_theoretical_BPSK, 'b --', 'linewidth', 1.5);
hold on;
semilogy(EbN0_dB , BER_BPSK, 'r', 'linewidth', 1);
hold off;
title('BER of BPSK');
xlabel('EB/No(dB)');  ylabel('BER');  grid on;
legend('BPSK theoretical','BPSK Practical');
% Limit y-axis to 10^-4
ylim([1e-4, 1]);

% Plot 2 QPSK
figure('Name' , 'BER of Gray QPSK');
semilogy(EbN0_dB , BER_theoretical_QPSK, 'b --', 'linewidth', 1.5);
hold on;
semilogy(EbN0_dB , BER_QPSK, 'r', 'linewidth', 1);
hold off;
title('BER of Gray QPSK');
xlabel('EB/No(dB)');  ylabel('BER');  grid on;
legend('QPSK theoretical','QPSK Practical');
% Limit y-axis to 10^-4
ylim([1e-4, 1]);

% Plot 3 8-PSK
figure('Name' , 'BER of Gray 8-PSK');
semilogy(EbN0_dB , BER_theoretical_8PSK, 'b --', 'linewidth', 1.5);
hold on;
semilogy(EbN0_dB , BER_8PSK, 'r', 'linewidth', 1);
hold off;
title('BER of Gray 8-PSK');
xlabel('EB/No(dB)');  ylabel('BER');  grid on;
legend('8-PSK theoretical','8-PSK Practical');
% Limit y-axis to 10^-4
ylim([1e-4, 1]);

% Plot 4 16-QAM
figure('Name' , 'BER of 16-QAM');
semilogy(EbN0_dB , BER_theoretical_16QAM, 'b --', 'linewidth', 1.5);
hold on;
semilogy(EbN0_dB , BER_16QAM, 'r', 'linewidth', 1);
hold off;
title('BER of 16-QAM');
xlabel('EB/No(dB)');  ylabel('BER');  grid on;
legend('16-QAM theoretical','16-QAM Practical');
% Limit y-axis to 10^-4
ylim([1e-4, 1]);

% Plot 5 BFSK 
figure('Name' , 'BER of BFSK');
semilogy(EbN0_dB , BER_theoretical_BFSK, 'b --', 'linewidth', 1.5);
hold on;
semilogy(EbN0_dB , BER_BFSK, 'r', 'linewidth', 1);
hold off;
title('BER of BFSK');
xlabel('EB/No(dB)');  ylabel('BER');  grid on;
legend('BFSK theoretical','BFSK Practical');
% Limit y-axis to 10^-4
ylim([1e-4, 1]);

% Plotting Theoretical BER
figure('Name' , 'Theoretical BER of different modulation Schemes');
semilogy(EbN0_dB , BER_theoretical_BPSK, 'g', 'linewidth', 1.5);
hold on;
semilogy(EbN0_dB , BER_theoretical_BFSK, 'y', 'linewidth', 1.5);
hold on;
semilogy(EbN0_dB , BER_theoretical_QPSK, 'r --', 'linewidth', 2);
hold on;
semilogy(EbN0_dB , BER_theoretical_8PSK, 'b', 'linewidth', 1.5);
hold on;
semilogy(EbN0_dB , BER_theoretical_16QAM, 'm', 'linewidth', 1.5);
title('Theoretical BER of different modulation schemes');
xlabel('EB/No(dB)');  ylabel('BER');  grid on;
legend('BPSK theoretical', 'BFSK theoretical', 'QPSK theoretical', '8-PSK theoretical', '16-QAM theoretical');
% Limit y-axis to 10^-4
ylim([1e-4, 1]);

% Plotting Actual BER
figure('Name' , 'BER of different modulation Schemes');
semilogy(EbN0_dB , BER_BPSK, 'g', 'linewidth', 1.5);
hold on;
semilogy(EbN0_dB , BER_BFSK, 'y', 'linewidth', 1.5);
hold on;
semilogy(EbN0_dB , BER_QPSK, 'r --', 'linewidth', 2);
hold on;
semilogy(EbN0_dB , BER_8PSK, 'b', 'linewidth', 1.5);
hold on;
semilogy(EbN0_dB , BER_16QAM, 'm', 'linewidth', 1.5);
title('BER of different modulation schemes');
xlabel('EB/No(dB)');  ylabel('BER');  grid on;
legend('BPSK', 'BFSK', 'QPSK', '8-PSK', '16-QAM');
% Limit y-axis to 10^-4
ylim([1e-4, 1]);

% Plotting BER of QPSK with and without Gray coding
figure('Name' , 'BER of QPSK with and without Gray coding');
semilogy(EbN0_dB , BER_QPSK, 'b', 'linewidth', 1);
hold on;
semilogy(EbN0_dB , BER_QPSK_NoGray, 'r', 'linewidth', 1);
title('BER of QPSK');
xlabel('EB/No(dB)');  ylabel('BER');  grid on;
legend('QPSK with Gray coding', 'QPSK without Gray coding');
% Limit y-axis to 10^-4
ylim([1e-4, 1]);

% plotting the PSD for BFSK
bit_energy = 1;        % Bit Energy
num_bits = 100;        % number of randomly generated bits
num_waveforms = 5000; % number of realizations in the ensemble
bit_duration = 0.1;    % bit duration in seconds
sampling_freq = 100;   % sampling frequency in Hz

BFSK_plot_psd(bit_energy, num_bits, num_waveforms, bit_duration, sampling_freq);

%% ***********************************************************************
%  *                             Functions                               *
%  ***********************************************************************

% BPSK Mapper
% Maps binary data to BPSK symbols.
% Inputs:
%   binary_data: Binary data to be mapped to BPSK symbols.
% Outputs:
%   BPSK_Data_Mapped: BPSK symbols mapped from binary data.
function BPSK_Data_Mapped = BPSK_Mapper(binary_data)
BPSK_Data_Mapped = 2*binary_data-1;
end

% BPSK Demapper
% Demaps BPSK symbols to binary data.
% Inputs:
%   BPSK_Data: BPSK symbols to be demapped to binary data.
% Outputs:
%   BPSK_Data_Demapped: Binary data demapped from BPSK symbols.
function BPSK_Data_Demapped = BPSK_Demapper(BPSK_Data)
BPSK_Data_Demapped = zeros(size(BPSK_Data,1),1);
for i=1:size(BPSK_Data,1)
        if BPSK_Data(i)>=0  
            BPSK_Data_Demapped(i)=1;
        else
            BPSK_Data_Demapped(i)=0;
        end
end
end

% BFSK Mapper Function
% Maps binary data to BFSK symbols.
% Inputs:
%   binary_data: Binary input data.
% Outputs:
%   BFSK_Data_Mapped: Mapped BFSK symbols.
function BFSK_Data_Mapped = BFSK_Mapper(binary_data)
Eb_BFSK = 1;
BFSK_Data_Mapped = sqrt(Eb_BFSK)*binary_data + sqrt(Eb_BFSK)*1i*(~binary_data);
end

% Demaps BFSK symbols to binary data.
% Inputs:
%   BFSK_Data: BFSK input symbols.
% Outputs:
%   BFSK_Data_Demapped: Demapped binary data.
function BFSK_Data_Demapped = BFSK_Demapper(BFSK_Data)
BFSK_Data_Demapped = zeros(size(BFSK_Data,1),1);
BFSK_Table = [0+1i, 1+0i];
for i = 1:size(BFSK_Data, 1)
    % Calculate distance to each QPSK constellation point
    distances = abs(BFSK_Data(i) - BFSK_Table);
        
    % Find index of closest constellation point
    [~, index] = min(distances);

    % Convert the index to binary representation
    binary_BFSK = de2bi(index - 1, 1, 'left-msb');
    
    % Store the binary representation in the demapped array
    BFSK_Data_Demapped(i, :) = binary_BFSK;
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

% QPSK Mapper (Without Gray Code)
% Maps binary data to QPSK symbols
% Inputs:
%   binary_data: Binary data to be mapped to QPSK symbols.
% Outputs:
%   QPSK_Data_Mapped: QPSK symbols mapped from binary data.
% Notes:
%   - The QPSK constellation table used in this function is defined as:
%     QPSK_Table = [-1-1i, -1+1i, 1+1i, 1-1i];
function QPSK_Data_Mapped = QPSK_NoGray_Mapper(binary_data)
% Reshape the binary data to have each 2 consecutive bits as 1 symbol
% (Each Row is a Symbol)
QPSK_Table = [-1-1i, -1+1i, 1+1i, 1-1i];
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

% QPSK Demapper (Without Gray Code)
% Demaps received QPSK symbols to binary data 
% Inputs:
%   QPSK_Data: Received QPSK symbols to be demapped to binary data.
% Outputs:
%   QPSK_Data_Demapped: Demapped binary data from received QPSK symbols.
% Notes:
%   - The QPSK constellation table used in this function is defined as:
%     QPSK_Table = [-1-1i, -1+1i, 1+1i, 1-1i];
function QPSK_Data_Demapped = QPSK_NoGray_Demapper(QPSK_Data)
numBits = size(QPSK_Data,1)*2;
QPSK_Data_Demapped = zeros(size(QPSK_Data, 1), 2);
QPSK_Table = [-1-1i, -1+1i, 1+1i, 1-1i];
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

% 8-PSK Mapper
% PSK8_Data_Mapped = PSK8_Mapper(binary_data)
% This function maps binary data to 8-PSK (Phase Shift Keying) symbols.
% Inputs:
%   binary_data: A binary vector containing the data bits to be mapped.
% Output:
%   PSK8_Data_Mapped: A column vector containing the mapped 8-PSK symbols.
% Notes:
%   - The function employs a mapping table (PSK8_Table) containing the
%     complex symbols corresponding to each phase in 8-PSK modulation.
% PSK8_Table = [1+0i, x+x*1i, -x+x*1i, 0+1i, x-x*1i, 0-1i, -1+0i, -x-x*1i];
%  where x = sqrt(2)/2;
function PSK8_Data_Mapped = PSK8_Mapper(binary_data)
x = sqrt(2)/2;
PSK8_Table = [1+0i, x+x*1i, -x+x*1i, 0+1i, x-x*1i, 0-1i, -1+0i, -x-x*1i];
PSK8_Data = reshape(binary_data, 3, []).';
numRows = size(PSK8_Data, 1);
PSK8_Data_Mapped = zeros(numRows, 1);
for i = 1:numRows
    % Convert binary data to decimal
    decimalValue = bi2de(PSK8_Data(i, :), 'left-msb');
    
    % Use decimal value as index to access corresponding symbol from 8PSK
    % table and add 1 because MATLAB indices start from 1
    PSK8_Data_Mapped(i) = PSK8_Table(decimalValue + 1);
end
end

% 8-PSK demapper
% PSK8_Data_Demapped = PSK8_Demapper(PSK8_data)
% This function demaps received 8-PSK symbols to binary data bits.
% Inputs:
%   PSK8_data: A column vector containing the received 8-PSK symbols.
% Output:
%   PSK8_Data_Demapped: A binary matrix containing the demapped binary bits
function PSK8_Data_Demapped = PSK8_Demapper(PSK8_data)
numBits = size(PSK8_data,1)*3;
PSK8_Data_Demapped = zeros(size(PSK8_data, 1), 3);
x = sqrt(2)/2;
PSK8_Table = [1+0i, x+x*1i, -x+x*1i, 0+1i, x-x*1i, 0-1i, -1+0i, -x-x*1i];
% Iterate over each received symbol
for i = 1:size(PSK8_data, 1)
    % Calculate distance to each 8PSK constellation point
    distances = abs(PSK8_data(i) - PSK8_Table);
        
    % Find index of closest constellation point
    [~, index] = min(distances);

    % Convert the index to binary representation
    binary_8PSK = de2bi(index - 1, 3, 'left-msb');
    
    % Store the binary representation in the demapped array
    PSK8_Data_Demapped(i, :) = binary_8PSK;
end
PSK8_Data_Demapped = reshape(PSK8_Data_Demapped.',1,numBits)';
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

% Add_Noise - Add Gaussian noise to the transmitted signal
% Syntax: 
%   dataWithNoise = Add_Noise(mapped_data, SNR_dB, ModulationScheme)
% Input:
%   mapped_data: The transmitted symbols before adding noise.
%   SNR_dB: The signal-to-noise ratio in decibels (dB).
%   ModulationScheme: The modulation scheme used for transmission 
%           options -> ('BPSK', 'QPSK', '8PSK', 'BFSK', or '16QAM').
% Output:
%   dataWithNoise: The transmitted symbols with Gaussian noise added.
% Note: 
%   - Variance of the Gaussian noise is calculated using the following
%     Formula: Variance = sqrt((No/2)*(Eavg/NumOfBitsPerSymbol))
function dataWithNoise = Add_Noise(mapped_data, SNR_dB, ModulationScheme)
switch ModulationScheme
    case 'BPSK'
        SNR_linear = 10^(SNR_dB/10);
        Eb_BPSK = 1;
        N0 = Eb_BPSK/SNR_linear; % noise power
        variance = sqrt(N0/2); % variance of the Gaussian noise
        real_noise = variance.* randn(1 , size(mapped_data,1));
        % Add the Generated noise to the data
        dataWithNoise = mapped_data + real_noise';
    case 'QPSK'
        SNR_linear = 10^(SNR_dB/10);
        Eb_QPSK = 1;
        N0 = Eb_QPSK/SNR_linear; % noise power
        variance = sqrt(N0/2); % variance of the Gaussian noise
        real_noise = variance.* randn(1 , size(mapped_data,1));
        img_noise = variance.* randn(1 , size(mapped_data,1));
        dataWithNoise = mapped_data + (real_noise+(1i*img_noise))';
    case '8PSK'
        SNR_linear = 10^(SNR_dB/10);
        Eb_MPSK = 1;
        N0 = Eb_MPSK/(SNR_linear); % noise power;
        variance = sqrt(N0/(2*3)); % variance of the Gaussian noise
        real_noise = variance.* randn(1 , size(mapped_data,1));
        img_noise = variance.* randn(1 , size(mapped_data,1));
        dataWithNoise = mapped_data + (real_noise+(1i*img_noise))';
    case '16QAM'
        SNR_linear = 10^(SNR_dB/10);
        Eb_16QAM = 1; % normalized
        N0 = Eb_16QAM/SNR_linear; % noise power
        variance = sqrt(2.5*N0/2); % variance of the Gaussian noise
        real_noise = variance.* randn(1 , size(mapped_data,1));
        img_noise = variance.* randn(1 , size(mapped_data,1));
        dataWithNoise = mapped_data + (real_noise+(1i*img_noise))';
    case 'BFSK'
        SNR_Linear = 10^(SNR_dB/10);
        Eb_BFSK = 1;
        N0 = Eb_BFSK/SNR_Linear;
        variance = sqrt(N0/2);
        real_noise = variance.* randn(1, size(mapped_data,1));
        img_noise = variance.* randn(1, size(mapped_data,1));
        dataWithNoise = mapped_data + (real_noise+(1i*img_noise))';
    otherwise
        disp('Add_Noise: Choose a valid Modulation Scheme!')
end
end


% Calc_Theoretical_BER - Calculate the theoretical bit error rate (BER)
% Syntax: 
%   BER_theoretical = Calc_Theoretical_BER(SNR_dB, ModulationScheme)
% Input:
%   SNR_dB: The signal-to-noise ratio (SNR) in decibels (dB).
%   ModulationScheme: The modulation scheme used for transmission 
%           options -> ('BPSK', 'QPSK', '8PSK', 'BFSK' or '16QAM').
% Output:
%   BER_theoretical: The theoretical bit error rate (BER) for the specified
%                    modulation scheme and SNR.
function BER_theoretical = Calc_Theoretical_BER(SNR_dB, ModulationScheme)
switch ModulationScheme
    case 'BPSK'
        SNR_linear = 10^(SNR_dB/10);
        Eb_BPSK = 1;
        N0 = Eb_BPSK/SNR_linear;
        BER_theoretical = 0.5*erfc(sqrt(Eb_BPSK/N0));
    case 'QPSK'
        SNR_linear = 10^(SNR_dB/10);
        Eb_QPSK = 1;
        N0 = Eb_QPSK/SNR_linear;
        BER_theoretical = 0.5*erfc(sqrt(Eb_QPSK/N0));
    case '8PSK'
        SNR_linear = 10^(SNR_dB/10);
        Eb_8PSK = 1;
        N0 = Eb_8PSK/SNR_linear;
        BER_theoretical = erfc(sqrt(3*Eb_8PSK/N0)*sin(pi/8))/3;
    case '16QAM'
        SNR_linear = 10^(SNR_dB/10);
        Eb_QAM = 1;  
        N0 = Eb_QAM/SNR_linear;
        BER_theoretical = (1.5/4)*erfc(sqrt(1/(2.5*N0)));
    case 'BFSK'
        SNR_linear = 10^(SNR_dB/10);
        Eb_BFSK = 1;
        N0 = Eb_BFSK/SNR_linear;
        BER_theoretical = (0.5)*erfc(sqrt(Eb_BFSK/(2*N0)));
    otherwise
        disp('Calc_Theoretical_BER: Choose a valid Modulation Scheme!')
end
end

% BFSK_plot_psd
% Description:
%   This function plots the PSD of BFSK theoretically and using random
%   generated data to plot a practical plot.
% Syntax:
%   BFSK_plot_psd(bit_energy, num_bits, num_waveforms, bit_duration, sampling_frequency)
% Inputs:
%   - bit_energy: Energy per bit (scalar)
%   - num_bits: Number of bits per waveform (scalar)
%   - num_waveforms: Number of waveforms (scalar)
%   - bit_duration: Duration of one bit in seconds (scalar)
%   - sampling_frequency: Sampling frequency in Hz (scalar)
% Output: None
function BFSK_plot_psd(bit_energy, num_bits, num_waveforms, bit_duration, sampling_frequency)
fs = sampling_frequency;
Tb = bit_duration;
samples = floor(fs*Tb);
Eb = bit_energy;
% generating binary data, extra bit for random initial start
binary_data = randi([0, 1], num_waveforms, num_bits + 1); 
% repeating each bit by the amount of samples
binary_data = repelem(binary_data, 1, samples);
% time vector
t = 0:(1/fs):(Tb-0.01);
% BFSK data before applying random initial start
BFSK_data_no_rand_start = zeros(num_waveforms,(num_bits+1)*samples); 
% BFSK data after applying random initial start
BFSK_data = zeros(num_waveforms,(num_bits*samples));
% BFSK baseband equivalent representation
S1BB = sqrt(Eb*2/Tb);
S2BB = S1BB*(cos(2*pi*(1/Tb)*t)+1i*sin(2*pi*(1/Tb)*t));

sample_count=1;
for i=1:num_waveforms
    for j=1:(num_bits+1)*samples
        if(binary_data(i,j)==1)
            BFSK_data_no_rand_start(i,j)=S1BB;
        else
            if(sample_count>samples)
                sample_count=1;
            end
            BFSK_data_no_rand_start(i,j)=S2BB(sample_count);
            sample_count=sample_count+1;
        end
    end
end

% Applying the random initial start
for i=1:num_waveforms
    % Generate Random initial shift
    initial_shift = randi([0, samples]);
    % Apply the time shifts
    BFSK_data(i, :) = BFSK_data_no_rand_start(i, initial_shift+1:initial_shift+num_bits*samples);
end
% autocorrelation array initalization
BFSK_auto_corr = zeros(1, num_bits*samples);

% Loop over tau to calculate the statistical autocorrelation
for tau = (-num_bits*samples/2+1):num_bits*samples/2
     BFSK_auto_corr(tau + num_bits*samples/2)= sum(conj(BFSK_data(:, num_bits*samples/2)) .* BFSK_data(:, num_bits*samples/2 + tau))/ num_waveforms;
end

% getting the FFT of the BFSK autocorrelation (PSD)
fft_BFSK = abs(fft(BFSK_auto_corr));
N = length(BFSK_auto_corr); 
% frequency vector
f = (-N/2:N/2-1) * fs/N *Tb;
% define offset to shift the actual PSD over the theoretical PSD to compare
offset = 0;
f_shifted=f-offset;

% frequency vector for theoretical PSD plot
f_theoretical = (-num_bits*samples/2:num_bits*samples/2-1) * fs/(num_bits*samples) ;
% theoretical PSD calculation
delta_1 = 99 * double(abs(f_theoretical - 0.5 / Tb) < 0.01);
delta_2 = 99 * double(abs(f_theoretical + 0.5 / Tb) < 0.01);  
theoretical_psd = ((2/Tb) * (delta_1 + delta_2)) + ...
                  ((8 * cos(pi * Tb * f_theoretical).^2) ./ (pi^2 * (4 * Tb^2 * f_theoretical.^2 - 1).^2));
% plotting
figure;
plot(f_shifted, fftshift(fft_BFSK)/fs,'r', 'linewidth', 1);
hold on;
plot(f_theoretical*Tb,theoretical_psd,'b', 'linewidth', 1);
title('PSD of BFSK');
xlabel('1/Tb');
ylabel('Magnitude');
legend ('Actual PSD', 'Theoretical PSD');
ylim ([0 2])
end