% Generate a random signal of length 8192
N = 8192;
x_n = randn(1, N);

% Measure the execution time of the DFT
tic;            % Start timing
X_k = DFT(x_n); % Compute the DFT using the custom function
time_DFT = toc; % End timing and get elapsed time
fprintf('Execution time of custom DFT: %.6f seconds\n', time_DFT);

% Measure the execution time of the FFT
tic;                    % Start timing
X_k_builtin = fft(x_n); % Compute the FFT using the built-in function
time_builtin = toc;     % End timing and get elapsed time
fprintf('Execution time of built-in FFT: %.6f seconds\n', time_builtin);
fprintf('FFT is %.6f times faster than DFT \n',time_DFT/time_builtin);

function X_k = DFT(x_n)
    N = length(x_n);
    X_k = zeros(1,N);
    n = 0:N-1;
    for k = 0:N-1
        X_k(k+1) = sum(x_n.*exp(-2j * pi * k * n / N));
    end
end