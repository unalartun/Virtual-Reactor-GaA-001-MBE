clear; clc; close all;
%% ---------- Системный таймер ----------
tic; 
%% ---------- Параметры модели ----------
L = 50;
MaxH = 40;
H = ones(L, L);
TypeMap = zeros(L, L, MaxH); 
x_Al = 0; 

% --- ПАРАМЕТРЫ ВРЕМЕНИ И ПОТОКА ---
max_time_sec = 5.0; 
F_GroupIII = 1.0;   
As_excess = 25;     
TargetLayers = ceil(max_time_sec * F_GroupIII); % Авто-определение кол-ва слоев

nu0 = 1e13; 
difmatrx = [1 0; 0 -1; -1 0; 0 1]; 
T = 850; kB = 8.617e-5; 

% Энергетические барьеры
Ediff_Ga = 1.1; Ediff_Al = 1.5; Ediff_As = 2.0; 
E_lat_G3 = 0.24; E_lat_G5 = 0.15; E_ES = 0.20; 
max_jumps = 400; 

%% ----------- Инициализация логов (ИСПРАВЛЕНО) -----------
R_mat = zeros(L, L, 4); 
R_node_total = zeros(L, L);
R_total_system = 0; 
t_global = 0;
step = 0; 

RHEED_intensity = [];
RHEED_ML = [];
Roughness_history = [];
Layer_Coverage_History = []; % Будет расширяться динамически

rheed_every = 5; % Оптимальный шаг для Фурье

%% ----------- Главный цикл -----------
while t_global < max_time_sec
    dt_ads = -log(rand()) / (F_GroupIII * L^2);
    if (t_global + dt_ads) > max_time_sec
        dt_ads = max_time_sec - t_global;
        t_global = max_time_sec;
        break; 
    end

    t_global = t_global + dt_ads;
    step = step + 1;

%     --- АДСОРБЦИЯ ---
    rx = randi(L); ry = randi(L);
    if mod(H(rx, ry), 2) == 0 
        new_h = H(rx, ry) + 1;
        H(rx, ry) = new_h;
        TypeMap(rx, ry, new_h) = (rand() < x_Al) + 1; 
        [R_mat, R_node_total, R_total_system] = update_patch(rx, ry, L, H, TypeMap, R_mat, R_node_total, R_total_system, nu0, kB, T, Ediff_Ga, Ediff_Al, Ediff_As, E_lat_G3, E_lat_G5, E_ES);
    end

%     --- ДИФФУЗИЯ ---
    t_int = 0; jump_count = 0;
    while t_int < dt_ads && jump_count < max_jumps && R_total_system > 0
        dt_jump = -log(rand()) / R_total_system;
        if (t_int + dt_jump) > dt_ads, break; end
        t_int = t_int + dt_jump; jump_count = jump_count + 1;

        idx = find(cumsum(R_node_total(:)) >= rand() * R_total_system, 1);
        [r, c] = ind2sub([L, L], idx);
        m = find(cumsum(R_mat(r, c, :)) >= rand() * R_node_total(r, c), 1);

        nr = mod(r + difmatrx(m,1) - 1, L) + 1;
        nc = mod(c + difmatrx(m,2) - 1, L) + 1;

        old_h = H(r, c); new_h = H(nr, nc) + 1;
        TypeMap(nr, nc, new_h) = TypeMap(r, c, old_h);
        TypeMap(r, c, old_h) = 0;
        H(r, c) = old_h - 1; H(nr, nc) = new_h;

        [R_mat, R_node_total, R_total_system] = update_patch(r, c, L, H, TypeMap, R_mat, R_node_total, R_total_system, nu0, kB, T, Ediff_Ga, Ediff_Al, Ediff_As, E_lat_G3, E_lat_G5, E_ES);
        [R_mat, R_node_total, R_total_system] = update_patch(nr, nc, L, H, TypeMap, R_mat, R_node_total, R_total_system, nu0, kB, T, Ediff_Ga, Ediff_Al, Ediff_As, E_lat_G3, E_lat_G5, E_ES);
    end
% 
%     --- МЫШЬЯК ---
    for k = 1:As_excess
        ax = randi(L); ay = randi(L);
        if mod(H(ax, ay), 2) == 1 
            new_h = H(ax, ay) + 1; H(ax, ay) = new_h; 
            TypeMap(ax, ay, new_h) = 0; 
            [R_mat, R_node_total, R_total_system] = update_patch(ax, ay, L, H, TypeMap, R_mat, R_node_total, R_total_system, nu0, kB, T, Ediff_Ga, Ediff_Al, Ediff_As, E_lat_G3, E_lat_G5, E_ES);
        end
    end

    % --- СБОР ДАННЫХ ---
    if mod(step, rheed_every) == 0
        RHEED_ML(end+1) = t_global; 
        S = sum(exp(1i * pi * H(:) / 2)); 
        RHEED_intensity(end+1) = abs(S / L^2)^2;
        Roughness_history(end+1) = std(H(:), 1);

        current_cov = zeros(MaxH, 1);
        for h = 1:MaxH
            current_cov(h) = sum(H(:) >= h) / L^2;
        end
        Layer_Coverage_History(:, end+1) = current_cov;
    end
end

% % ---------- АНАЛИЗ ФУРЬЕ (НОВОЕ) ----------
% Удаляем постоянную составляющую и интерполируем для равномерности
Fs = 1 / mean(diff(RHEED_ML)); % Частота дискретизации
L_fft = length(RHEED_intensity);
NFFT = 2^nextpow2(L_fft); 
Y_fft = fft(RHEED_intensity - mean(RHEED_intensity), NFFT) / L_fft;
f = (Fs/2) * linspace(0, 1, NFFT/2 + 1);
Amplitude = 2 * abs(Y_fft(1:NFFT/2 + 1));

% % ---------- Визуализация ----------
figure('Color', 'w', 'Position', [100 100 1100 800]);

% RHEED
subplot(2,2,1);
plot(RHEED_ML, RHEED_intensity, 'b', 'LineWidth', 2); grid on;
title(' RHEED'); xlabel('Time(s)'); ylabel('Intensity');

% Roughness
subplot(2,2,2);
plot(RHEED_ML, Roughness_history, 'r', 'LineWidth', 2); grid on;
title('RMS'); xlabel('Time (s)');

% FFT Spectrum
subplot(2,2,3);
plot(f, Amplitude, 'k', 'LineWidth', 2); grid on;
title('Frequancy'); xlabel('Fr(Hz)'); ylabel('Ampitude');
xlim([0 F_GroupIII*2]); % Ограничиваем частоту для наглядности

% Layer Coverage
subplot(2,2,4); hold on; grid on;
for h = 1:min(TargetLayers*2, MaxH)
    color = [0.5 0.5 0.5]; 
    if mod(h,2)==1, color = [1 0 0]; else, color = [0 0 1]; end
    plot(RHEED_ML, Layer_Coverage_History(h, :), 'Color', color);
end
title('Layer covarage'); xlabel('Time (s)'); ylabel('\theta');

fprintf('Моделирование завершено за %.2f с.\n', toc);

% % ---------- Функции ----------
function [R_mat, R_node_total, R_total_system] = update_patch(x, y, L, H, TypeMap, R_mat, R_node_total, R_total_system, nu0, kB, T, Ediff_Ga, Ediff_Al, Ediff_As, E_lat_G3, E_lat_G5, E_ES)
    dx = [0, 1, -1, 0, 0, 2, -2, 0, 0];
    dy = [0, 0, 0, 1, -1, 0, 0, 2, -2];
    for k = 1:length(dx)
        r = mod(x + dx(k) - 1, L) + 1;
        c = mod(y + dy(k) - 1, L) + 1;
        R_total_system = R_total_system - R_node_total(r, c);
        [R_mat, R_node_total] = calc_single_node(r, c, L, H, TypeMap, R_mat, R_node_total, nu0, kB, T, Ediff_Ga, Ediff_Al, Ediff_As, E_lat_G3, E_lat_G5, E_ES);
        R_total_system = R_total_system + R_node_total(r, c);
    end
    if R_total_system < 1e-18, R_total_system = 0; end
end

function [R_mat, R_node_total] = calc_single_node(r, c, L, H, TypeMap, R_mat, R_node_total, nu0, kB, T, Ediff_Ga, Ediff_Al, Ediff_As, E_lat_G3, E_lat_G5, E_ES)
    R_mat(r, c, :) = 0;
    h_rc = H(r,c);
    if h_rc == 0, R_node_total(r, c) = 0; return; end
    current_type = mod(h_rc, 2);
    my_id = TypeMap(r, c, h_rc);
    if current_type == 1, axis_strong = 'x'; 
        if my_id == 2, Ediff_base = Ediff_Al; else, Ediff_base = Ediff_Ga; end
    else, axis_strong = 'y'; Ediff_base = Ediff_As; end
    nb_x = (H(r, mod(c, L) + 1) >= h_rc) + (H(r, mod(c-2, L) + 1) >= h_rc);
    nb_y = (H(mod(r, L) + 1, c) >= h_rc) + (H(mod(r-2, L) + 1, c) >= h_rc);
    E_lat = (current_type == 1) * E_lat_G3 + (current_type == 0) * E_lat_G5;
    if axis_strong == 'x', E_binding = nb_x * E_lat * 1.2 + nb_y * E_lat;
    else, E_binding = nb_x * E_lat + nb_y * E_lat * 1.2; end
    dif = [1 0; -1 0; 0 1; 0 -1];
    for m = 1:4
        nr = mod(r + dif(m,1) - 1, L) + 1;
        nc = mod(c + dif(m,2) - 1, L) + 1;
        if H(nr, nc) < h_rc && current_type ~= mod(H(nr, nc), 2)
            cur_E = Ediff_base + E_binding;
            if H(nr, nc) < h_rc - 1, cur_E = cur_E + E_ES; end
            R_mat(r, c, m) = nu0 * exp(-cur_E / (kB * T));
        end
    end
    R_node_total(r, c) = sum(R_mat(r, c, :));
end