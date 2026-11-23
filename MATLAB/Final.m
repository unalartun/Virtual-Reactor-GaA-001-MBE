clear; clc; close all;

%% =========================================================================
% БЛОК 1: ГЕНЕРАЦИЯ ДАТАСЕТА (parfor)
% =========================================================================
fprintf('Начинаем генерацию обучающей выборки...\n');
tic;

N_samples = 10; 
L_size = 30;    
max_time_sec = 2;
nu0 = 1e13;
kB = 8.617e-5;

% Преаллокация массивов для нейросети
X_features = zeros(N_samples, 63); % 3 параметра + 30 FFT + 30 шероховатость
Y_targets = zeros(N_samples, 2);   % [log10(k_Ga), log10(k_As)]

% Параметры для генерации (задаем диапазоны случайных величин)
T_range = [800, 810];        % Температура (К)
F_range = [1.0, 1.05];        % Поток (ML/s)
E_Ga_range = [1.1, 1.15];     % Барьер диффузии Ga (эВ)
E_As_range = [2.1, 2.15 ];     % Барьер диффузии As (эВ)
As_ex_range = [10, 11];      % Избыток As

% Настраиваем parfor (если Parallel Computing Toolbox установлен)
parfor i = 1:N_samples
    % 1. Генерируем случайные физические условия для итерации
    T_curr = T_range(1) + rand() * diff(T_range);
    F_curr = F_range(1) + rand() * diff(F_range);
    E_Ga_curr = E_Ga_range(1) + rand() * diff(E_Ga_range);
    E_As_curr = E_As_range(1) + rand() * diff(E_As_range);
    As_ex_curr = round(As_ex_range(1) + rand() * diff(As_ex_range));
    
    % Вычисляем ИСТИННЫЕ значения k (целевые переменные)
    k_Ga_true = nu0 * exp(-E_Ga_curr / (kB * T_curr));
    k_As_true = nu0 * exp(-E_As_curr / (kB * T_curr));
    
    % 2. Запускаем "черный ящик" KMC
    [rheed_t, rheed_int, rough_hist] = run_kmc_iteration(L_size, max_time_sec, F_curr, As_ex_curr, T_curr, E_Ga_curr, E_As_curr);
    
    % 3. Обработка сигналов (Приведение к единому размеру для Нейросети)
    % Интерполируем шероховатость на 30 фиксированных точек
    t_fixed = linspace(0.1, max_time_sec, 30);
    rough_fixed = interp1(rheed_t, rough_hist, t_fixed, 'pchip', 'extrap');
    
    % Фурье преобразование RHEED
    rheed_interp = interp1(rheed_t, rheed_int, linspace(0, max_time_sec, 256), 'pchip');
    rheed_detrend = rheed_interp - mean(rheed_interp);
    Y_fft = abs(fft(rheed_detrend) / 256);
    fft_features = Y_fft(2:31); % Берем первые 30 значимых гармоник (без DC)
    
    % 4. Сохраняем вектор признаков (X) и целей (Y)
    % X: [T, F, As_ex, FFT_1...FFT_30, Rough_1...Rough_30]
    X_features(i, :) = [T_curr, F_curr, As_ex_curr, fft_features, rough_fixed];
    
    % Y: предсказываем логарифмы констант!
    Y_targets(i, :) = [log10(k_Ga_true), log10(k_As_true)];
end

fprintf('Генерация завершена за %.2f с.\n', toc);

%% =========================================================================
% БЛОК 2: ОБУЧЕНИЕ НЕЙРОСЕТИ
% =========================================================================
fprintf('Обучение нейросети...\n');
X = X_features'; % MATLAB требует размерность [Features x Samples]
Y = Y_targets';  % [Targets x Samples]

% Архитектура: 2 скрытых слоя по 32 и 16 нейронов
net = fitnet([32, 16]);
net.trainParam.showWindow = false; % Отключаем всплывающее окно для скорости

% Разбивка выборки: 70% обучение, 15% валидация, 15% тест
net.divideParam.trainRatio = 0.70;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;

% Обучаем сеть
[net, tr] = train(net, X, Y);

% Получаем предсказания на всей выборке
Y_pred = net(X);

%% =========================================================================
% БЛОК 3: МЕТРИКИ И ВИЗУАЛИЗАЦИЯ ДЛЯ ДИПЛОМА
% =========================================================================
% Возвращаем значения из логарифмов обратно в нормальный вид
k_real = 10.^Y;
k_pred = 10.^Y_pred;

% Расчет коэффициента детерминации R^2 (Достоверность)
R2_Ga = 1 - sum((k_real(1,:) - k_pred(1,:)).^2) / sum((k_real(1,:) - mean(k_real(1,:))).^2);
R2_As = 1 - sum((k_real(2,:) - k_pred(2,:)).^2) / sum((k_real(2,:) - mean(k_real(2,:))).^2);

fprintf('=== МЕТРИКИ КАЧЕСТВА ===\n');
fprintf('R^2 for Gs: %.4f\n', R2_Ga);
fprintf('R^2 for As: %.4f\n', R2_As);

% --- График Классическая корреляция (Идеал для научных статей) ---
figure('Color', 'w', 'Position', [100, 100, 900, 400]);
subplot(1,2,1);
scatter(k_real(1,:), k_pred(1,:), 40, 'b', 'filled', 'MarkerEdgeColor', 'k'); hold on;
plot([min(k_real(1,:)), max(k_real(1,:))], [min(k_real(1,:)), max(k_real(1,:))], 'r--', 'LineWidth', 2);
title(sprintf('Ga diffusion k_{Ga} (R^2 = %.2f)', R2_Ga));
xlabel('True vallue (KMC)'); ylabel('Predictin (NN)'); grid on; set(gca, 'XScale', 'log', 'YScale', 'log');

subplot(1,2,2);
scatter(k_real(2,:), k_pred(2,:), 40, 'g', 'filled', 'MarkerEdgeColor', 'k'); hold on;
plot([min(k_real(2,:)), max(k_real(2,:))], [min(k_real(2,:)), max(k_real(2,:))], 'r--', 'LineWidth', 2);
title(sprintf('Ar diffusion k_{As} (R^2 = %.2f)', R2_As));
xlabel('True vallue (KMC)'); ylabel('Prediction (NN)'); grid on; set(gca, 'XScale', 'log', 'YScale', 'log');

% --- График 2: 3D Зависимость (Поток, Температура -> k_Ga) ---
figure('Color', 'w', 'Position', [200, 200, 800, 600]);
scatter3(X_features(:,1), X_features(:,2), k_real(1,:)', 100, 'b', 'o', 'filled'); hold on;
scatter3(X_features(:,1), X_features(:,2), k_pred(1,:)', 80, 'r', 'x', 'LineWidth', 2);
title('For k_{Ga} = f(T, F_{III})');
xlabel('Тemperature, К'); ylabel('Flux, ML/s'); zlabel('diffusion rate k_{Ga}, с^{-1}');
legend('KMC model', 'NN prediction', 'Location', 'best');
grid on; set(gca, 'ZScale', 'log'); view(-30, 30);

%% =========================================================================
% ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (KMC ЯДРО)
% =========================================================================
function [RHEED_ML, RHEED_intensity, Roughness_history] = run_kmc_iteration(L, max_time, F, As_excess, T, Ediff_Ga, Ediff_As)
    % Базовые константы
    nu0 = 1e13; kB = 8.617e-5; Ediff_Al = 1.5; E_lat_G3 = 0.24; E_lat_G5 = 0.15; E_ES = 0.20;
    difmatrx = [1 0; -1 0; 0 1; 0 -1];
    
    H = zeros(L); MaxH = 40; TypeMap = zeros(L, L, MaxH);
    R_mat = zeros(L, L, 4); R_node_total = zeros(L, L); R_total_system = 0;
    t_global = 0; step = 0; rheed_every = 10;
    
    RHEED_intensity = []; RHEED_ML = []; Roughness_history = [];
    
    while t_global < max_time
        dt_ads = -log(rand()) / (F * L^2);
        if (t_global + dt_ads) > max_time
            break; 
        end
        t_global = t_global + dt_ads;
        step = step + 1;

        % Адсорбция
        rx = randi(L); ry = randi(L);
        if mod(H(rx, ry), 2) == 0 
            new_h = H(rx, ry) + 1; H(rx, ry) = new_h; TypeMap(rx, ry, new_h) = 1;
            [R_mat, R_node_total, R_total_system] = update_patch_local(rx, ry, L, H, TypeMap, R_mat, R_node_total, R_total_system, nu0, kB, T, Ediff_Ga, Ediff_Al, Ediff_As, E_lat_G3, E_lat_G5, E_ES);
        end

        % Диффузия (ограничена 50 прыжками для скорости генерации выборки)
        t_int = 0; jump_count = 0;
        while t_int < dt_ads && jump_count < 200 && R_total_system > 0
            dt_jump = -log(rand()) / R_total_system;
            if (t_int + dt_jump) > dt_ads, break; end
            t_int = t_int + dt_jump; jump_count = jump_count + 1;
            
            idx = find(cumsum(R_node_total(:)) >= rand() * R_total_system, 1);
            [r, c] = ind2sub([L, L], idx);
            m = find(cumsum(R_mat(r, c, :)) >= rand() * R_node_total(r, c), 1);
            
            nr = mod(r + difmatrx(m,1) - 1, L) + 1; nc = mod(c + difmatrx(m,2) - 1, L) + 1;
            old_h = H(r, c); new_h = H(nr, nc) + 1;
            TypeMap(nr, nc, new_h) = TypeMap(r, c, old_h); TypeMap(r, c, old_h) = 0;
            H(r, c) = old_h - 1; H(nr, nc) = new_h;
            
            [R_mat, R_node_total, R_total_system] = update_patch_local(r, c, L, H, TypeMap, R_mat, R_node_total, R_total_system, nu0, kB, T, Ediff_Ga, Ediff_Al, Ediff_As, E_lat_G3, E_lat_G5, E_ES);
            [R_mat, R_node_total, R_total_system] = update_patch_local(nr, nc, L, H, TypeMap, R_mat, R_node_total, R_total_system, nu0, kB, T, Ediff_Ga, Ediff_Al, Ediff_As, E_lat_G3, E_lat_G5, E_ES);


end

        % Избыток As
        for k = 1:As_excess
            ax = randi(L); ay = randi(L);
            if mod(H(ax, ay), 2) == 1 
                new_h = H(ax, ay) + 1; H(ax, ay) = new_h; TypeMap(ax, ay, new_h) = 0; 
                [R_mat, R_node_total, R_total_system] = update_patch_local(ax, ay, L, H, TypeMap, R_mat, R_node_total, R_total_system, nu0, kB, T, Ediff_Ga, Ediff_Al, Ediff_As, E_lat_G3, E_lat_G5, E_ES);
            end
        end

        % Сбор данных
        if mod(step, rheed_every) == 0
            RHEED_ML(end+1) = t_global; 
            S = sum(exp(1i * pi * H(:) / 2)); 
            RHEED_intensity(end+1) = abs(S / L^2)^2;
            Roughness_history(end+1) = std(H(:), 1);
        end
    end
end

function [R_mat, R_node_total, R_total_system] = update_patch_local(x, y, L, H, TypeMap, R_mat, R_node_total, R_total_system, nu0, kB, T, Ediff_Ga, Ediff_Al, Ediff_As, E_lat_G3, E_lat_G5, E_ES)
    dx = [0, 1, -1, 0, 0, 2, -2, 0, 0]; dy = [0, 0, 0, 1, -1, 0, 0, 2, -2];
    for k = 1:length(dx)
        r = mod(x + dx(k) - 1, L) + 1; c = mod(y + dy(k) - 1, L) + 1;
        R_total_system = R_total_system - R_node_total(r, c);
        
        R_mat(r, c, :) = 0; h_rc = H(r,c);
        if h_rc > 0
            cur_type = mod(h_rc, 2); my_id = TypeMap(r, c, h_rc);
            if cur_type == 1, ax_str = 'x'; 
                if my_id == 2, E_base = Ediff_Al; else, E_base = Ediff_Ga; end
            else, ax_str = 'y'; E_base = Ediff_As; end
            
            nb_x = (H(r, mod(c, L)+1) >= h_rc) + (H(r, mod(c-2, L)+1) >= h_rc);
            nb_y = (H(mod(r, L)+1, c) >= h_rc) + (H(mod(r-2, L)+1, c) >= h_rc);
            E_lat = (cur_type == 1)*E_lat_G3 + (cur_type == 0)*E_lat_G5;
            
            if ax_str == 'x', E_bind = nb_x*E_lat*1.2 + nb_y*E_lat; else, E_bind = nb_x*E_lat + nb_y*E_lat*1.2; end
            
            dif = [1 0; -1 0; 0 1; 0 -1];
            for m = 1:4
                nr = mod(r+dif(m,1)-1, L)+1; nc = mod(c+dif(m,2)-1, L)+1;
                if H(nr, nc) < h_rc && cur_type ~= mod(H(nr, nc), 2)
                    cur_E = E_base + E_bind;
                    if H(nr, nc) < h_rc - 1, cur_E = cur_E + E_ES; end
                    R_mat(r, c, m) = nu0 * exp(-cur_E / (kB * T));
                end
            end
        end
        R_node_total(r, c) = sum(R_mat(r, c, :));
        R_total_system = R_total_system + R_node_total(r, c);
    end
    if R_total_system < 1e-18, R_total_system = 0; end
end




