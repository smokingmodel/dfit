	function chedata()
	close all; clear; clc;
	tspan = [0, 8];
	t_eval = 0:1:8;
	empirical_data = [63, 77, 192, 212, 113, 470, 689, 1000, 1077];
	nu = 0.95;
	Delta  = 0.1;   kappa1 = 0.01;  kappa2 = 0.001; d = 0.05;
	D1     = 0.33;  D2     = 0.04;  gamma1 = 0.003; gamma2 = 0.003;
	tau1   = 0.05;  tau2   = 0.05;  tau3   = 0.05;
	params_initial = [Delta, kappa1, kappa2, d, D1, D2, ...
	gamma1, gamma2, tau1, tau2, tau3];
	lb = [0.05, 0.001, 0.00001, 0.001, 0.01, 0.001, ...
	0.0001, 0.0001, 0.001, 0.001, 0.001];
	ub = [0.2, 0.1, 0.01, 0.1, 1, 0.1, ...
	0.01, 0.01, 0.1, 0.1, 0.1];
	initial_conditions = [1200, 1100, 65, 30, 10];
	options = optimoptions('fmincon', ...
	'Algorithm', 'sqp', ...
	'MaxIterations', 500, ...
	'MaxFunctionEvaluations', 2000, ...
	'StepTolerance', 1e-6, ...
	'FunctionTolerance', 1e-6);
	[params_optimized, ~, ~] = fmincon(@(params) ...
	objective_function_fixed_nu(params, initial_conditions, ...
	tspan, t_eval, empirical_data, nu), ...
	params_initial, [], [], [], [], lb, ub, [], options);
	[t, Y_opt] = simulate_fractional_model_fixed_nu(params_optimized, ...
	initial_conditions, tspan, t_eval, nu);
	model_predictions = Y_opt(3, :);
	SS_res = sum((empirical_data - model_predictions).^2);
	SS_tot = sum((empirical_data - mean(empirical_data)).^2);
	R2 = 1 - (SS_res / SS_tot);	
	MAE  = mean(abs(empirical_data - model_predictions));
	MSE  = mean((empirical_data - model_predictions).^2);
	RMSE = sqrt(MSE);
	fprintf('MAE=%.4f\n', MAE);
	fprintf('MSE=%.4f\n', MSE);
	fprintf('RMSE=%.4f\n', RMSE);
	fprintf('R-squared=%.4f\n', R2);
    end
	function error = objective_function_fixed_nu(params, initial_conditions, ...
	tspan, t_eval, empirical_data, nu)
	[~, Y] = simulate_fractional_model_fixed_nu(params, ...
	initial_conditions, tspan, t_eval, nu);
	model_predictions = Y(3, :);
	error = sum((empirical_data - model_predictions).^2);
    end
	function [t, Y] = simulate_fractional_model_fixed_nu(params, ...
	initial_conditions, tspan, t_eval, nu)
	Delta  = params(1); kappa1 = params(2); kappa2 = params(3); d = params(4);
	D1 = params(5); D2 = params(6); gamma1 = params(7); gamma2 = params(8);
	tau1 = params(9); tau2 = params(10); tau3 = params(11);
	h = 0.13;
	t = t_eval;
	n = length(t_eval);
	m = length(initial_conditions);
	Y = zeros(m, n);
	Y(:,1) = initial_conditions';
	for i = 2:n
	steps = ceil((t(i) - t(i-1)) / h);
	current_state = Y(:, i-1);
	current_time = t(i-1);
	for j = 1:steps
	dydt = model_equations_fixed_nu(current_time, current_state, ...
	Delta, kappa1, kappa2, d, D1, D2, gamma1, gamma2, ...
	tau1, tau2, tau3);
	current_state = current_state + ...
	(h^nu / gamma(nu + 1)) * dydt;
	current_state = max(current_state, 0);
	current_time = current_time + h;
    end
	Y(:, i) = current_state;
    end
    end
	function dydt = model_equations_fixed_nu(~, y, Delta, kappa1, kappa2, ...
	d, D1, D2, gamma1, gamma2, tau1, tau2, tau3)
	phi1 = max(y(1), 0);
	phi2 = max(y(2), 0);
	phi3 = max(y(3), 0);
	phi4 = max(y(4), 0);
	phi5 = max(y(5), 0);
	dphi1dt = Delta - kappa1*phi1*phi2 + gamma1*phi2 + gamma2*phi3 - d*phi1;
	dphi2dt = kappa1*phi1*phi2 - kappa2*phi2*phi3 - (gamma1 + tau1 + D1 + d)*phi2;
	dphi3dt = kappa2*phi2*phi3 - (gamma2 + tau2 + D2 + d)*phi3;
	dphi4dt = tau2*phi3 + tau1*phi2 - (tau3 + d)*phi4;
	dphi5dt = tau3*phi4 - d*phi5;
	dydt = [dphi1dt; dphi2dt; dphi3dt; dphi4dt; dphi5dt];
	end
