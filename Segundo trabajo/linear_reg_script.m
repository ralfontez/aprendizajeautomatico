%function ReadAndRunLinearRegresion
clear all;
close all;
clc;
data = load('univariate_reg_data.txt');
X = data(:, 1); y = data(:, 2);

X = (X - min(X)) / ( max(X) - min(X) ); % para normalizar.
size(X);
size(y);


% grado del polinomio = 1
% normal(X,y,1);
% Gradient_descent_mul_var(X, y, 1, 1000, 1);

% grado del polinomio = 2
% normal(X,y,2);
% Gradient_descent_mul_var(X, y, 1, 1000, 2);

% grado del polinomio = 3
% normal(X,y,3);
% Gradient_descent_mul_var(X, y, 1, 1000, 3);

% con raiz
temp = sqrt(X);
temp = (temp -min(temp))/(max(temp) - min(temp));
X = [X, temp];
%normal(X,y,1);
Gradient_descent_mul_var(X, y, 1, 1000, 1);

%FuncionNormal(X, y);
