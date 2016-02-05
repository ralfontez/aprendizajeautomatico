%function ReadAndRunLinearRegresion
clear all;
close all;
clc;
data = load('univariate_reg_data.txt');
X = data(:, 1); y = data(:, 2);

X = (X - min(X)) / ( max(X) - min(X) ); % para normalizar.
size(X);
size(y);


gradient_descent_one_var(X, y, 1, 0.5, 1000);
%normal(X,y);
%Gradient_descent_mul_var(X, y, 1, 6000, 60);
%FuncionNormal(X, y);
