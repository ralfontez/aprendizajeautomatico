%function ReadAndRunLinearRegresion
clear all;
close all;
clc;
data = load('univariate_reg_data.txt');
X = data(:, 1); y = data(:, 2);

X = (X - min(X)) / ( max(X) - min(X) ); % para normalizar.
size(X);
size(y);

Gradient_descent_mul_var(X, y, 1, 0.05, 1000, 1);

 
