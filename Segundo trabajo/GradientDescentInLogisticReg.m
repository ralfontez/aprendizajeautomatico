function [Theta, Jcost] = GradientDescentInLogisticReg(X, y, alpha, num_iters)
%n = size(X, 2); % Numero de variables incluyendo x0 y variables al cuadrado, cubo, etc.
m = length(y); % Numero de ejemplos de entrenamiento
Theta = zeros(size(X, 2), 1); % Almacena los valores finales de los parametros Theta
J_history = zeros(1, num_iters); % Almacena el historico de la funcion costo J
for iter = 1:num_iters
    %if mod( iter - 1, 10 ) == 0
        % Plot Boundary
        %close all;
        %plotDecisionBoundary(Theta, X, y);
        %Theta
        %pause();
    %end
    f_sigmoidal = 1 ./ ( 1 + exp( - X * Theta ) );
    %Theta
    %pause();
    J_history(iter) = ( 1 / ( 2 * m ) ) * ( f_sigmoidal - y )' * ( f_sigmoidal - y );
    Theta = Theta - alpha * (1 / m ) * (( f_sigmoidal - y )' * X)';
    
    %disp(Theta);
end
%disp(Theta);
%plotDecisionBoundary(Theta, X, y);
%figure;
plot ([1:num_iters], J_history);
Jcost = J_history(end);