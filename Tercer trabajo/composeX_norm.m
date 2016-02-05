function X = composeX_norm( X, poly_degree)
m = size(X, 1);
n = size(X, 2);
for i_var = 1:n
    for degree = 2:poly_degree
        TempX = X(:,i_var) .^ degree;
        TempX = (TempX - min(TempX) ) / ( max(TempX) - min(TempX) ); % para normalizar.
        X = [X, TempX];
    end
end
X = [ones(m, 1), X];