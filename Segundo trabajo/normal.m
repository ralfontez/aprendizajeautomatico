function theta = normal(x,y, poly_degree)
n=size(x,2);
X = composeX_norm(x, poly_degree);
theta = pinv(X'*X)*X'*y;
disp(theta)

if n==1
    plot(x(:,1),y,'*');
    TestX = [0:0.0001:1]';
    TestX = composeX_norm(TestX, poly_degree);
    h_y=TestX*theta;

    hold on;
    plot(TestX(:,2), h_y, 'r');
    hold off;
 
end

