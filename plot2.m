% Define the grid for X1 and X2
[X1, X2] = meshgrid(-0.5:0.1:1.5, -0.5:0.1:1.5);

% Calculate the decision boundary plane for O1 (Z-axis)
% Equation: X1 + X2 - 2*O1 - 0.5 = 0  =>  2*O1 = X1 + X2 - 0.5  => O1 = (X1 + X2 - 0.5) / 2
O1_plane = (X1 + X2 - 0.5) / 2;

% Plot the plane
figure;
surf(X1, X2, O1_plane, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
hold on;
xlabel('X1 Input');
ylabel('X2 Input');
zlabel('O1 Hidden Unit Output');
title('Decision Boundary Plane for O2: X1 + X2 - 2*O1 - 0.5 = 0');
grid on;

% Plot the specific inpjt points in (X1, X2, O1) space
% Points are formatted as [x1, x2, o1_val]
points = [0 0 0; 0 1 0; 1 0 0; 1 1 1];
plot3(points(:,1), points(:,2), points(:,3), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
legend('Decision Plane', 'Input (00, 01, 10, 11)');