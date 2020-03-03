% Written by Jason Rennie <jrennie@ai.mit.edu> for 6.891, Oct. 2000

% Plot some points on a graph
a = [1;2;3;4;5];
plot(a,'bx');

% Plot some more points on that same graph
b = [6;5;4;3;2];
hold on;
plot(b, 'go');

% See help to figure out what 'bx' and 'go' mean
help plot
helpwin plot

% Clear the graph with every new plot command
hold off;

% Plot a square---note that here we are controling both x and y axes
c = [2;2;2;3;4;4;4;3;2];
d = [2;3;4;4;4;3;2;2;2];
plot(c,d,'k-');

% Plot two squares on the same graph (without using the 'hold on' command)
plot(c,d,'b-',c+1,d+1,'r--');

% Set range of x and y axes
axis([0 6 1 7]);

% Put multiple plots in the same window
subplot(2,1,1), plot(a,'bx');
subplot(2,1,2), plot(c,d,'k-');
axis([0 6 1 7]);

% Return to single plot in window
subplot(1,1,1);
plot(c,d,'b-',c+1,d+1,'r--');
axis([0 6 1 7]);

% Set labels and title.  Note that you can use '_' for subscript.
xlabel 'Position (x_1)';
ylabel 'Height (y_2)'
