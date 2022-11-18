s = tf('s');
spring_sys = tf([1], [0.01 0 15 0 0]);
feedback_loop = tf([-0.01 0 -14.25 2 2], 1);
closed_loop = feedback(spring_sys, feedback_loop,  -1)
t = 0:0.005:7.5;
step(closed_loop, t);
S = stepinfo(closed_loop)

