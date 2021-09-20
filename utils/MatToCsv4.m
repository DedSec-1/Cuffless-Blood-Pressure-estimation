tic
clc;
clear all;
close all;
load('Part_1');

for d=1:3000
    d;
    Y = (Part_1{1,d});
    location = sprintf("./data_csv/Part_1/pat_%d.csv", d);
    csvwrite(location, Y);
end
toc
clear all;
