% video sequence processing
%% disparity map

clear all;
close all;
%%
I = load('0000000083.txt');
I = I/16;
I(I == -1) = 0;
I = uint8(I);
I = I(:,129:end);

[m,n] = size(I);
figure;
% imshow(I);
imagesc(I);
colormap('default');axis image;

%% v-disparity
V = zeros(m,127);
for i = 1:m
    for j = 1:n
        d = I(i,j);
        if d ~= 0
            V(i,d) = V(i,d) + 1;
        end
    end
end
Vmax = max(max(V));
Vmean = uint8(mean(V(V~=0))) + 5;
ZV = uint8(V);
% ZV(V == 0) = 128;
ZV(V > Vmean) = 255;
ZV(V <= Vmean) = 0;
figure;
imshow(ZV);


%% Ground computation
G = zeros(1,127);
for i = 1:127
    [Gx,Gy] = size(find(ZV(:,i)>0));
    if Gx > 0
        if find(ZV(:,i)>0, 1, 'last' ) > m/2;
            G(i) = find(ZV(:,i)>0, 1, 'last' );
        else
            G(i) = 0;
        end
    else
        G(i) = 0;
    end
end
Ground = zeros(m,127);
Groundpoint = [0 0];
Ground(:,:) = 255;
for i = 1:127
    if G(i) > 0;
        Ground(G(i),i) = 0;
        Groundpoint = [Groundpoint;i,G(i)];
    end
end
figure;
hold on
imshow(Ground);
Groundpoint = Groundpoint(2:end,:);
Groundline = robustfit(Groundpoint(:,1),Groundpoint(:,2));
for i = 1:127
    linep(i) = Groundline(1) + Groundline(2)*i;
end
Groundlinep = zeros(m,127);
Groundlinep(:,:) = 255;
% xcor = [];
% ycor = [];
for i = 1:127
    if linep(i) <= m
        Groundlinep(round(linep(i)),i) = 0;
        if i > 1
        line([(i-1) i], ...
         [round(linep(i-1)) round(linep(i))], 'Color', 'c');
        end
    end
end
hold off
figure;
imshow(Groundlinep);

figure;
hold on
imshow(ZV);


%% Seperate Ground and Obstacle
I_ground = zeros(m,n);
I_obstacle = zeros(m,n);
for i = 1:m
    for j = 1:n
        d = I(i,j);
        if d ~= 0;
            if (i - linep(d)) < -15
                I_obstacle(i,j) = I(i,j);
            else
                I_ground(i,j) = I(i,j);
            end
        end
    end
end
figure;
imagesc(uint8(I_ground));
colormap('default');axis image;

figure;
imagesc(uint8(I_obstacle));
colormap('default');axis image;

%% u-disparity
U = zeros(127,n);
for i = 1:m
    for j = 1:n
        d = I_obstacle(i,j);
        if d ~= 0
            U(d,j) = U(d,j) + 1;
        end
    end
end
% Umax = max(max(U));
% Umean = mean(U(U~=0));
Z = uint8(U);
% Z(U == 0) = 128;
Z(U > 5) = 255;
Z(U <= 5) = 0;
figure;
imshow(Z)


%% Occupancy Grid Computation
% for each disparity value d, the ground point is store at linep(d)
disparity = 1:127;
lineh = linep - 4.0.*disparity;
lineh(round(lineh) == 0) = 1;
Skylinep1 = Groundlinep;
for i = 1:127
    if lineh(i) <= m
        Skylinep1(round(lineh(i)),i) = 0;
    end
end
figure;
imshow(Skylinep1);
% Np(U)
NpU = zeros(1,127);
lineh = round(lineh);
linep = round(linep);
for i = 1:127
    if linep(i) > m
        NpU(i) = round(m - lineh(i));
    else
        NpU(i) = round(linep(i) - lineh(i));
    end
end
% No(U) & Nv(U)
NoU = zeros(127,n);
NvU = zeros(127,n);
for d = 1:127
    for j = 1:n
        for k = lineh(d):lineh(d) + NpU(d) - 1
            if(I_obstacle(k,j) == d)
                NoU(d,j) = NoU(d,j) + 1;
            end
            if((I_obstacle(k,j) <= d) && (I_obstacle(k,j) > 0))
                NvU(d,j) = NvU(d,j) + 1;
            end
        end 
    end
end
% P(Vu = 1) = Nv(U)/Np(U)
NpUnew = NpU(ones(n,1),:);
NpUnew = NpUnew';
PVu = NvU./NpUnew;
% ro(U) = No(U)/Nv(U)
% P(Cu = 1) = 1 - e^(-ro(U)/ro)
roU = zeros(127,n);
PCu = zeros(127,n);
ro = 0.15;
for d = 1:127
    for j = 1:n
        if NvU(d,j) ~= 0
            roU(d,j) = NoU(d,j)/NvU(d,j);
        end
        PCu(d,j) = 1 - exp(-roU(d,j)/ro);
    end
end
% P(Ou) = P(Vu = 1)P(Cu = 1)(1 - Pfp) + P(Vu = 1)(1 - P(Cu = 1))Pfn + (1 - P(Vu = 1))*0.5
Pfp = 0.01;
Pfn = 0.05;
POu = PVu.*PCu*(1 - Pfp) + PVu.*(1 - PCu)*Pfn + (1 - PVu)*0.5;
figure;
imshow(POu);



