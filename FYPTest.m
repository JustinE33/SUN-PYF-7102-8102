%Test Script to implement PCA, FLD, RFLD

%PCA:
%Goal - fine-tune the actual inputs being used
%Values = double(rgb2gray(imread('matthew-pumpkin-grayscale.jpg'))); %convert to gray before PCA
%% load yale b
load('croppedYaleB.mat');

%% extract person
numLight = 64;
numPeople = 10;

personImage = zeros(size(mpeg_data,1),numPeople);
for person = 1:numPeople
    data = mpeg_data(:,class_label == person);
    [PCA_V,projection] = princomp(data', 'econ');
    reconstruction = PCA_V*projection';
    diff = data - reconstruction;
    personImage(:,person) = mean(diff,2);
end

[PCA_V,projTraining,latent]= pca(data);

[SbMatrixFLD, SwMatrixFLD, VectorsFLD] = FLD(PCA_V, projTraining);
%implement FLD and RFLD functions here; defined below



%FLD
% Inputs:
%   faces [dim * n] - n columns of face vectors in reduced dimension
%   class [1 * n] - label of each image
%   k - number of feature vectors being extracted (k<n)
% Outputs:
%   Sb [dim * dim] - between class scatter
%   Sw [dim * dim] - within class scatter
%   V [dim * k] - feature vectors, i.e. eigenvectors of Sw^(-1)*Sb

function [Sb, Sw, V] = FLD (faces, class)
dim = size(faces, 1);
uniqueClass = unique(class); %values of the unique classes
numClass = numel(uniqueClass); %number of classes

% initialize classMeans, which stores means for all the classes
classMeans = zeros (dim, numClass);

% initialize classNumOfElements, which stores number of elements
classNumOfElements = zeros (1, numClass);

% compute Sw
    % initialize classwCentralized, each column of which stores x - mi
    % where x is the image vector and mi is its class mean vector
    classwCentralized = zeros (dim, 0);
    % find images for each class and compute class mean and
    % classwCentralized
    for i=1:numClass
        index = class == uniqueClass(i);
        classData = faces(:,index);
        classMeans(:,i) = mean(classData,2);
        classNumOfElements(1,i) = size(classData,2);
        classwCentralized = cat(2, classwCentralized, bsxfun(@minus,classData,classMeans(:,i)));
    end
    % compute Sw
    Sw = classwCentralized * classwCentralized';

% compute Sb
meanFace = mean(faces, 2);
classbCentralized = bsxfun(@minus, classMeans, meanFace);
classSums = bsxfun(@times, classbCentralized, classNumOfElements);
Sb = classSums*classbCentralized';

% compute first numClass-1 eigenvectors of Sw^(-1)*Sb
[V,~] = eigs (Sb,Sw,numClass-1);
V = normc(V);
end

%RFLD
% Inputs:
%   image [dim * n] - n columns of face vectors in reduced dimension
%   class [1 * n] - label of each image
%   numFeatures - number of feature vectors being extracted, k
% Outputs:
%   V [dim * k] - k columns of feature vectors

function [Sb, Sw, V] = RFLD (faces, class, beta)
dim = size(faces, 1); %length of only first dimension
uniqueClass = unique(class);
numClass = numel(uniqueClass);
% initialize classMeans, which stores means for all the classes
classMeans = zeros (dim, numClass);
% initialize classNumOfElements, which stores number of elements
classNumOfElements = zeros (1, numClass);

% compute Sw
    % initialize classwCentralized, each colume of which stores x - mi
    % where x is the image vector and mi is its class mean vector
    classwCentralized = zeros (dim, 0);
    % find images for each class and compute class mean and
    % classwCentralized
    for i=1:numClass
        index = class == uniqueClass(i);
        classData = faces(:,index);
        classMeans(:,i) = mean(classData,2);
        classNumOfElements(1,i) = size(classData,2);
        classwCentralized = cat(2, classwCentralized, bsxfun(@minus,classData,classMeans(:,i)));
    end
    % compute Sw
    Sw = classwCentralized * classwCentralized';
    Sw = Sw + beta*mean(eig(Sw))*eye(dim);

% compute Sb
meanFace = mean(faces, 2);
classbCentralized = bsxfun(@minus, classMeans, meanFace);
classSums = bsxfun(@times, classbCentralized, classNumOfElements);
Sb = classSums*classbCentralized';

% compute first numClass-1 eigenvectors of Sw^(-1)*Sb
[V,~] = eigs (Sb,Sw,numClass-1);
V = normc(V);
end