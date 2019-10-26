% StaplerDataTable = StaplergTruth;
% StaplergTruth.imageFilename = fullfile(pwd,StaplergTruth.imageFilename);
% StaplergTruth(1:4,:);
% StaplergTruth.imageFilename = fullfile(pwd,StaplergTruth.imageFilename);

I = imread(StaplergTruth.imageFilename{95});
I = insertShape(I,'Rectangle',StaplergTruth.Stapler{95});
I = imresize(I,3);
imshow(I);
rng(0);
%% Select randomly
shuffled = randperm(height(StaplergTruth));
Mugshuffled = randperm(height(MugsgTruth));
Laptopshuffled = randperm(height(LaptopgTruth));
%%Split the data
idx = floor(0.6 * length(shuffled));
Mugsidx = floor(0.6 * length(Mugshuffled));
Laptopidx = floor(0.6 * length(Laptopshuffled));
%% Assign the Training and test data
trainingData = StaplergTruth(shuffled(1:idx),:);
MugsTrainingData = MugsgTruth(Mugshuffled(1:Mugsidx),:);
LaptopTrainingData = LaptopgTruth(Laptopshuffled(1:idx),:);

testData = StaplergTruth(shuffled(idx+1:end),:);
MugstestData = MugsgTruth(Mugshuffled(idx+1:end),:);
LaptoptestData = LaptopgTruth(Laptopshuffled(idx+1:end),:);
 %% 
imageSize = [224 224 3];

numClasses = width(StaplergTruth)-1;
MugsnumClasses = width(MugsgTruth)-1;
LaptopnumClasses = width(LaptopgTruth)-1;
%% now to help us know the anchor boxes,we predict it using the kmedoids clusteruing
% datadir = fullfile(toolboxdir('vision'),'visiondata');
% StaplergTruth(1:4,:);
% StaplergTruth.imageFilename = fullfile(datadir,StaplergTruth.imageFilename);
% summary(StaplergTruth);
allboxes = vertcat(StaplergTruth.Stapler{:});
mugsallboxes = vertcat(MugsgTruth.Mugs{:});
laptopallboxes = vertcat(LaptopgTruth.Laptop{:});

aspectRatio = allboxes(:,3) ./ allboxes(:,4);
mugsaspectRatio= mugsallboxes(:,3) ./ mugsallboxes(:,4);
laptopaspectRatio = laptopallboxes(:,3) ./laptopallboxes(:,4);


area = prod(allboxes(:,3:4),2);
mugsarea = prod(mugsallboxes(:,3:4),2);
laptoparea = prod(laptopallboxes(:,3:4),2);

figure; scatter(area,aspectRatio); xlabel("Box Area"); ylabel("Aspect Ratio (width/height)");
title("Box area vs. Aspect ratio");

figure; scatter(mugsarea,mugsaspectRatio); xlabel(" MugsBox Area"); ylabel(" MugsAspect Ratio (width/height)");
title(" MugsBox area vs. MugsAspect ratio");

figure; scatter(laptoparea,laptopaspectRatio); xlabel("laptopBox Area"); ylabel("LaptopAspectRatio(width/height)");
title("LaptopBox Area vs .LaptopAspectRatio");

%% To determine the bounding boxes
numAnchors = 4;
[clusterAssignments,anchorBoxes,sumd] = kmedoids(allboxes(:,3:4),numAnchors,'Distance',...
    @iouDistanceMetric);
[mugsclusterAssignments,mugsanchorBoxes,mugssumd] = kmedoids(mugsallboxes(:,3:4),numAnchors,'Distance',...
    @iouDistanceMetric);
[laptopclusterAssignment,laptopanchorBoxes,laptopsumd] =kmedoids(laptopallboxes(:,3:4),4,'Distance',...
    @iouDistanceMetric);
%%

baseNetwork = resnet50;

featureLayer = 'activation_40_relu';

lgraph = yolov2Layers(imageSize,numClasses,anchorBoxes,baseNetwork,featureLayer);
mugslgraph = yolov2Layers(imageSize,MugsnumClasses,mugsanchorBoxes,baseNetwork,featureLayer);
laptoplgraph = yolov2Layers(imageSize,LaptopnumClasses,(laptopanchorBoxes-70),baseNetwork,featureLayer);
%%
doTraining = true;
if doTraining
    
options = trainingOptions('sgdm', ...
        'MiniBatchSize', 16, ....
        'InitialLearnRate',1e-3, ...
        'MaxEpochs',10,...
        'CheckpointPath', tempdir, ...
        'Shuffle','every-epoch');    
    
    
     [detector,info] = trainYOLOv2ObjectDetector(StaplergTruth,lgraph,options);
     [mugsdetector,muginfo] = trainYOLOv2ObjectDetector(MugsgTruth,mugslgraph,options);
     [laptopdetector,laptopinfo] = trainYOLOv2ObjectDetector(LaptopgTruth,laptoplgraph,options);
     
     else
    % Load pretrained detector for the example.
    pretrained = load('yolov2ResNet50VehicleExample.mat');
    detector = pretrained.detector;
end
%% to test the detector
% I = imread(testData.imageFilename{end});
% 
% % Run the detector.
% [bboxes,scores] = detect(detector,I);
% 
% % Annotate detections in the image.
% I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
% imshow(I)
%% The functions definitions
function dist = iouDistanceMetric(boxWidthHeight,allBoxWidthHeight)
% Return the IoU distance metric. The bboxOverlapRatio function
% is used to produce the IoU scores. The output distance is equal
% to 1 - IoU.

% Add x and y coordinates to box widths and heights so that
% bboxOverlapRatio can be used to compute IoU.
boxWidthHeight = prefixXYCoordinates(boxWidthHeight);
allBoxWidthHeight = prefixXYCoordinates(allBoxWidthHeight);

% Compute IoU distance metric.
dist = 1 - bboxOverlapRatio(allBoxWidthHeight, boxWidthHeight);
end

function boxWidthHeight = prefixXYCoordinates(boxWidthHeight)
% Add x and y coordinates to boxes.
n = size(boxWidthHeight,1);
boxWidthHeight = [ones(n,2) boxWidthHeight];
end


%%





