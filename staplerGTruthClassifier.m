% StaplerDataTable = StaplergTruth;
% StaplergTruth.imageFilename = fullfile(pwd,StaplergTruth.imageFilename);
% StaplergTruth(1:4,:);
% StaplergTruth.imageFilename = fullfile(pwd,StaplergTruth.imageFilename);
I = imread(StaplergTruth.imageFilename{95});
I = insertShape(I,'Rectangle',StaplergTruth.Stapler{95});
I = imresize(I,3);
imshow(I);
rng(0);
shuffled = randperm(height(StaplergTruth));
idx = floor(0.6 * length(shuffled));
trainingData = StaplergTruth(shuffled(1:idx),:);
testData = StaplergTruth(shuffled(idx+1:end),:);

imageSize = [224 224 3];

numClasses = width(StaplergTruth)-1;
%%
% datadir = fullfile(toolboxdir('vision'),'visiondata');
% StaplergTruth(1:4,:);
% StaplergTruth.imageFilename = fullfile(datadir,StaplergTruth.imageFilename);
% summary(StaplergTruth);
allboxes = vertcat(StaplergTruth.Stapler{:});
aspectRatio = allboxes(:,3) ./ allboxes(:,4);
area = prod(allboxes(:,3:4),2);
figure; scatter(area,aspectRatio); xlabel("Box Area"); ylabel("Aspect Ratio (width/height)");
title("Box area vs. Aspect ratio");
%% To determine the bounding boxes
numAnchors = 4;
[clusterAssignments,anchorBoxes,sumd] = kmedoids(allboxes(:,3:4),numAnchors,'Distance',...
    @iouDistanceMetric);
%%

baseNetwork = resnet50;

featureLayer = 'activation_40_relu';

lgraph = yolov2Layers(imageSize,numClasses,anchorBoxes,baseNetwork,featureLayer);


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
     
     else
    % Load pretrained detector for the example.
    pretrained = load('yolov2ResNet50VehicleExample.mat');
    detector = pretrained.detector;
end
%% to test the detector
I = imread(testData.imageFilename{end});

% Run the detector.
[bboxes,scores] = detect(detector,I);

% Annotate detections in the image.
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
imshow(I)
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





