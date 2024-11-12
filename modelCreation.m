
imdsKLGrade = imageDatastore("sorted\KLGrade\","IncludeSubfolders",true,"LabelSource","foldernames");
[imdsTrainKLGrade, imdsValKLGrade, imdsTestKLGrade] = splitEachLabel(imdsKLGrade, 0.7, 0.1, 'randomized');

imds = imageDatastore("sorted\withoutKLGrade\","IncludeSubfolders",true,"LabelSource","foldernames");
[imdsTrain, imdsVal, imdsTest] = splitEachLabel(imds, 0.7, 0.1, 'randomized');

imdsTrainKLGrade = shuffle(imdsTrainKLGrade);
imdsTrain = shuffle(imdsTrain);

net = imagePretrainedNetwork("inceptionresnetv2",NumClasses=5);
[layerName,learnableNames] = networkHead(net);
net = freezeNetwork(net,LayerNamesToIgnore=layerName);
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( RandXReflection=true, RandXTranslation=pixelRange, RandYTranslation=pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrainKLGrade, DataAugmentation=imageAugmenter, ColorPreprocessing="gray2rgb");
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValKLGrade, ColorPreprocessing="gray2rgb");
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTestKLGrade, ColorPreprocessing="gray2rgb");


options_0 = trainingOptions("adam", ValidationData=augimdsValidation, ValidationFrequency=5, Plots="training-progress", Metrics="accuracy", Verbose=false,Shuffle="every-epoch",ExecutionEnvironment="cpu", MaxEpochs=3);
options = trainingOptions("adam", ValidationData=augimdsValidation, ValidationFrequency=5, Plots="training-progress", Metrics="accuracy", Verbose=false,Shuffle="every-epoch",ExecutionEnvironment="cpu", MaxEpochs=3);
net_1 = trainnet(augimdsTrain,net_1,"crossentropy",options);
net_1 = trainnet(imdsTrainKLGrade,net_1,"crossentropy",options);


YTest = minibatchpredict(net_1,augimdsTest);
classNames = categories(imds.Labels);

YTest = scores2label(YTest,classNames);


options = trainingOptions("adam", ValidationData=imdsVal, ValidationFrequency=5, Plots="training-progress", Metrics="accuracy", Verbose=false,Shuffle="every-epoch",ExecutionEnvironment="cpu", MaxEpochs=3);
net_New = trainnet(augimdsTrain,net_2,"crossentropy",options);
