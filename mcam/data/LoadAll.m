clear;
close all;

data = load('data.txt');
time.year = data(1);
time.month = data(2);
time.date = data(3);
time.hour = data(4);
time.min = data(5);
time.sec = data(6);
img.width = data(7);
img.height = data(8);
genAlgo.left = data(9);
genAlgo.top = data(10);
genAlgo.right = data(11);
genAlgo.bottom = data(12);
genAlgo.numParents = data(13);
genAlgo.bind = data(14);
genAlgo.iterations = data(15);
numBetweenSave = data(16);

fileList = dir('IM*.bin');
for i=1:size(fileList,1)
    fileName = fileList(i).name;
    fid = fopen(fileName, 'r');
    img.data = fread(fid, [img.width img.height], '*int16');
    fclose(fid);
    img.cost = str2num(fileName(10:end-4));
    imshow(img.data);
    pause();
end
