clear;
close all;

data = load('data.txt');
year = data(1);
month = data(2);
date = data(3);
hour = data(4);
min = data(5);
sec = data(6);
img.width = data(7);
img.height = data(8);
rectAlgo.left = data(9);
rectAlgo.top = data(10);
rectAlgo.right = data(11);
rectAlgo.bottom = data(12);
numParents = data(13);
bind = data(14);
iterations = data(15);
numBetweenSave = data(16);

fileList = dir('IM*.bin');
for i=1:size(fileList,1)
    fileName = fileList(i).name;
    fid = fopen(fileName, 'r');
    img.data = fread(fid, [img.width img.height], '*int16');
    fclose(fid);
    imshow(img.data);
    pause();
end
