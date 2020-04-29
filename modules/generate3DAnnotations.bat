set /P dirPath=Enter directory path: 
set /P s1=Enter Sync frame for camera 1 (top): 
set /P s2=Enter Sync frame for camera 2 (front): 

echo %dirPath% %s1% %s2%

cd annotation_tool

python combine2DAnnotations.py -d %dirPath% -c top
python combine2DAnnotations.py -d %dirPath% -c front

python create3DAnnotations.py -d %dirPath% -s1 %s1% -s2 %s2%

cd ..
