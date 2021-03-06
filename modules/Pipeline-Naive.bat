set /P dirPath=Enter directory path: 

echo %dirPath%

cd detection 
python BgDetector.py -f %dirPath% -c 1
python BgDetector.py -f %dirPath% -c 2

cd ..
cd tracking

python TrackerVisual.py -f %dirPath% -c 1 --preDetector
python TrackerVisual.py -f %dirPath% -c 2 --preDetector

cd ..
cd reconstruction

python TrackletMatching.py -f %dirPath%
python FinalizeTracks.py -f %dirPath%

cd ..

pause