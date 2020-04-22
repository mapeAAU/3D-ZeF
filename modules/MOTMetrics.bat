set /P dirPath=Enter directory path: 

echo %dirPath%

cd evaluation

python MOT_evaluation.py -detCSV "%dirPath%/processed/tracklets_2d_cam1.csv" -gtCSV "%dirPath%/gt/annotations_full.csv" -task "cam1" -thresh 20.0 -outputPath "%dirPath%/metrics" -outputFile MOT_cam1.txt
python MOT_evaluation.py -detCSV "%dirPath%/processed/tracklets_2d_cam2.csv" -gtCSV "%dirPath%/gt/annotations_full.csv" -task "cam2" -thresh 20.0 -outputPath "%dirPath%/metrics" -outputFile MOT_cam2.txt

python MOT_evaluation.py -detCSV "%dirPath%/processed/tracklets_3d.csv" -gtCSV "%dirPath%/gt/annotations_full.csv" -task "cam1" -thresh 20.0 -outputPath "%dirPath%/metrics" -outputFile MOT_cam1_3Dtracklets.txt
python MOT_evaluation.py -detCSV "%dirPath%/processed/tracklets_3d.csv" -gtCSV "%dirPath%/gt/annotations_full.csv" -task "cam2" -thresh 20.0 -outputPath "%dirPath%/metrics" -outputFile MOT_cam2_3Dtracklets.txt

python MOT_evaluation.py -detCSV "%dirPath%/processed/tracklets_3d.csv" -gtCSV "%dirPath%/gt/annotations_full.csv" -task "3D" -thresh 0.5 -outputPath "%dirPath%/metrics" -outputFile MOT_3D_tracklets.txt
python MOT_evaluation.py -detCSV "%dirPath%/processed/tracks_3d.csv" -gtCSV "%dirPath%/gt/annotations_full.csv" -task "3D" -thresh 0.5 -outputPath "%dirPath%/metrics" -outputFile MOT_3D_tracks.txt
python MOT_evaluation.py -detCSV "%dirPath%/processed/tracks_3d_interpolated.csv" -gtCSV "%dirPath%/gt/annotations_full.csv" -task "3D" -thresh 0.5 -outputPath "%dirPath%/metrics" -outputFile MOT_3D_tracks_interpolated.txt

cd ..