## Processing the data:

Follow are steps given below : 
1. Download the dataset from https://drive.google.com/drive/folders/1pwh9khgS-77tLXwNYXzxyKMBwrhcISwV?usp=sharing.
2. Use cut_segments.py to extract short videos from longer videos.
3. Use OpenPose(https://github.com/CMU-Perceptual-Computing-Lab/openpose) to find poses from videos.
4. Use Inverse Kinematics approach(https://github.com/gopeith/SignLanguageProcessing) to correct the OpenPose keypoints.
5. Save the keypoints, audio features, text and file names in separate folder using save_preprocessed_data.py
