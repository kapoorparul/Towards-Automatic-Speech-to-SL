import os
import subprocess as sp
import glob
import webvtt

''' To break longer videos into short segments using vtt file downloaded along with videos'''


dst_dir = "Data" ## destination folder that will have audio, text and later Openpose poses
src_dir = "Ish"   ## source folder has .mp4 and .vtt file
vid_dir = "Video"  ##where you want to save cut videos 


for file in glob.glob(src_dir+"/*.mp4"):

	in_vid_file = file 
	time_chunks = []

	video = os.path.basename(in_vid_file)[:-4].replace(' ','_') ## remove spaces from video name

	if not os.path.exists(os.path.join(dst_dir, video)):
		os.mkdir(os.path.join(dst_dir, video))
		os.mkdir(os.path.join(dst_dir, video, "audio"))
		os.mkdir(os.path.join(dst_dir, video, "text"))
	else:
		print("already exists ", video)
		continue
		
	if not os.path.exists(os.path.join(vid_dir, video)):
		os.mkdir(os.path.join(vid_dir, video))

	prev_start = '00:00:01.000'
	subs = in_vid_file[:-4]+'.en.vtt'

	'''
	If subtitles are from t1-t2 and t3-t4 then t5-t6
	then vid1  = t1,t3 (initially t1 = 00:00:01.000)
	next vid2   = t3,t5 and so on
	.
	.
	'''

	for i, caption in enumerate(webvtt.read(subs)):
	    if i==0:
	    	prev_caption = caption.text
	    	continue
	    end_time = caption.start
	    time_chunks.append((prev_start, end_time, prev_caption))
	    
	    prev_start = caption.start
	    prev_caption = caption.text


	for i, t in enumerate(time_chunks):
	    beg_t = t[0]
	    end_t = t[1]
	    text = t[2]

	    out_vid_file = os.path.join(vid_dir, video, str(i).zfill(4)+'.mp4')
	    out_aud_file = os.path.join(dst_dir, video, "audio", str(i).zfill(4)+'.wav')
	    out_text_file = os.path.join(dst_dir, video, "text", str(i).zfill(4)+'.txt')
	    
	    
	    cmd = ["ffmpeg",  "-i", in_vid_file, "-ss", beg_t, "-to", end_t, "-y", "-avoid_negative_ts", "1", "-acodec" ,"copy", out_vid_file]
	    sp.run(cmd, stderr=sp.STDOUT) ## to get video for this segment

	    cmd = ["ffmpeg", "-i",  out_vid_file, "-ab", "160k", "-ac", "2", "-ar", "16000", "-vn", out_aud_file]
	    sp.run(cmd, stderr=sp.STDOUT)  ## to get audio for this segment

	    with open(out_text_file, "w") as text_file:
    		text_file.write(text)  ## save text
