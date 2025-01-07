#!/usr/bin/python3

import rospy
import pandas as pd
import numpy as np
from lerobot.srv import DrawingRequest, DrawingCompleted, DrawingCompletedResponse
from geometry_msgs.msg import Point
from glob import glob
import os, sys, torch, tempfile, whisper, subprocess, pyaudio, wave
from diffusers import AutoPipelineForText2Image
from gtts import gTTS

# Add the path to the transform module
sys.path.append(os.path.join(os.path.dirname(__file__)))
from utils.transform import transform_to_configuration_space
import utils.sdxl_vectorize_to_stroke as stroke_func
from utils.json_to_stroke import refine_stroke_and_to_csv

def numpy_to_point_array(pts):
    """
    Converts a numpy array of shape (data_length, 3) into a list of Point messages.
    """
    return [Point(x=pt[0], y=pt[1], z=pt[2]) for pt in pts]

def filter_out(data):
    
    # x와 y 좌표
    x = data['x'].to_numpy()
    y = data['y'].to_numpy()
    # 연속된 행의 x, y 좌표 차이 계산
    dx = np.diff(x)
    dy = np.diff(y)
    # 유클리디안 거리 계산
    distances = np.sqrt(dx**2 + dy**2)
    # 거리 조건을 만족하는 인덱스 찾기
    indices = np.where(distances >= distances.mean() + distances.std())[0]
    # Set z as 1 for indices
    data.iloc[indices, 2] = 1
    
    pts = data.iloc[:, :3].values
    
    # Transform the points
    t_pts = transform_to_configuration_space(pts)
                
    # 결과를 저장할 리스트
    new_pts = [t_pts[0]]  # 첫 번째 점은 항상 유지
    for i in range(1, len(t_pts)):
        prev_point = t_pts[i - 1]
        curr_point = t_pts[i]
        x_prev, y_prev, z_prev = prev_point
        x_curr, y_curr, z_curr = curr_point
        # A -> A'에서 Z값 변화(0.51 -> 1.0)
        # Z 보간만 수행 (X, Y는 고정)
        skip_append = False
        if z_prev < z_curr:
            num_steps = 100  # 원하는 보간 스텝 수
            z_lin = np.linspace(z_prev, z_curr, num_steps, endpoint=False)[1:]
            x_lin = np.full_like(z_lin, x_prev)
            y_lin = np.full_like(z_lin, y_prev)
            z_points = np.column_stack((x_lin, y_lin, z_lin))
            new_pts.extend(z_points)
        # A' → B' (X, Y 보간, Z=1.0로 고정)
        # 만약 curr_point가 A'(0,0,1.0)이고 다음 점 B(1,1,0.51)로 가기 전에
        # 'B'(1,1,1.0)이라는 가상의 중간 점을 만든 뒤 X,Y 보간
        
        if np.isclose(z_curr, 0.1) and i + 1 < len(t_pts):
            next_point = t_pts[i + 1]  # 실제로는 B(1,1,0.51)
            x_next, y_next, z_next = next_point
            if z_curr == z_next:
                
                # B' 정의: (x_next, y_next, 1.0)
                # A'(x_curr, y_curr, z_curr) -> B'(x_next, y_next, 1.0)
                num_steps = 100
                x_lin = np.linspace(x_curr, x_next, num_steps, endpoint=False)[1:]
                y_lin = np.linspace(y_curr, y_next, num_steps, endpoint=False)[1:]
                z_lin = np.full_like(z_lin, 0.1)  # Z는 0.1으로 고정
                xy_points = np.column_stack((x_lin, y_lin, z_lin))
                new_pts.extend(xy_points)
                print("A' -> B' X, Y 보간 추가 완료")
        
            if z_curr > z_next:
                # B' -> B, Z 감소 보간 (X, Y는 B의 값으로 고정)
                num_steps = 100
                z_lin = np.linspace(0.1, 0.051, num_steps, endpoint=False)[1:]
                x_lin = np.full_like(z_lin, x_next)
                y_lin = np.full_like(z_lin, y_next)
                z_points = np.column_stack((x_lin, y_lin, z_lin))
                new_pts.extend(z_points)
                print("B' -> B Z 감소 보간 추가 완료")
        if not skip_append:
            # 현재 점 추가
            new_pts.append(curr_point)
    t_pts = np.array(new_pts)
    
    return t_pts

########## Whisper and SDXL ##########
script_path = os.path.abspath(__file__)

def speak(message):
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as temp_audio:
        tts = gTTS(text=message, lang='en')
        tts.save(temp_audio.name)
        subprocess.run(
            ["mpg123", temp_audio.name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

def record_audio(filename, duration=5, rate=16000, chunk=1024):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)
    frames = []

    for _ in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    speak("Recording complete.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

def transcribe_audio(model, filename):
    # model = whisper.load_model("base")
    result = model.transcribe(filename, language="en")
    return result['text']

def confirm_with_voice(model: whisper.Whisper):
    confirmation_file = "confirmation.wav"
    
    while not rospy.is_shutdown():
        record_audio(confirmation_file, duration=3)
        response = transcribe_audio(model, confirmation_file).strip().lower()
        rospy.loginfo(f"Extracted text from voice: {response}")
        
        response = 'yes'
        if "yes" in response:
            speak("You said Yes.")
            return True
        elif "no" in response:
            speak("You said No.")
            return False
        else:
            speak("I did not understand. Please try again.")
            rospy.loginfo("I did not understand. Please try again.")
            
            
if __name__ == "__main__":
    
    assert torch.cuda.is_available(), "CUDA is not available. Please install CUDA."
    
    rospy.init_node("drawing_client")
    
    # Wait for the service to be available
    rospy.loginfo("Waiting for the drawing service to be available.")
    rospy.wait_for_service("drawing_request")
    rospy.loginfo("Waiting for the completed service to be available.")
    rospy.wait_for_service("drawing_completed")
    
    rospy.loginfo("Loading Whisper model...")
    listener = whisper.load_model("base")
    
    rospy.loginfo("Loading Stable Diffusion model...")
    pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",  #
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to("cuda")
    
    rospy.loginfo("Ready to listen.")
    
    audio_file = "input.wav"
    filename = "image"
    
    while not rospy.is_shutdown():
        
        speak("Please record the name of the object you want to draw.")
        rospy.loginfo("Please speak what you want to draw.")
        
        record_audio(audio_file, duration=5)
        prompt = transcribe_audio(listener, audio_file).strip()
        
        prompt = "a cute cat" 
        rospy.loginfo(f"Extracted text from voice: {prompt}")
        speak(f"You said: {prompt}. Is this correct? Please say Yes or No.")
        
        if confirm_with_voice(listener):
            speak("Thank you. Proceeding to the next step.")
            break
        else:
            speak("Let's try again.")
            rospy.loginfo("Recording again.")
    
    threshold_image, threshold_image_name = stroke_func.prompt_to_line_art_img(prompt, filename, pipeline_text2image)
    stroke_list = stroke_func.img_to_svg_to_stroke(filename, threshold_image_name)
    
    output_dir = "drawing_data"
    output_dir = os.path.join(os.path.dirname(script_path), output_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    csv_filename = f"refined_{filename}"
    path_to_save = os.path.join(output_dir, csv_filename)
    refine_stroke_and_to_csv(path_to_save, stroke_list=stroke_list)
    rospy.loginfo(f"Stroke data saved to {path_to_save}.")
    
    try:
        # Create the service proxy
        drawing_service = rospy.ServiceProxy("drawing_request", DrawingRequest)
        complete_service = rospy.ServiceProxy("drawing_completed", DrawingCompleted)
        
        # Read the CSV files
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_files = glob(os.path.join(script_dir, 'drawing_data/*.csv'))
        rospy.loginfo(f"Found {len(csv_files)} csv files.")
        
        # Process each file
        for file_path in csv_files:
            data = pd.read_csv(file_path)
            
            pts = filter_out(data)

            # Transform the points
            transformed_pts = transform_to_configuration_space(pts)

            # Convert to Point[] message
            points_msg = numpy_to_point_array(transformed_pts)

            rospy.loginfo("Sending points to the drawing server.")
            
            # Call the service
            response = drawing_service(points_msg)
            
            if response.success:
                rospy.logwarn(f"Drawing Requested for file: {file_path}")
            else:
                rospy.logerr(f"Failed to requesting drawing for file: {file_path}")

            while not complete_service().success:
                # rospy.loginfo("Waiting for the drawing server to complete the task.")
                rospy.sleep(1)
                
                if rospy.is_shutdown():
                    break
                
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
