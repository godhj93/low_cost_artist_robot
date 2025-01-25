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
from queue import Queue

# Add the path to the transform module
sys.path.append(os.path.join(os.path.dirname(__file__)))
from utils.transform import transform_to_configuration_space
import utils.sdxl_vectorize_to_stroke as stroke_func
from utils.json_to_stroke import refine_stroke_and_to_csv
from termcolor import colored

DRAWING_POINT_DIST = 0.002 # 1cm
MIC = False

def numpy_to_point_array(pts):
    """
    Converts a numpy array of shape (data_length, 3) into a list of Point messages.
    """
    return [Point(x=pt[0], y=pt[1], z=pt[2]) for pt in pts]

def filter_out(data):
    
    data = pd.read_csv(file_path)   
            
    # t_pts = filter_out(data)
    # Interpolartion

    x = data['x'].to_numpy()
    y = data['y'].to_numpy()
    
    dx = np.diff(x)
    dy = np.diff(y)
    
    distances = np.sqrt(dx**2 + dy**2)
    indices = np.where(distances >= distances.mean() + distances.std())[0]
    data.iloc[indices, 2] = 1
    
    pts = data.iloc[:, :3].values
    
    t_pts = transform_to_configuration_space(pts)
    
    target_queue = Queue()
    
    for pt in t_pts:
        
        new_point = pt
        
        if not target_queue.empty():
            last_point = target_queue.queue[-1]
        
        else:
            try:
                last_point = np.array([0.0, 0.3, 0.15]) # Initial pose of the robot
            except Exception as e:
                print(e)
                raise Exception(f"Initial pose of the robot is not defined. {e}")
            
        distance = np.linalg.norm(new_point - last_point)
        
        if distance > DRAWING_POINT_DIST:
            
            num_interpolated_points = int(np.ceil(distance / DRAWING_POINT_DIST))
            interpolated_points = np.linspace(last_point, new_point, num_interpolated_points, endpoint=False)
            
            for points in interpolated_points[1:]:
                target_queue.put(points)
        
        target_queue.put(new_point)
    
    t_pts = np.array(list(target_queue.queue))
    
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
        record_audio(confirmation_file, duration=0.1)
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
    
    rospy.init_node("drawing_client", anonymous=True)
    
    # Wait for the service to be available
    rospy.loginfo("Waiting for the drawing service to be available.")
    rospy.wait_for_service("drawing_request")
    rospy.loginfo("Waiting for the completed service to be available.")
    rospy.wait_for_service("drawing_completed")
    
    if MIC:
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
    
    rospy.loginfo(f"MIC is set to {MIC}")
    while not rospy.is_shutdown():
        
        if MIC:
            speak("Please record the name of the object you want to draw.")
            rospy.loginfo("Please speak what you want to draw.")
        
            record_audio(audio_file, duration=5)
            prompt = transcribe_audio(listener, audio_file).strip()
            rospy.loginfo(f"Extracted text from voice: {prompt}")

            speak(f"You said: {prompt}. Is this correct? Please say Yes or No.")
            
            if confirm_with_voice(listener):
                speak("Thank you. Proceeding to the next step.")
                break
            else:
                speak("Let's try again.")
                rospy.loginfo("Recording again.")
                
        else:
            prompt = "a cute cat" 
            rospy.loginfo(f"Prompt: {prompt}")
        
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
                
                transformed_pts = filter_out(data)

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
