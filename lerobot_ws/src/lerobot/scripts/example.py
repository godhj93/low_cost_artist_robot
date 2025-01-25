import argparse
import os
import tempfile
import subprocess
import wave
import pyaudio
import torch
from gtts import gTTS
from diffusers import AutoPipelineForText2Image
import whisper

import func.sdxl_vectorize_to_stroke as stroke_func
from func.json_to_stroke import refine_stroke_and_to_csv

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
    print("녹음 중...")
    frames = []

    for _ in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    speak("Recording complete.")
    print("녹음 종료")
    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))


def transcribe_audio(filename):
    model = whisper.load_model("base")
    result = model.transcribe(filename, language="en")
    return result['text']


def confirm_with_voice():
    confirmation_file = "confirmation.wav"
    while True:
        record_audio(confirmation_file, duration=3)
        response = transcribe_audio(confirmation_file).strip().lower()
        print(f"음성에서 추출된 텍스트: {response}")
        if "yes" in response:
            speak("You said Yes.")
            return True
        elif "no" in response:
            speak("You said No.")
            return False
        else:
            speak("I did not understand. Please try again.")
            print("인식되지 않았습니다. 다시 시도해주세요.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", default="image")
    parser.add_argument("--savefig", action="store_true")
    args = parser.parse_args()

    filename = args.filename
    
    audio_file = "input.wav"
    while True:
        speak("Please record the name of the object you want to draw.")
        print("그리고 싶은 객체의 이름을 말해주세요.")

        record_audio(audio_file, duration=5)
        prompt = transcribe_audio(audio_file).strip()

        print(f"음성에서 추출된 텍스트: {prompt}")
        speak(f"You said: {prompt}. Is this correct? Please say Yes or No.")

        if confirm_with_voice():
            speak("Thank you. Proceeding to the next step.")
            break
        else:
            speak("Let's try again.")
            print("녹음을 다시 시작합니다.")
            
    '''
    pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")
    '''

    pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",  #
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to("cuda")
    threshold_image, threshold_image_name = stroke_func.prompt_to_line_art_img(prompt, filename, pipeline_text2image)

    stroke_list = stroke_func.img_to_svg_to_stroke(filename, threshold_image_name)


    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    csv_filename = f"refined_{filename}"

    refine_stroke_and_to_csv(os.path.join(output_dir, csv_filename), stroke_list=stroke_list)

    if not args.savefig:
        os.remove(f"original_{filename}.png")
        os.remove(f"output_{filename}.svg")
        os.remove(f"recon_from_output_{filename}.svg.png")
        os.remove(f"thresholded_{filename}.png")
