import streamlit as st
import requests
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import os
import tempfile
import numpy as np
import soundfile as sf

# Backend API URL
API_URL = "http://127.0.0.1:8000"  # Replace with your FastAPI backend URL

st.title("Unified Frontend for FastAPI Backend")

# Navigation
menu = st.sidebar.selectbox("Choose a feature", [
    "Home",
    "Extract Resume Data",
    "AI Job Recommendations",
    "Get Job Recommendations",
    "Chat with HR System",
    "Transcribe Audio",
])

if menu == "Home":
    st.write("Welcome to the Unified Frontend for the FastAPI Backend!")
    st.write("Use the sidebar to navigate through the features.")

elif menu == "Extract Resume Data":
    st.header("Extract Resume Data")
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    if uploaded_file is not None:
        files = {"resumefile": uploaded_file}
        response = requests.post(f"{API_URL}/extract", files=files)
        if response.status_code == 200:
            st.write("Extracted Resume Data:")
            st.json(response.json())
        else:
            st.error(f"Error: {response.json().get('detail')}")

elif menu == "Get Job Recommendations":
    st.header("Get Job Recommendations")
    
    user_skills = st.text_area("Enter your skills (comma-separated):")
    job_descs = st.text_area("Enter multiple job descriptions (JSON format):", height=200)

    if st.button("Get Recommendations"):
        try:
            job_descriptions = eval(job_descs)  # Convert input to list of dicts
            data = {
                "user_skills": user_skills,
                "job_descriptions": job_descriptions
            }
            response = requests.post(f"{API_URL}/recomendersystem", json=data)
            if response.status_code == 200:
                st.write("Job Recommendations:")
                st.json(response.json())
            else:
                st.error(f"Error: {response.json().get('detail')}")
        except Exception as e:
            st.error("Invalid JSON format for job descriptions.")

elif menu == "Chat with HR System":
    st.header("Chat with HR System")
    user_message = st.text_input("Your Message")
    chat_history = st.text_area("Chat History (JSON format)", height=200)
    skills = st.text_area("Skills (JSON format)", height=100)
    work_experience = st.text_area("Work Experience (JSON format)", height=100)
    projects = st.text_area("Projects (JSON format)", height=100)

    if st.button("Send Message"):
        try:
            chat_history = eval(chat_history) if chat_history else None
            skills = eval(skills)
            work_experience = eval(work_experience)
            projects = eval(projects)

            data = {
                "user_message": user_message,
                "chat_history": chat_history,
                "skills": skills,
                "work_experience": work_experience,
                "projects": projects,
            }
            response = requests.post(f"{API_URL}/chat", json=data, stream=True)
            if response.status_code == 200:
                st.write("Chat Response:")
                for line in response.iter_lines():
                    st.write(line.decode("utf-8"))
            else:
                st.error(f"Error: {response.json().get('detail')}")
        except Exception as e:
            st.error("Invalid JSON format in input fields.")

elif menu == "Transcribe Audio":
    st.header("Transcribe Audio")

    # Option 1: Record Audio

    st.subheader("Record Audio")
    webrtc_ctx = webrtc_streamer(
        key="audio-recorder",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"audio": True, "video": False},
        frontend_rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        server_rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    if webrtc_ctx and webrtc_ctx.state.playing:
        if webrtc_ctx.audio_receiver:
            # Wait for audio frames to be collected
            audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=30)
            if len(audio_frames) > 0:
                st.info("Processing audio...")

                try:
                    # Combine audio frames
                    audio_data = b"".join(frame.planes[0].to_bytes() for frame in audio_frames)

                    # Convert raw bytes to NumPy array
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)

                    # Save to a .wav file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_file:
                        sample_rate = audio_frames[0].sample_rate  # Get the sample rate from the first frame
                        sf.write(audio_file.name, audio_array, samplerate=sample_rate, subtype="PCM_16")
                        audio_file_path = audio_file.name

                    st.success("Audio recorded successfully!")
                    st.audio(audio_file_path, format="audio/wav")

                    # Transcribe the recorded audio
                    st.write("Transcribing recorded audio...")
                    with open(audio_file_path, "rb") as f:
                        files = {"audio": f}
                        response = requests.post(f"{API_URL}/transcribe", files=files)

                    os.remove(audio_file_path)  # Clean up temporary file after use

                    if response.status_code == 200:
                        st.write("Transcription:")
                        st.write(response.json().get("transcription"))
                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
            else:
                st.warning("No audio frames detected. Please try recording again.")
        else:
            st.warning("Audio receiver is not initialized. Please start recording.")
    else:
        st.warning("WebRTC context is not active. Please start recording.")  
           
    # Option 2: Upload Audio File
    st.subheader("Upload Audio File")
    uploaded_file = st.file_uploader("Upload Audio File", type=["mp3", "wav"])
    if uploaded_file is not None:
        files = {"audio": uploaded_file}
        response = requests.post(f"{API_URL}/transcribe", files=files)

        # Display transcription result
        if response.status_code == 200:
            st.write("Transcription:")
            st.write(response.json().get("transcription"))
        else:
            st.error(f"Error: {response.json().get('detail')}")


elif menu == "AI Job Recommendations":
    st.title("AI-Powered Job Recommendation System")

    # Input section for user skills
    st.header("Skills")
    skills = st.text_area(
        "Enter your skills (e.g., Python, Machine Learning, SQL)", 
        placeholder="Example: Python, Machine Learning, SQL"
    )

    # Input section for work experience
    st.header("Work Experience")
    work_experience = st.text_area(
        "Enter your work experience (e.g., roles, companies, duration)", 
        placeholder="Example: Software Engineer at XYZ Corp for 3 years"
    )

    # Input section for projects
    st.header("Projects")
    projects = st.text_area(
        "Enter details of your projects (e.g., title, description, technologies used)", 
        placeholder="Example: Developed a machine learning model to predict sales"
    )

    # Submit button
    if st.button("Get Recommendations"):
        if not skills or not work_experience or not projects:
            st.error("Please fill out all fields before submitting!")
        else:
            # Prepare data for API request
            payload = {
                "skills": [{"skill_name": skill.strip()} for skill in skills.split(",")],
                "work_experience": [{"experience": work_experience.strip()}],
                "projects": [{"project": projects.strip()}]
            }

            try:
                # Send POST request to the API
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()  # Raise an error for HTTP codes 4xx/5xx

                # Display recommendations
                recommendations = response.json().get("response", [])
                if recommendations:
                    st.header("Top Job Recommendations")
                    for i, rec in enumerate(recommendations, start=1):
                        st.subheader(f"{i}. {rec['job_title']}")
                        st.write(f"**Similarity Score:** {rec['similarity_score']}")
                else:
                    st.info("No recommendations found.")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to fetch recommendations: {str(e)}")