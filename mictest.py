import pyaudio

audio = pyaudio.PyAudio()

print(audio.get_device_count())

for i in range(audio.get_device_count()):
    device_info = audio.get_device_info_by_index(i)
    print(f"Device {i}: {device_info['name']} (Input Channels: {device_info['maxInputChannels']})")


mic_index = 0  # Change this to your microphone index

device_info = audio.get_device_info_by_index(mic_index)
print(f"\n🎤 Microphone: {device_info['name']} (Index {mic_index})")
print(f"🔍 Supported Sample Rate: {device_info['defaultSampleRate']} Hz")

audio.terminate()
