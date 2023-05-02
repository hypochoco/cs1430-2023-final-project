# Use pip install pyttsx3 prior to running this
import pyttsx3 as tts

class TTS:
    '''
    A class that defines the properties for a pyttsx3
    engine and provides a function that reads aloud 
    strings using that engine.
    '''
    def __init__(self, rate: int, volume: float, voice: int):
        '''
        Initialization function for a TTS object.
        
        Input: rate is the number of words read per minute,
               volume is a floating point value between 0.0 and 1.0,
               voice is either 0 for male or 1 for female

        Output: A TTS object with the given attributes

        Exceptions: Raises a TypeError if either the volume or voice values 
                    are out of bounds.
        '''
        # Recommened value: 135
        self.rate = rate 
        if 0.0 <= volume <= 1.0:
            self.volume = volume
        else: 
            raise TypeError("Volume should be a floating point value between 0.0 and 1.0.")
        if (voice == 0) or (voice == 1):
            self.voice = voice
        else:
            raise TypeError("Voice should be either 0 for a male voice or 1 for a female voice.")

    def speak(self, text: str) -> None:
        '''
        Using the pyttsx3 library, this function reads
        out the given string.

        Input: String to read out

        Output: None
        '''
        # Set up the text-to-speach engine
        engine = tts.init()
        voices = engine.getProperty("voices")
        engine.setProperty("rate", self.rate)
        engine.setProperty("volume", self.volume)
        engine.setProperty("voice", voices[self.voice].id)
        # Queue up the given text
        engine.say(text)
        # Say the text and clear the queue
        engine.runAndWait()
        engine.stop()
