import time
import pyttsx3


def get_tts():
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.setProperty("voice", engine.getProperty("voices")[1].id)
    return engine


def tts_say(tts: pyttsx3.engine.Engine, text: str):
    tts.say(text)
    tts.runAndWait()


def direct_to_spot():
    pass


if __name__ == "__main__":
    tts = get_tts()
    tts_say(tts, "Hi and welcome to the guided head scanning.")
    tts_say(
        tts,
        "Place your phone in an upright position at shoulder height with the camera pointing towards you.",
    )
    tts_say(tts, "Then take two steps back and follow my instructions.")
    time.sleep(5)

