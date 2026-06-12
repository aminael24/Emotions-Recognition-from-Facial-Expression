import pyttsx3


class Speaker:
    """
    Version pyttsx3 avec voix anglaise.
    """

    def __init__(self):
        self._enabled = True

        try:
            engine = pyttsx3.init()
            self._set_english_voice(engine)
            engine.setProperty("rate", 150)
            engine.setProperty("volume", 1.0)
            engine.stop()
            print("Speaker pyttsx3 initialized successfully")
        except Exception as e:
            self._enabled = False
            print("Speaker initialization error:", e)

    def _set_english_voice(self, engine):
        voices = engine.getProperty("voices")

        for voice in voices:
            voice_info = f"{voice.id} {voice.name}".lower()

            if (
                "english" in voice_info
                or "united states" in voice_info
                or "united kingdom" in voice_info
                or "en-us" in voice_info
                or "en-gb" in voice_info
                or "zira" in voice_info
                or "david" in voice_info
            ):
                engine.setProperty("voice", voice.id)
                print("English voice selected:", voice.name)
                return

        print("No English voice found. Using default voice.")

    def say(self, text):
        if not self._enabled:
            print("Speaker disabled:", text)
            return

        try:
            print("SPEAKER SAYS:", text)

            engine = pyttsx3.init()
            self._set_english_voice(engine)
            engine.setProperty("rate", 150)
            engine.setProperty("volume", 1.0)

            engine.say(text)
            engine.runAndWait()
            engine.stop()

        except Exception as e:
            print("Speaker error:", e)