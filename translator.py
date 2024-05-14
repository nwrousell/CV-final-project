# from google.cloud import translate

class Translator:
    def __init__(self):
        # translations is a nested dictionary
        # (text: (target_language_code: translation))
        self.translations = {}
    
    def get_translation(self, source_language: str, target_language: str, text: str):
        # check if translation already exists
        text_entry = self.translations.get(text)
        if text_entry != None:
            result = text_entry.get(target_language)
            if result != None:
                return result
        
        # no existing translation, so translate and insert into dictionary
        # print("Translating...")
        translation = self.translate_text(text, target_language)
        if self.translations.get(text) == None:
            self.translations[text] = {target_language: translation}
            print(self.translations)
        elif self.translations.get(text).get(target_language) == None:
            self.translations[text][target_language] = translation
        return translation
        
    def translate_text(self, text: str, target_language: str):
        return text
        # return "Héllø wörld ǎgain but with diaçritics ;)"
        # return "Really long string to test line breaks and bounding box fitting! How cool?!! Whataboutareallylongwordthatdefinitelydoesn'tfitinoneline?"
        
        client = translate.TranslationServiceClient()

        project_id = "a-fine-matrix"
        location = "global"

        parent = f"projects/{project_id}/locations/{location}"

        # Translate text from English to French
        # Detail on supported types can be found here:
        # https://cloud.google.com/translate/docs/supported-formats
        response = client.translate_text(
                parent = parent,
                contents = [text],
                mime_type = "text/plain",  # mime types: text/plain, text/html
                source_language_code = source_language,
                target_language_code = target_language,
        )

        # Display the translation for each input text provided
        result = ""
        for translation in response.translations:
            print(f"Translated text: {translation.translated_text}")
            result += translation.translated_text

        return result