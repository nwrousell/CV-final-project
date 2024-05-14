from google.cloud import translate
from spellchecker import SpellChecker
from typing import List


class Translator:
    def __init__(self):
        # translations is a nested dictionary
        # (text: (target_language_code: translation))
        self.translations = {}
        self.use_spellchecker = True
        self.spellchecker = SpellChecker()
    
    def get_translation(self, source_language: str, target_language: str, texts: List[str]):
        translations = []
        to_translate = []
        for text in texts:
            # cleanup and spellcheck
            text = text.split(" ")[0]
            if self.use_spellchecker:
                text = self.get_spellchecked_word(text)

            # check if translation already exists
            text_entry = self.translations.get(text)
            if text_entry != None:
                result = text_entry.get(target_language)
                if result != None:
                    translations.append(result)
                    continue
            to_translate.append(text)

        # no existing translation, so translate and insert into dictionary
        if len(to_translate) > 0:
            new_translations = self.translate_text(source_language, target_language, to_translate)
            for i in range(len(new_translations)):
                text = to_translate[i]
                translation = new_translations[i]
                if self.translations.get(text) == None:
                    self.translations[text] = {target_language: translation}
                elif self.translations.get(text).get(target_language) == None:
                    self.translations[text][target_language] = translation
                translations.append(translation)
        return translations

    def get_spellchecked_word(self, word: str):
        correction = self.spellchecker.correction(word)
        if correction == None:
            return word
        else:
            return correction
        
    def translate_text(self, source_language: str, target_language: str, texts: List[str]):
        return texts
        # return "Héllø wörld ǎgain but with diaçritics ;)"
        # return "Really long string to test line breaks and bounding box fitting! How cool?!! Whataboutareallylongwordthatdefinitelydoesn'tfitinoneline?"
        
        client = translate.TranslationServiceClient()

        project_id = "a-fine-matrix"
        location = "global"

        parent = f"projects/{project_id}/locations/{location}"

        # Detail on supported types can be found here:
        # https://cloud.google.com/translate/docs/supported-formats
        response = client.translate_text(
                parent = parent,
                contents = texts,
                mime_type = "text/plain",  # mime types: text/plain, text/html
                source_language_code = source_language,
                target_language_code = target_language,
        )

        # Display the translation for each input text provided
        result = []
        for translation in response.translations:
            # print(f"Translated text: {translation.translated_text}")
            result.append(translation.translated_text)

        return result