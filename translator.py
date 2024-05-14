from google.cloud import translate
# from spellchecker import SpellChecker
from typing import List
from spellwise import Levenshtein



class Translator:
    def __init__(self, source_lang, target_lang):
        # translations is a nested dictionary
        # (text: (target_language_code: translation))
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.translations = {}
        self.use_spellchecker = True
        # self.spellchecker = SpellChecker(language=self.source_lang, distance=2)
        self.levenshtein = Levenshtein()
        self.levenshtein.add_from_path("data/american_english.txt")
        
        # self.client = translate.TranslationServiceClient()
    
    def get_translation(self, texts: List[str]):
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
                result = text_entry.get(self.target_lang)
                if result != None:
                    translations.append(result)
                    continue
            to_translate.append(text)

        # no existing translation, so translate and insert into dictionary
        if len(to_translate) > 0:
            new_translations = self.translate_text(to_translate)
            for i in range(len(new_translations)):
                text = to_translate[i]
                translation = new_translations[i]
                if self.translations.get(text) == None:
                    self.translations[text] = {self.target_lang: translation}
                elif self.translations.get(text).get(self.target_lang) == None:
                    self.translations[text][self.target_lang] = translation
                translations.append(translation)
        return translations

    def get_spellchecked_word(self, word: str):
        suggestions = self.levenshtein.get_suggestions(word)
        if len(suggestions) >= 1 and suggestions[0]['distance'] <= 1:
            return suggestions[0]['word']
        else:
            return word
        # correction = self.spellchecker.correction(word)
        # if correction == None:
        #     return word
        # else:
        #     return correction
        
    def translate_text(self, texts: List[str]):
        return texts
        project_id = "a-fine-matrix"
        location = "global"

        parent = f"projects/{project_id}/locations/{location}"

        # Detail on supported types can be found here:
        # https://cloud.google.com/translate/docs/supported-formats
        response = self.client.translate_text(
                parent = parent,
                contents = texts,
                mime_type = "text/plain",  # mime types: text/plain, text/html
                source_language_code = self.source_lang,
                target_language_code = self.target_lang,
        )

        # Display the translation for each input text provided
        result = []
        for translation in response.translations:
            # print(f"Translated text: {translation.translated_text}")
            result.append(translation.translated_text)

        return result