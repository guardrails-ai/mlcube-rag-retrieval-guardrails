from enum import Enum


class TextLanguage(str, Enum):
    """Text language of the text in NLP tasks

    Fields
    ------

    ITALIAN
    ENGLISH
    """

    ITALIAN = "italian"
    ENGLISH = "english"
