import typing
from typing import Dict, Text, List, Any, Optional, Type

from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.nlu.utils.spacy_utils import SpacyNLP
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.nlu.constants import SPACY_DOCS
from rasa.shared.nlu.training_data.message import Message

if typing.TYPE_CHECKING:
    from spacy.tokens.doc import Doc

POS_TAG_KEY = "pos"


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER, is_trainable=False
)
class MySpacyTokenizer(Tokenizer):
    """Tokenizer that uses SpaCy."""
    # print("Kiểm tra khởi tạo tokenizer")
    @classmethod
    def required_components(cls) -> List[Type]:
        """Components that should be included in the pipeline before this component."""
        return [SpacyNLP]
    # print("### required_components ###")
    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """The component's default config (see parent class for full docstring)."""
        return {
            # Flag to check whether to split intents
            "intent_tokenization_flag": False,
            # Symbol on which intent should be split
            "intent_split_symbol": "_",
            # Regular expression to detect tokens
            "token_pattern": None,
            # Symbol on which prefix should be split
            "prefix_separator_symbol": None,
        }
    # print("### get_default_config ###")
    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return ["spacy"]
    # print("### required_packages ###")
    def _get_doc(self, message: Message, attribute: Text) -> Optional["Doc"]:
        return message.get(SPACY_DOCS[attribute])
    # print("### _get_doc ###")
    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        """Tokenizes the text of the provided attribute of the incoming message."""
        doc = self._get_doc(message, attribute)
        # print("DOC:", doc)
        # print(type(doc))
        if not doc:
            return []

        tokens = [
            Token(
                t.text, t.idx, lemma=t.lemma_, data={POS_TAG_KEY: self._tag_of_token(t)}
            )
            for t in doc
            if t.text and t.text.strip()
        ]
        # print("# TOKENS #:", tokens)
        print("### TOKENIZED MESSAGE ###:", self._apply_token_pattern(tokens))
        # print("## TYPE ##:", type(self._apply_token_pattern(tokens)[0]))
        return self._apply_token_pattern(tokens)
    # print("### tokenize ###")
    @staticmethod
    def _tag_of_token(token: Any) -> Text:
        import spacy

        if spacy.about.__version__ > "2" and token._.has("tag"):
            return token._.get("tag")
        else:
            return token.tag_
    # print("### _tag_of_token ###")