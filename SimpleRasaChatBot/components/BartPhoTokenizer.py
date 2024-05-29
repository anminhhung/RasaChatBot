import typing
from typing import Dict, Text, List, Any, Optional, Type

from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.nlu.constants import TEXT
from rasa.shared.nlu.training_data.message import Message

from transformers import AutoTokenizer

if typing.TYPE_CHECKING:
    from spacy.tokens.doc import Doc

POS_TAG_KEY = "pos"

@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER, is_trainable=False
)
class BartPhoTokenizer(Tokenizer):
    @classmethod
    def required_components(cls) -> List[Type]:
        return []

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
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

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return ["transformers"]

    def __init__(self, config: Dict[Text, Any]) -> None:
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-word")

    def tokenize(self, message: Message, attribute: Text = TEXT) -> List[Token]:
        """Tokenizes the text of the provided attribute of the incoming message."""
        sentence = message.get(attribute)
        if not sentence:
            return []

        # Use BARTpho tokenizer to tokenize the text
        token_text = self.tokenizer.tokenize(sentence)
        tokens = []
        for token in token_text:
            start_offset = sentence.find(token)
            end_offset = start_offset + len(token) - 1
            tokens.append(Token(text=token, start=start_offset, end=end_offset))

        return self._apply_token_pattern(tokens)
    
    @staticmethod
    def _tag_of_token(token: Any) -> Text:
        # Implement POS tagging if required, or remove if not necessary
        return ""

