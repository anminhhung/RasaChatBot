import typing
from typing import Any, Dict, List, Text, Type

from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.constants import ENTITIES, TEXT
from rasa.nlu.utils.spacy_utils import SpacyModel, SpacyNLP
from rasa.nlu.extractors.extractor import EntityExtractorMixin
from rasa.shared.nlu.training_data.message import Message

if typing.TYPE_CHECKING:
    from spacy.tokens.doc import Doc


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR,
    is_trainable=False,
    model_from="SpacyNLP",
)
class MySpacyEntityExtractor(GraphComponent, EntityExtractorMixin):
    """Entity extractor which uses SpaCy."""
    print("### Kiểm tra class MySpacyEntityExtractor ###")
    print("### required_components ###")
    @classmethod
    def required_components(cls) -> List[Type]:
        """Components that should be included in the pipeline before this component."""
        print("required_components")
        return [SpacyNLP]
    print("### get_default_config ###")
    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """The component's default config (see parent class for full docstring)."""
        print("1. get_default_config")
        
        return {
            # by default all dimensions recognized by spacy are returned
            # dimensions can be configured to contain an array of strings
            # with the names of the dimensions to filter for
            "dimensions": None
        }
    print("### __init__ ###")
    def __init__(self, config: Dict[Text, Any]) -> None:
        """Initialize SpacyEntityExtractor."""
        print("3. __init__")
        self._config = config
        print("Config:", self._config)
    print("### create ###")
    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        """Creates a new component (see parent class for full docstring)."""
        print("2. create")
        return cls(config)
    print("### required_packages ###")
    @staticmethod
    def required_packages() -> List[Text]:
        print("required_packages")
        """Lists required dependencies (see parent class for full docstring)."""
        return ["spacy"]
    print("### process ###")
    def process(self, messages: List[Message], model: SpacyModel) -> List[Message]:
        """Extract entities using SpaCy.

        Args:
            messages: List of messages to process.
            model: Container holding a loaded spacy nlp model.

        Returns: The processed messages.
        """
        print("4. process")
        print("messages:", messages)
        for message in messages:
            # can't use the existing doc here (spacy_doc on the message)
            # because tokens are lower cased which is bad for NER
            print("message:", message)
            spacy_nlp = model.model
            print(type(spacy_nlp))
            doc = spacy_nlp(message.get(TEXT))
            print("doc:", doc)
            all_extracted = self.add_extractor_name(self._extract_entities(doc))
            print("ALL EXTRACTED:", all_extracted)
            dimensions = self._config["dimensions"]
            print("dimensions:", dimensions)
            extracted = self.filter_irrelevant_entities(all_extracted, dimensions)
            print("Extracted:", extracted)
            message.set(
                ENTITIES, message.get(ENTITIES, []) + extracted, add_to_output=True
            )
            print("message:", message)

        return messages
    print("### _extract_entities ###")
    @staticmethod
    def _extract_entities(doc: "Doc") -> List[Dict[Text, Any]]:
        print("_extract_entities")
        entities = [
            {
                "entity": ent.label_,
                "value": ent.text,
                "start": ent.start_char,
                "confidence": None,
                "end": ent.end_char,
            }
            for ent in doc.ents
        ]
        print("ENTITIES:", entities)
        return entities