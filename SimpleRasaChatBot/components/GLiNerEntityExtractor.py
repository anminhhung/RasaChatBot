import typing
from typing import Any, Dict, List, Text, Type

from gliner import GLiNER
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.nlu.extractors.extractor import EntityExtractorMixin
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import ENTITIES, TEXT


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR,
    is_trainable=False,
)
class GLiNerEntityExtractor(GraphComponent, EntityExtractorMixin):
    """Entity extractor which uses GLiNER."""
    
    @classmethod
    def required_components(cls) -> List[Type]:
        """Components that should be included in the pipeline before this component."""
        return []

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """The component's default config (see parent class for full docstring)."""
        return {
            # by default all dimensions recognized by spacy are returned
            # dimensions can be configured to contain an array of strings
            # with the names of the dimensions to filter for
            "dimensions": None
        }
    
    def __init__(self, config: Dict[Text, Any]) -> None:
        """Initialize GLiNerEntityExtractor."""
        self._config = config
        self.gliner_model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
        self.gliner_model_threshold = 0.35

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        """Creates a new component (see parent class for full docstring)."""
        return cls(config)
    
    @staticmethod
    def required_packages() -> List[Text]:
        """Lists required dependencies (see parent class for full docstring)."""
        return ["gliner"]
    
    def process(self, messages: List[Message]) -> List[Message]:
        """Extract entities using GLiNER.

        Args:
            messages: List of messages to process.
            model: Container holding a loaded spacy nlp model.

        Returns: The processed messages.
        """
        for message in messages:
            text = message.get(TEXT)
            doc = self.gliner_model.predict_entities(text = text, 
                                                    labels = self._config["dimensions"],
                                                    threshold = self.gliner_model_threshold)
            all_extracted = self.add_extractor_name(self._extract_entities(doc))
            dimensions = self._config["dimensions"]
            extracted = self.filter_irrelevant_entities(all_extracted, dimensions)
            message.set(
                ENTITIES, message.get(ENTITIES, []) + extracted, add_to_output=True
            )

        return messages

    @staticmethod
    def _extract_entities(doc: List[Dict[Text, Any]]) -> List[Dict[Text, Any]]:
        entities = [
            {
                "entity": ent['label'],
                "value": ent['text'],
                "start": ent['start'],
                "confidence": ent['score'],
                "end": ent['end'],
            }
            for ent in doc
        ]
        return entities