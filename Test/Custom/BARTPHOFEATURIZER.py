import numpy as np
import torch
import typing
import logging
from typing import Any, Text, Dict, List, Type

from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.featurizers.dense_featurizer.dense_featurizer import DenseFeaturizer
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.constants import DENSE_FEATURIZABLE_ATTRIBUTES, FEATURIZER_CLASS_ALIAS
from rasa.shared.nlu.constants import TEXT, FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE
from rasa.utils.tensorflow.constants import POOLING, MEAN_POOLING

from transformers import AutoModel
from Custom.BARTPHOTOKENIZER import BartPhoTokenizer  # Đảm bảo rằng bạn đã tạo BartphoTokenizer trước đó

logger = logging.getLogger(__name__)

@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER, is_trainable=False
)
class BartPhoFeaturizer(DenseFeaturizer, GraphComponent):

    @classmethod
    def required_components(cls) -> List[Type]:
        return [BartPhoTokenizer]

    @staticmethod
    def required_packages() -> List[Text]:
        return ["transformers", "torch"]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {
            **DenseFeaturizer.get_default_config(),
            POOLING: MEAN_POOLING,
        }

    def __init__(self, config: Dict[Text, Any], name: Text) -> None:
        super().__init__(name, config)
        self.pooling_operation = self._config[POOLING]
        self.model = AutoModel.from_pretrained("vinai/bartpho-word")
        self.tokenizer = BartPhoTokenizer(self._config)

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        return cls(config, execution_context.node_name)

    def process(self, messages: List[Message]) -> List[Message]:
        for message in messages:
            for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
                # print("Xử lí message:", message)
                self._set_bartpho_features(message, attribute)
        return messages

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        self.process(training_data.training_examples)
        return training_data

    def _set_bartpho_features(self, message: Message, attribute: Text = TEXT) -> None:
        tokens = self.tokenizer.tokenize(message, attribute)
        # print("# FEATURE TOKEN Ở ĐÂY #:", tokens)
        # print("Message sau khi dc token:", tokens)
        if not tokens:
            return
        # print("Giá trị text")
        texts = [token.text for token in tokens]
        vectorizedText = torch.tensor([self.tokenizer.tokenizer.convert_tokens_to_ids(texts)])

        with torch.no_grad():
            outputs = self.model(vectorizedText)
        sequence_features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        # return sequence_features
        sentence_features = self.aggregate_sequence_features(
            sequence_features, self.pooling_operation
        )

        final_sequence_features = Features(
            sequence_features,
            FEATURE_TYPE_SEQUENCE,
            attribute,
            self._config[FEATURIZER_CLASS_ALIAS],
        )
        message.add_features(final_sequence_features)
        final_sentence_features = Features(
            sentence_features,
            FEATURE_TYPE_SENTENCE,
            attribute,
            self._config[FEATURIZER_CLASS_ALIAS],
        )
        message.add_features(final_sentence_features)
        # print(" ### FINAL FEATURE ###:", final_sentence_features)

    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        pass
