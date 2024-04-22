from csref.config import LazyCall
from csref.models.speech_encoders.wav2vec2 import Wav2vec2
from csref.models.text_encoder.bert import Bert
from csref.models.csref_CSA import ContrastiveSemanticAlignment

model = LazyCall(ContrastiveSemanticAlignment)(
    speech_encoder=LazyCall(Wav2vec2)(
        hidden_size=768,
        flat_glimpses=1,
        dropout_rate=0.1,
        target_sr=16000,
        pretrained_path="data/weights/wav2vec2-base",
        freeze_model=True,
        use_one_hidden_state_as_feat=True,
        hidden_state_index=-1,
        use_att_flat_mask=True,
        freeze_layers=False,  # just freeze feature feature encoder
    ),
    text_encoder=LazyCall(Bert)(
        pretrained_path="data/weights/bert-large-uncased",
        freeze_model=True,
        hidden_state_index=-1,
    )
)
