"""SOTA(kd3_all) train 70k 추론 → 오답 분석용 로짓 저장."""
import numpy as np
from ai_challenge.models.common import predict_logits, load_train_records
from ai_challenge.datasets import SERIALIZE_PRESETS

recs = load_train_records()
lg = predict_logits("runs/kd3_all/model", recs, max_length=640, batch_size=96,
                    serialize_kwargs=SERIALIZE_PRESETS["base"])
np.save("runs/kd3_all/train_logits_self.npy", lg)
print("saved", lg.shape)
