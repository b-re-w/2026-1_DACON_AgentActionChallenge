"""모델·공통 학습 로직 패키지.

공통 빌딩 블록은 ``common`` 에 있다(build_model/build_tokenizer/build_training_args/
WeightedTrainer/predict_logits 등). 커스텀 모델(헤드 변경·아키텍처 수정)이 필요하면
이 패키지에 새 모듈로 추가한다.
"""

from __future__ import annotations

from .common import *  # noqa: F401,F403
from . import common  # noqa: F401
