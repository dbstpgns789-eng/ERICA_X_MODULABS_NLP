# 실행 환경과 검증 범위

## 원래 Colab 실행

| 항목 | 확인된 값 |
|---|---|
| GPU | Tesla T4 |
| PyTorch | 2.11.0+cu130 |
| NumPy | 2.4.6 |
| Matplotlib | 3.10.0 |
| SentencePiece | 0.2.2 |
| KoNLPy | 0.6.0 |
| Java | /usr/bin/java가 존재함 |
| 한국어 MeCab | konlpy.tag.Mecab().morphs 및 pos 실행 성공 |

근거: 제공된 전체 노트북의 셀 53·62·142·143. pandas·scikit-learn·MeCab 본체와 사전의 정확한 버전, Colab Python 버전은 저장된 출력에서 확정하지 못했다. 임의의 버전 잠금 파일을 만들지 않았다.

## 시작할 때 필요한 것

Python 패키지는 torch, numpy, pandas, matplotlib, scikit-learn, sentencepiece, konlpy이다. 한국어 MeCab은 Python 모듈 외에 본체와 사전 연결이 필요하다. 일반 Colab의 기본 설치 상태라고 가정하지 않는다. [KoNLPy 설치 안내](https://konlpy.org/en/latest/install/)와 [SentencePiece Python 안내](https://github.com/google/sentencepiece/tree/master/python)를 참고한다.

노트북 시작 셀은 프로젝트 구간의 공통 import를 추가한 것이며, 새로 실행한 결과를 붙이지 않았다. 전체 재실행은 네 모델을 각각 학습한다. 기존 출력 열람만으로 다시 학습할 필요는 없다.

## 정리 과정의 검증

원본 저장 출력과 CSV의 일치, Python 문법, 노트북 구조, 내부 링크와 이미지, 모듈의 작은 CPU end-to-end 연결을 점검한다. 로컬 CPU 점검은 원래 Colab 모델 정확도의 재현이 아니다. 원본 checkpoint와 토크나이저 모델 파일은 제공받지 않았으므로 정확도를 재평가하지 않았다.

확인 결과는 results/review_verification.json에 남긴다. 다음으로 필요한 재현 확인은 새 Colab 런타임에서 MeCab 설치부터 전체 실행하는 것이다.
