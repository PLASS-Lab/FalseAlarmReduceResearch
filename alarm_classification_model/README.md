# 보안약점 정적 분석 알람 분류 (Alarm Classification Model - ACM)
- BWA 모델을 통해 분석된 결과(라인별로 보안약점 발생 가능성 점수)와 다수의 정적 분석결과를 입력하고 Decision 트리 모델을 통해서 알람을 재 분류하는 것이다.
- 해당 모델을 아키텍쳐는 다음과 같다.

<p align="center">
  <img src="./docs/alarm-classification.png" /><br>
  <span>보안약점 정적 분석 알람 분류 모델</span>
</p>

- 다수 정적 분석 도구들을 실험환경 구축 및 BERT 기반으로 보안약점을 분석 모델을 학습이 완료되면 이를 통해서 데이터셋을 하나 만들게 된다. 이는 위에 과 같이 해당 데이터 셋에서 각 CWE의 수많은 테스트 케이스 파일 기준으로 다수 정적 분석 도구들을 통해해서 분석된 결과, BERT 모델의 분석 결과 (라인별로 Attention Score), 메타 데이터가 있다. 다음은 일부 예제 데이터를 보여주고 있다.

<p align="center">
  <br><img src="./docs/acm_input.png" /><br>
  <span>알람 분류 모델에 입력한 일부 예제 데이터</span>
</p>

## 모델 개발환경 및 라이브러리
- Python 

## 모델 실행
- ***classification_models.ipynb*** 파일을 사용해서 각 단계를 명령어를 실행한다.

## 참고
- [Decision tree](https://scikit-learn.org/stable/modules/tree.html)