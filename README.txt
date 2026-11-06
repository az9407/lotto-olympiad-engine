Lotto Olympiad Engine — 다음 회차 종합 로그오즈 점수 상위 15개

실행
  python predict_top15.py
최신 엑셀 행까지 읽고 다음 회차/날짜를 자동 계산하며 CSV/HTML 저장 후
공개 GitHub 저장소 생성·커밋·푸시·원격 커밋 검증까지 실행합니다.

입력 파일을 명시하려면
  python predict_top15.py --input "로또_역대_당첨번호.xlsx"
로컬 계산만 하려면 --no-push, GitHub 업로드만 다시 하려면
  python publish_github.py

원본은 현재 폴더의 xlsx 또는 .codex-attachments 아래에서 찾습니다.
복수 입력이면 수정 시각이 가장 최근인 파일을 사용하므로 --input을 권장합니다.
원본 엑셀은 수정하거나 GitHub에 업로드하지 않습니다.
분석 라이브러리가 없는 기본 Python에서는 설치된 데스크톱 분석 런타임을
자동으로 찾아 재실행합니다. 다른 PC에서는 Python3.10+ 및
  python -m pip install -r requirements.txt
를 준비하십시오.

자동 분할과 활성 조건
최근250회는 검증, 이전 전체는 학습입니다. 현재는1~998/999~1248이며
새1249회 행 추가 시1~999/1000~1249로 이동하고1250회를 예측합니다.
조건 정의의 분위수는 학습자료에서만 구하며 목표 회차의 날짜/회차와
실제 T-1,T-2,T-3의 관측값으로 활성 여부를 평가합니다.
A: 학습 조건 표본>=100, 검증 조건 표본>=30, 양 구간 적중률>=55%.
B: 전체 경험적 적중률/정확 이론통과율>1.05 및 검증 적중률>=55%.
정확 이론통과율은8,145,060개 본번호 조합 전체를 열거해 계산합니다.

채점
A는 검증 로그오즈 우위 순서로 최대100개(최소30개)를 선정합니다.
동일한 미래 결론은 A에서 한 번 반영하고, 모든 해당 선행조건은
eligible_active_rules.csv에 기록합니다. 적격 B 필터는 전부 포함합니다.
w_r=logit(검증 적중률)-logit(이론확률 q).
목표 회차로 고정한 조건식을 동일 학습 번호조합에 대입한 피어슨 상관의
절댓값이 기존 조건식과0.5 이상이면
적용 가중치=w_r*(1-최대|rho|)입니다. 같은 사건은 가중치0입니다.
0/1 적중률 경계에서만 유한 로그값을 위한0.5/(검증N+1) 절단을 합니다.
S_rule=Σ 적용가중치*규칙충족여부.
S_num=0.05*Σ 조합내6번호 log(번호별 경험적 출현 추정치).
S=S_rule+S_num. 최종 CSV는 소수점8자리로 표시합니다.

번호별 전이 모델
번호identity, 직전1~3회 이월/보너스/이웃수, 부호있는 거리와 최소거리,
번호·끝수 미출현 길이, 최근5/10/20회 번호빈도와 끝수빈도를 사용합니다.
확률은 경험빈도를6/45 쪽으로 축소하고 고정 logit 혼합 후 sum(p)=6으로
정규화합니다. 검증은 학습 계수를 고정하고, 다음 회차 예측은 같은 명세를
전체 관측에 재학습합니다. 미세 성분과 검증 Brier/log-loss를 공개합니다.

하드 필터와 다양성
실제 적용 가중치가 높은 핵심3개 규칙의 교집합을 적용합니다.
비어 있거나10000개 미만인 교집합, 모든 후보에 같은 번호를 강제하는
조건 등은 이유를 기록하고 다음 적격 규칙을 검사합니다.
전수 S 순위를 내려가며 이미 선정된 조합과 공통번호<=3,
어떤 번호도15개 전체에서 최대4회 조건을 만족하는 다음 최고점을 뽑습니다.
15개를 채우지 못하거나 활성 규칙이 부족하면 오류를 냅니다.
조건을 무시하거나 전체 조합을 그대로 반환하지 않습니다.

동점과 점수 의미
임의 추출이나 무작위 jitter를 사용하지 않습니다. float64 S를 내림차순으로
정렬하고 정확한 동점만 번호 사전순으로 처리합니다. 실제 실수 동점과
8자리 반올림 표시 동점을 전수 감사하여 HTML과 로컬 JSON에 공개합니다.
경험적 유한자료에서 모든 미래 회차의 점수가 유일함을 보장할 수 없습니다.
S는 검증자료로 규칙을 고른 합성 순위 점수이며 실제 1등 당첨확률로
보정된 값이 아닙니다. 전체 점수 순위와 다양성 선정 순위는 구분합니다.

결과 위치
results/round_<목표회차>/top15.csv
results/round_<목표회차>/active_rules.csv
results/round_<목표회차>/eligible_active_rules.csv
results/round_<목표회차>/number_probabilities.csv
results/round_<목표회차>/report.html
위 CSV/HTML과 소스·테스트만 공개합니다. 원본xlsx, 인증정보, 로그,
캐시와 실행 JSON은 업로드하지 않습니다. run_summary.json은 로컬에서
원점수/동점 감사/모델 진단을 보존합니다.

GitHub
GH_TOKEN/GITHUB_TOKEN, GitHub CLI 또는 Git credential helper로 인증합니다.
토큰을 출력하거나 URL/파일에 저장하지 않습니다. 기본 공개 저장소는
인증계정/lotto-olympiad-engine이며 feature/continuous-probability-engine
브랜치를 사용합니다. 기존 원격을 덮어쓰거나 강제 푸시/main 병합하지 않습니다.

검증
  python -m unittest discover -v
  python exact_stats.py
  python combinatorial.py --cache-dir .cache

모듈
predict_top15.py: 전수 점수화/하드필터/다양성/출력/자동푸시
rule_engine.py: 입력·자동 분할·실제 활성 A/B 탐색
lotto_analysis.py: 수학적 특징/조건/사건 정의
combinatorial.py: 정확 전체 조합 분포와 범위필터
number_model.py: 개별 번호 경험적 전이 모델
publish_github.py: 공개 원격 생성·허용 파일만 커밋·푸시·검증
generate_candidates.py,rank_candidates.py: 통합 CLI 호환 진입점
