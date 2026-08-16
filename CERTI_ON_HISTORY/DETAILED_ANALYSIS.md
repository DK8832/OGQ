# CERTI:ON Flutter Project — Deep Version & File Analysis

## 1. 분석 범위

- 원본 archive entries: **4,398개**
- 실제 파일: **3,390개**
- 원본 비압축 총량: **618.2 MiB**
- 역사 프로젝트 폴더: **24개**
- 정렬 기준: 각 최상위 프로젝트 폴더의 ZIP 보존 수정시각(KST), 가장 오래된 것부터 v01~v24.

## 2. 전체 파일 분류

| Category | Files | Size (MiB) |
|---|---:|---:|
| Generated/build/cache | 1937 | 593.537 |
| Image assets | 437 | 9.594 |
| Flutter app source | 24 | 6.551 |
| Node backend | 134 | 2.976 |
| Official data assets | 46 | 2.673 |
| Other | 5 | 1.114 |
| Flutter web | 105 | 0.623 |
| Android native/project | 100 | 0.295 |
| Documentation/report | 145 | 0.247 |
| Build/utility scripts | 76 | 0.219 |
| Launcher/script | 142 | 0.197 |
| Flutter/project config | 120 | 0.142 |
| IDE config | 82 | 0.053 |
| Tests | 23 | 0.014 |
| GitHub workflow | 14 | 0.006 |

### 핵심 해석

- 원본 용량의 대부분은 `build/`, `.dart_tool/`, Gradle/Android build cache 같은 **재생성 가능한 산출물**입니다.
- 실제 Flutter/Backend/asset/config/source는 훨씬 작기 때문에 GitHub 공개용 history package에서는 build/cache를 제외했습니다.
- 버전별 source package에는 Flutter `lib/`, `assets/`, `backend/`, Android project source(존재하는 버전), web, test, tools, BAT/PowerShell, project config를 유지했습니다.

## 3. 보안 검사

- 역사 스냅샷 5개의 `backend/.env`에서 OpenAI 형식 실제 secret token pattern을 발견했습니다.
- 값은 보고서에 기록하지 않았고 공개용 package에서 모든 `.env`를 제외했습니다.
- 공개용 package 재검사 결과 known secret-pattern match는 0건입니다.
- `android/local.properties`도 로컬 SDK 경로 파일이므로 공개 package에서 제외했습니다.

## 4. 아키텍처 진화

### Phase A — v01~v09: Flutter 시제품 → 공식 데이터/클라우드 AI 안정화

- v01 ULTIMATE: 공식 일정 101건+상시시험 6종, 이미지, Node backend, sync workflow 통합.
- v02 경량 시제품으로 축소한 뒤 v03에서 ULTIMATE 구조를 다시 복원.
- v04는 이미지/UI 변경, v05~v09는 OpenAI backend와 Windows/Chrome/Android 실행·연결 안정화가 중심.
- v09에서 localhost/CORS/port 충돌과 Flutter Web ListTile assertion을 별도로 수정.

### Phase B — v10~v15: OpenAI → PC Ollama/Qwen3 로컬 AI

- v10에서 클라우드 API 의존을 제거하고 Ollama+Qwen3 14B 중심으로 전환.
- v12는 Android wrapper까지 포함한 original local Qwen14B 기준점을 만들었고, v13~v14는 smart chat/Chrome 실행 문제를 다듬음.
- v15는 14B/8B/4B 모델 선택과 생성 요청 no-timeout 구조를 추가.

### Phase C — v16~v24: 휴대폰 Standalone AI + 선택적 PC AI

- v16에서 가장 큰 구조 전환: 휴대폰 1.7B/0.6B GGUF 로컬 AI를 기본, PC Ollama를 optional 고성능 모드로 분리.
- v18 NDK 28.2 pin, v19 Ollama 자동 시작/방화벽/빠른 모델 다운로드 복구, v20 UTF-8 한글 token streaming patch.
- v21 Qwen3 empty-answer fallback, v22 reasoning leak guard, v23 placeholder/stale-backend guard.
- v24에서 실행 BAT를 `RUN_CERTION_ALL.bat` 하나로 통합하고 build/runtime 구조를 정리.

## 5. 24개 버전 상세 인덱스

| Hist | Modified KST | Original folder | pubspec | Original files | Source files | Source ZIP MiB | + / - / ~ | 핵심 변화 |
|---|---|---|---|---:|---:|---:|---:|---|
| v01 | 2026-08-15 15:53:22 | `certi_on_ogq_ultimate` | `2.0.0+2` | 251 | 58 | 0.594 | +58 / -0 / ~0 | 공식 일정 101건+상시시험 6종, 이미지·백엔드·자동 동기화까지 포함한 첫 ULTIMATE 통합본. |
| v02 | 2026-08-16 00:12:44 | `certi_on_ogq` | `1.0.0+1` | 194 | 22 | 0.079 | +0 / -36 / ~9 | 백엔드·공식 data asset·이미지·workflow를 제거한 경량 데모 시제품. |
| v03 | 2026-08-16 01:11:24 | `certi_on_ogq_ultimate1` | `2.0.0+2` | 251 | 58 | 0.595 | +36 / -0 / ~9 | ULTIMATE 전체 구조 복원. |
| v04 | 2026-08-16 01:16:58 | `certi_on_ogq_ultimate2` | `2.0.0+2` | 252 | 58 | 0.526 | +0 / -0 / ~21 | UI/이미지 asset 대규모 갱신. |
| v05 | 2026-08-16 01:42:48 | `certi_on_ogq_ultimate3` | `2.0.0+2` | 255 | 61 | 0.541 | +3 / -0 / ~11 | OpenAI key setup/Android configure/backend 확장. |
| v06 | 2026-08-16 01:51:54 | `certi_on_ogq_ultimate4` | `2.0.0+2` | 241 | 47 | 0.485 | +0 / -14 / ~4 | 중복 main/문서/점검 script 정리. |
| v07 | 2026-08-16 01:59:54 | `z_real_final_final_final_certi_on_ogq_ai_fixed` | `2.0.0+2` | 244 | 50 | 0.489 | +3 / -0 / ~4 | VS Code/Chrome AI 실행 흐름 보강. |
| v08 | 2026-08-16 11:09:16 | `CERTI_ON_OGQ_FULL_FIXED` | `2.0.0+2` | 241 | 52 | 0.490 | +6 / -4 / ~7 | RUN_CERTION 중심 Full Fixed. |
| v09 | 2026-08-16 11:40:18 | `CERTI_ON_OGQ_FULL_FIXED_V2` | `2.0.0+2` | 240 | 51 | 0.489 | +3 / -4 / ~9 | Web ListTile assertion/localhost Failed to fetch 수정. |
| v10 | 2026-08-16 12:08:46 | `CERTI_ON_OGQ_LOCAL_AI_SMART` | `2.0.0+2` | 56 | 55 | 0.492 | +4 / -0 / ~11 | OpenAI → Ollama+Qwen3 로컬 AI 전환. |
| v11 | 2026-08-16 12:14:02 | `CERTI_ON_OGQ_LOCAL_AI_SMART_V2` | `2.0.0+2` | 57 | 56 | 0.492 | +1 / -0 / ~7 | Local AI 실행/모델 선택 사용성 개선. |
| v12 | 2026-08-16 13:27:22 | `CERTI_ON_OGQ_ORIGINAL_LOCAL_QWEN14B` | `2.0.0+2` | 273 | 77 | 0.551 | +30 / -9 / ~10 | Android wrapper 포함 original local Qwen14B. |
| v13 | 2026-08-16 13:38:06 | `CERTI_ON_OGQ_QWEN14B_SMART_CHAT_FIXED` | `2.0.0+2` | 88 | 77 | 0.552 | +0 / -0 / ~3 | Smart Chat prompt/응답 처리 개선. |
| v14 | 2026-08-16 13:44:00 | `CERTI_ON_OGQ_QWEN14B_SMART_CHAT_FIXED_V2` | `2.0.0+2` | 122 | 79 | 0.554 | +2 / -0 / ~1 | Chrome/web-server 실행 안정화. |
| v15 | 2026-08-16 14:03:24 | `FINAL_ULTIMATE_CERTI_ON_OGQ_AI_MODEL_SELECTOR_NO_TIMEOUT` | `2.0.0+2` | 125 | 82 | 0.559 | +3 / -0 / ~7 | 14B/8B/4B selector + no-timeout. |
| v16 | 2026-08-16 18:37:54 | `FINAL_FINAL_ULTIMATE_CERTI_ON_OGQ_STANDALONE_AI_ALL_FIXED` | `2.1.0+3` | 48 | 47 | 0.469 | +12 / -47 / ~7 | 휴대폰 1.7B/0.6B standalone AI 대전환. |
| v17 | 2026-08-16 18:51:06 | `CERTI_ON_OGQ_STANDALONE_AI_ALL_FIXED_V2` | `2.1.1+4` | 50 | 49 | 0.468 | +2 / -0 / ~9 | Android Gradle/SDK build 검증 보강. |
| v18 | 2026-08-16 19:06:26 | `CERTI_ON_OGQ_STANDALONE_AI_ALL_FIXED_V3` | `2.1.2+5` | 52 | 51 | 0.470 | +3 / -1 / ~10 | NDK 28.2 pin + Wi-Fi PC AI 주소 개선. |
| v19 | 2026-08-16 19:23:58 | `CERTI_ON_OGQ_STANDALONE_AI_ALL_FIXED_V4` | `2.1.3+6` | 56 | 55 | 0.475 | +4 / -0 / ~7 | Ollama autostart/firewall/0.6B download recovery. |
| v20 | 2026-08-16 19:37:08 | `CERTI_ON_OGQ_STANDALONE_AI_ALL_FIXED_V5` | `2.1.4+7` | 59 | 58 | 0.480 | +4 / -1 / ~4 | UTF-8 한글 JNI token streaming patch. |
| v21 | 2026-08-16 19:42:04 | `CERTI_ON_OGQ_STANDALONE_AI_ALL_FIXED_V6` | `2.1.4+7` | 63 | 62 | 0.484 | +4 / -0 / ~5 | Qwen3 empty-answer fallback + APK builder. |
| v22 | 2026-08-16 20:07:34 | `CERTI_ON_OGQ_STANDALONE_AI_ALL_FIXED_V7` | `2.1.4+7` | 66 | 65 | 0.487 | +3 / -0 / ~5 | reasoning leak guard + Gradle lock cleanup. |
| v23 | 2026-08-16 20:42:02 | `CERTI_ON_OGQ_STANDALONE_AI_ALL_FIXED_V8` | `2.1.4+7` | 71 | 70 | 0.492 | +5 / -0 / ~6 | placeholder/stale-backend guard + 3X test. |
| v24 | 2026-08-16 21:09:12 | `CERTI_ON_OGQ_STANDALONE_AI_ALL_FIXED_V9` | `2.2.0+9` | 35 | 35 | 0.458 | +4 / -39 / ~5 | RUN_CERTION_ALL 통합 및 최종 최적화. |

## 6. 버전 번호 주의사항

폴더 수정 시각과 `pubspec.yaml`의 `version:`은 별개의 정보입니다. 예를 들어 시간상 v02인 폴더의 pubspec은 `1.0.0+1`, v01/v03~v15 다수는 `2.0.0+2`입니다. 따라서 GitHub 역사 정렬에는 **v01~v24 history ID**를 사용하고 앱 자체 버전은 별도 metadata로 유지하는 것이 안전합니다.

## 7. 공개용 패키징 결정

사용자 요청의 두 방식 중 **버전별 프로젝트 전체를 source ZIP으로 보존**하는 방식을 선택했습니다. 각 ZIP은 해당 버전의 source/config/assets/backend를 독립적으로 포함합니다. 원본 build/cache는 환경에서 재생성되는 산출물이며 원본 용량 대부분을 차지하므로 제외합니다. 원본 4,398개 엔트리는 별도 full manifest에서 하나씩 추적합니다.
