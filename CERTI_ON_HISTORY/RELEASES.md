# CERTI:ON — Version Archive / Release Catalog

> Chronology reconstructed from the original Windows folder modified timestamps preserved in `flutter project(6).zip`.

## Important packaging/security policy

- Each history version has its own complete source-project package prepared from the original folder.
- Build/cache folders (`build/`, `.dart_tool/`, Gradle caches) are excluded from public source packages because they are generated.
- Compiled outputs and machine-local `android/local.properties` are excluded.
- **Five historical `backend/.env` files contained OpenAI-style secret tokens. They are intentionally excluded from every public package.**
- `.env.example` remains when available.

## Chronological index — oldest → newest

| Hist. | Modified (KST) | Original folder | pubspec | Source files | Δ + / - / ~ |
|---|---|---|---|---:|---:|
| v01 | 2026-08-15 15:53:22 | `certi_on_ogq_ultimate` | `2.0.0+2` | 58 | +58 / -0 / ~0 |
| v02 | 2026-08-16 00:12:44 | `certi_on_ogq` | `1.0.0+1` | 22 | +0 / -36 / ~9 |
| v03 | 2026-08-16 01:11:24 | `certi_on_ogq_ultimate1` | `2.0.0+2` | 58 | +36 / -0 / ~9 |
| v04 | 2026-08-16 01:16:58 | `certi_on_ogq_ultimate2` | `2.0.0+2` | 58 | +0 / -0 / ~21 |
| v05 | 2026-08-16 01:42:48 | `certi_on_ogq_ultimate3` | `2.0.0+2` | 61 | +3 / -0 / ~11 |
| v06 | 2026-08-16 01:51:54 | `certi_on_ogq_ultimate4` | `2.0.0+2` | 47 | +0 / -14 / ~4 |
| v07 | 2026-08-16 01:59:54 | `z_real_final_final_final_certi_on_ogq_ai_fixed` | `2.0.0+2` | 50 | +3 / -0 / ~4 |
| v08 | 2026-08-16 11:09:16 | `CERTI_ON_OGQ_FULL_FIXED` | `2.0.0+2` | 52 | +6 / -4 / ~7 |
| v09 | 2026-08-16 11:40:18 | `CERTI_ON_OGQ_FULL_FIXED_V2` | `2.0.0+2` | 51 | +3 / -4 / ~9 |
| v10 | 2026-08-16 12:08:46 | `CERTI_ON_OGQ_LOCAL_AI_SMART` | `2.0.0+2` | 55 | +4 / -0 / ~11 |
| v11 | 2026-08-16 12:14:02 | `CERTI_ON_OGQ_LOCAL_AI_SMART_V2` | `2.0.0+2` | 56 | +1 / -0 / ~7 |
| v12 | 2026-08-16 13:27:22 | `CERTI_ON_OGQ_ORIGINAL_LOCAL_QWEN14B` | `2.0.0+2` | 77 | +30 / -9 / ~10 |
| v13 | 2026-08-16 13:38:06 | `CERTI_ON_OGQ_QWEN14B_SMART_CHAT_FIXED` | `2.0.0+2` | 77 | +0 / -0 / ~3 |
| v14 | 2026-08-16 13:44:00 | `CERTI_ON_OGQ_QWEN14B_SMART_CHAT_FIXED_V2` | `2.0.0+2` | 79 | +2 / -0 / ~1 |
| v15 | 2026-08-16 14:03:24 | `FINAL_ULTIMATE_CERTI_ON_OGQ_AI_MODEL_SELECTOR_NO_TIMEOUT` | `2.0.0+2` | 82 | +3 / -0 / ~7 |
| v16 | 2026-08-16 18:37:54 | `FINAL_FINAL_ULTIMATE_CERTI_ON_OGQ_STANDALONE_AI_ALL_FIXED` | `2.1.0+3` | 47 | +12 / -47 / ~7 |
| v17 | 2026-08-16 18:51:06 | `CERTI_ON_OGQ_STANDALONE_AI_ALL_FIXED_V2` | `2.1.1+4` | 49 | +2 / -0 / ~9 |
| v18 | 2026-08-16 19:06:26 | `CERTI_ON_OGQ_STANDALONE_AI_ALL_FIXED_V3` | `2.1.2+5` | 51 | +3 / -1 / ~10 |
| v19 | 2026-08-16 19:23:58 | `CERTI_ON_OGQ_STANDALONE_AI_ALL_FIXED_V4` | `2.1.3+6` | 55 | +4 / -0 / ~7 |
| v20 | 2026-08-16 19:37:08 | `CERTI_ON_OGQ_STANDALONE_AI_ALL_FIXED_V5` | `2.1.4+7` | 58 | +4 / -1 / ~4 |
| v21 | 2026-08-16 19:42:04 | `CERTI_ON_OGQ_STANDALONE_AI_ALL_FIXED_V6` | `2.1.4+7` | 62 | +4 / -0 / ~5 |
| v22 | 2026-08-16 20:07:34 | `CERTI_ON_OGQ_STANDALONE_AI_ALL_FIXED_V7` | `2.1.4+7` | 65 | +3 / -0 / ~5 |
| v23 | 2026-08-16 20:42:02 | `CERTI_ON_OGQ_STANDALONE_AI_ALL_FIXED_V8` | `2.1.4+7` | 70 | +5 / -0 / ~6 |
| v24 | 2026-08-16 21:09:12 | `CERTI_ON_OGQ_STANDALONE_AI_ALL_FIXED_V9` | `2.2.0+9` | 35 | +4 / -39 / ~5 |

---

## Release feed — newest first

## v24 — `CERTI_ON_OGQ_STANDALONE_AI_ALL_FIXED_V9`
**Folder modified:** 2026-08-16 21:09:12 KST  
**Internal pubspec:** `2.2.0+9`

Standalone V9 최종 정리/최적화. `RUN_CERTION_ALL.bat` 하나로 SDK/NDK/빌드/업데이트/방화벽/Ollama/PC backend/앱 실행을 통합. PC AI 기본 4B, LAN IP 자동 삽입, backend 데이터 메모리 로드, 중복 스크립트·보고서 제거.

Files: 35 · delta: +4 / -39 / ~5.

## v23 — `CERTI_ON_OGQ_STANDALONE_AI_ALL_FIXED_V8`
**Modified:** 2026-08-16 20:42:02 KST · **pubspec:** `2.1.4+7`

빈 문자열·`...` placeholder를 거부하고 공식 데이터 기반 deterministic fallback을 추가. 오래된 8787 backend 종료·정확한 backendVersion 검증과 3회 연속 AI 응답 테스트를 도입. 일반 실행 경로의 `flutter analyze`를 제거해 Windows DartWorker 불안정도 완화.

Files: 70 · delta: +5 / -0 / ~6.

## v22 — `CERTI_ON_OGQ_STANDALONE_AI_ALL_FIXED_V7`
**Modified:** 2026-08-16 20:07:34 KST · **pubspec:** `2.1.4+7`

휴대폰과 PC 양쪽에서 Qwen3 내부 추론이 사용자 화면에 새어 나오는 문제를 차단. raw Qwen3 prompt/empty-think prefill, reasoning guard와 PC raw generate fallback을 추가. Gradle/Kotlin daemon·파일 잠금 정리도 포함.

Files: 65 · delta: +3 / -0 / ~5.

## v21 — `CERTI_ON_OGQ_STANDALONE_AI_ALL_FIXED_V6`
**Modified:** 2026-08-16 19:42:04 KST · **pubspec:** `2.1.4+7`

Qwen3가 token budget을 thinking에 소모해 빈 답변을 내는 문제를 `think=false`, `/no_think`, chat 재시도, `/api/generate` fallback으로 해결. `<think>` 블록 제거와 휴대폰 없이도 Release APK를 만드는 `MAKE_FLUTTER_APK.bat` 추가.

Files: 62 · delta: +4 / -0 / ~5.

## v20 — `CERTI_ON_OGQ_STANDALONE_AI_ALL_FIXED_V5`
**Modified:** 2026-08-16 19:37:08 KST · **pubspec:** `2.1.4+7`

llama.cpp의 token piece가 UTF-8 문자 중간에서 끊길 때 한글이 U+FFFD로 깨지는 원인을 수정. `patch_llama_utf8.ps1`로 JNI token byte를 경계에 맞게 buffer하고 UTF-16 NewString으로 전달하도록 패치.

Files: 58 · delta: +4 / -1 / ~4.

## v19 — `CERTI_ON_OGQ_STANDALONE_AI_ALL_FIXED_V4`
**Modified:** 2026-08-16 19:23:58 KST · **pubspec:** `2.1.3+6`

PC AI의 Ollama `fetch failed` 문제 대응. Ollama API가 꺼져 있으면 `ollama serve`를 자동 시작하고 ECONNREFUSED 원인까지 표시. 방화벽 LocalSubnet 8787 규칙을 추가. 휴대폰 0.6B 빠른 모델 URL, HTTP 416 resume recovery, 손상된 GGUF/.part 정리도 수정.

Files: 55 · delta: +4 / -0 / ~7.

## v18 — `CERTI_ON_OGQ_STANDALONE_AI_ALL_FIXED_V3`
**Modified:** 2026-08-16 19:06:26 KST · **pubspec:** `2.1.2+5`

Android NDK를 `28.2.13676358`로 고정하고 minify/resource shrink 비활성화를 검증. Wi-Fi PC AI 주소 자동 탐색/표시 보강. 정적/구성 시뮬레이션 67/67 PASS 기록.

Files: 51 · delta: +3 / -1 / ~10.

## v17 — `CERTI_ON_OGQ_STANDALONE_AI_ALL_FIXED_V2`
**Modified:** 2026-08-16 18:51:06 KST · **pubspec:** `2.1.1+4`

Standalone 초기본의 Android Gradle/SDK/NDK 준비와 build 설정 검증을 보강. `verify_android_gradle.ps1` 및 V2 시뮬레이션 보고서 추가.

Files: 49 · delta: +2 / -0 / ~9.

## v16 — `FINAL_FINAL_ULTIMATE_CERTI_ON_OGQ_STANDALONE_AI_ALL_FIXED`
**Modified:** 2026-08-16 18:37:54 KST · **pubspec:** `2.1.0+3`

아키텍처 전환점. 휴대폰 자체 1.7B/0.6B GGUF AI를 기본으로 하고 PC Ollama 14B/8B/4B는 선택적 고성능 모드로 분리. Android wrapper를 프로젝트에 고정하지 않고 실행 시 현재 Flutter에 맞게 준비하는 구조 도입.

Files: 47 · delta: +12 / -47 / ~7.

## v15 — `FINAL_ULTIMATE_CERTI_ON_OGQ_AI_MODEL_SELECTOR_NO_TIMEOUT`
**Modified:** 2026-08-16 14:03:24 KST · **pubspec:** `2.0.0+2`

앱 UI에서 `qwen3:14b / 8b / 4b`를 바로 선택하도록 모델 selector 추가. 실제 생성 요청 강제 timeout을 제거하여 14B 첫 로딩이 오래 걸려도 응답을 기다리도록 변경. 4B 설치/선택 지원 추가.

Files: 82 · delta: +3 / -0 / ~7.

## v14 — `CERTI_ON_OGQ_QWEN14B_SMART_CHAT_FIXED_V2`
**Modified:** 2026-08-16 13:44:00 KST · **pubspec:** `2.0.0+2`

Smart Chat 수정본의 Chrome 실행 문제를 보완. `CHROME_RUN_HELP.txt`, 내부 web-server runner를 추가하고 `RUN_CHROME_AI.bat`을 수정.

Files: 79 · delta: +2 / -0 / ~1.

## v13 — `CERTI_ON_OGQ_QWEN14B_SMART_CHAT_FIXED`
**Modified:** 2026-08-16 13:38:06 KST · **pubspec:** `2.0.0+2`

Qwen14B Smart Chat 개선. CERTI:ON의 앱 기능 문맥을 프롬프트에 더 제공하고 백엔드와 Flutter의 채팅 응답 처리를 조정.

Files: 77 · delta: +0 / -0 / ~3.

## v12 — `CERTI_ON_OGQ_ORIGINAL_LOCAL_QWEN14B`
**Modified:** 2026-08-16 13:27:22 KST · **pubspec:** `2.0.0+2`

OpenAI 계열을 제거하고 원래 CERTI:ON UI/공식 데이터 기능을 Ollama `qwen3:14b` 중심으로 재정리. Android Gradle wrapper/MainActivity/Manifest/resources까지 프로젝트에 포함한 시점.

Files: 77 · delta: +30 / -9 / ~10.

## v11 — `CERTI_ON_OGQ_LOCAL_AI_SMART_V2`
**Modified:** 2026-08-16 12:14:02 KST · **pubspec:** `2.0.0+2`

Local AI Smart 사용성 개선. `LOCAL_AI_QUICK_START.txt` 추가, 설치·실행·모델 선택 BAT 동작을 보정.

Files: 56 · delta: +1 / -0 / ~7.

## v10 — `CERTI_ON_OGQ_LOCAL_AI_SMART`
**Modified:** 2026-08-16 12:08:46 KST · **pubspec:** `2.0.0+2`

클라우드 OpenAI에서 **Ollama + Qwen3 로컬 AI**로 전환. 기본 14B, 경량 8B, 고성능 30B 선택 스크립트를 제공하고 공식 일정 DB를 먼저 추출해 local model에 전달하는 RAG형 구조 도입.

Files: 55 · delta: +4 / -0 / ~11.

## v09 — `CERTI_ON_OGQ_FULL_FIXED_V2`
**Modified:** 2026-08-16 11:40:18 KST · **pubspec:** `2.0.0+2`

Flutter Web의 ListTile/CheckboxListTile assertion과 Chrome/Android에서 local backend `/health`를 못 찾아 `Failed to fetch`가 발생하던 문제를 수정. 포트를 8791로 분리하고 Node launcher 기반 실행 자동화 추가.

Files: 51 · delta: +3 / -4 / ~9.

## v08 — `CERTI_ON_OGQ_FULL_FIXED`
**Modified:** 2026-08-16 11:09:16 KST · **pubspec:** `2.0.0+2`

`RUN_CERTION.bat` 중심 Full Fixed. OpenAI 실제 연결 상태 검사, Android/Chrome 자동 선택, AI 상태 UI와 MY 기능 안정화, backend status endpoint 정리.

Files: 52 · delta: +6 / -4 / ~7.

## v07 — `z_real_final_final_final_certi_on_ogq_ai_fixed`
**Modified:** 2026-08-16 01:59:54 KST · **pubspec:** `2.0.0+2`

VS Code tasks, `PREPARE_AI.bat`, `RUN_CHROME_AI.bat`을 추가해 AI backend와 Chrome 테스트 흐름을 보강. pubspec 및 `lib/main.dart` AI 동작 수정.

Files: 50 · delta: +3 / -0 / ~4.

## v06 — `certi_on_ogq_ultimate4`
**Modified:** 2026-08-16 01:51:54 KST · **pubspec:** `2.0.0+2`

대회 실행에 필요한 파일만 남기는 정리본. 중복 루트 `main.dart`, 여러 설명/점검 스크립트를 제거하고 `ONE_CLICK_RUN.bat` / `START_BACKEND.bat` 중심으로 구조를 단순화.

Files: 47 · delta: +0 / -14 / ~4.

## v05 — `certi_on_ogq_ultimate3`
**Modified:** 2026-08-16 01:42:48 KST · **pubspec:** `2.0.0+2`

OpenAI 실사용 연결을 위한 key setup, Android configure 도구, backend 로직 확장. 역사 `.env`에 실제 OpenAI 형식 비밀키가 있었으므로 공개 아카이브에서는 `.env`를 제외.

Files: 61 · delta: +3 / -0 / ~11.

## v04 — `certi_on_ogq_ultimate2`
**Modified:** 2026-08-16 01:16:58 KST · **pubspec:** `2.0.0+2`

19개 카테고리/기능 이미지 asset과 `lib/main.dart`를 함께 갱신한 UI/비주얼 개선 스냅샷.

Files: 58 · delta: +0 / -0 / ~21.

## v03 — `certi_on_ogq_ultimate1`
**Modified:** 2026-08-16 01:11:24 KST · **pubspec:** `2.0.0+2`

v02 경량 시제품에서 ULTIMATE 전체 구조를 다시 복원. 공식 일정 JSON, 19개 이미지 asset, Node backend, GitHub sync workflow, 관련 문서와 실행 도구를 재통합.

Files: 58 · delta: +36 / -0 / ~9.

## v02 — `certi_on_ogq`
**Modified:** 2026-08-16 00:12:44 KST · **pubspec:** `1.0.0+1`

경량 Flutter 시제품. backend/공식 data asset/이미지/workflow를 빼고 Flutter SDK 기본 위젯 중심으로 홈·탐색·캘린더·AI 요약·MY 상호작용을 시연하는 데모 구조.

Files: 22 · delta: +0 / -36 / ~9.

## v01 — `certi_on_ogq_ultimate`
**Modified:** 2026-08-15 15:53:22 KST · **pubspec:** `2.0.0+2`

확인 가능한 가장 오래된 ULTIMATE 통합본. 공식 고정 일정 101건 + 상시시험 6종, 이미지, Node backend, 자동 동기화 workflow, one-click 실행 구조를 포함.

Files: 58 · initial snapshot.
