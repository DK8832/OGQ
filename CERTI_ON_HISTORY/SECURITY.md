# CERTI:ON history packaging security report

## 검사 결과

- 원본 `flutter project(6).zip`의 파일/폴더 엔트리 4,398개를 검사했습니다.
- 역사 버전 중 **5개 `backend/.env` 파일에서 OpenAI 형식 비밀키 패턴**이 발견되었습니다.
- 비밀키 값 자체는 이 문서나 GitHub 저장소에 복사하지 않았습니다.
- 공개용 source package를 다시 검사한 결과 알려진 비밀키 패턴 매치는 **0건**이었습니다.

## 공개 패키지에서 제외한 민감 경로

- `certi_on_ogq_ultimate3/backend/.env`
- `certi_on_ogq_ultimate4/backend/.env`
- `z_real_final_final_final_certi_on_ogq_ai_fixed/backend/.env`
- `CERTI_ON_OGQ_FULL_FIXED/backend/.env`
- `CERTI_ON_OGQ_FULL_FIXED_V2/backend/.env`

모든 실제 `.env`는 공개 source package에서 제외하고 `.env.example`만 유지합니다. 또한 `android/local.properties`처럼 PC별 SDK 경로가 들어가는 로컬 설정도 제외합니다.

## 추가 제외 항목

`build/`, `.dart_tool/`, Android/Gradle cache와 APK 등 컴파일 산출물은 소스가 아니며 용량이 매우 커서 버전 source package에서는 제외합니다. 원본 아카이브에는 그대로 존재하며 전체 manifest에서 존재 여부와 크기를 추적할 수 있습니다.

> 과거 `.env` 안에 실제 키가 저장된 적이 있으므로 해당 키가 아직 유효하다면 폐기/교체하는 것이 안전합니다.
