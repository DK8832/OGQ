# CERTI:ON — 24-version history archive

이 폴더는 `flutter project(6).zip` 안의 Flutter 프로젝트들을 **Windows 폴더 수정 시각 기준으로 가장 오래된 버전부터 최신 버전까지** 재구성한 기록입니다.

## 정렬 기준

- v01 = 가장 오래된 폴더
- v24 = 가장 최신 폴더
- 기준 시각은 원본 ZIP에 보존된 폴더 수정 시각(KST)입니다.
- 각 버전의 원래 폴더명은 그대로 기록했습니다.
- `pubspec.yaml`의 내부 버전도 별도로 기록했습니다. 폴더 수정 순서와 pubspec 버전 번호는 항상 일치하지 않으므로, **히스토리 번호(v01~v24)는 파일 수정 시각 순서**를 뜻합니다.

## 분석 범위

원본 압축파일의 4,398개 파일/폴더 엔트리를 검사하여 Flutter 앱 소스, 공식 일정 데이터, 이미지 asset, Node 백엔드, Android 네이티브 프로젝트, Web, 테스트, 실행 BAT/PowerShell 도구, 문서, build/cache 산출물을 분류했습니다.

공개용 소스 패키지는 `build/`, `.dart_tool/`, Gradle cache, APK 등 생성 산출물을 제외하고 **프로젝트를 다시 만들 수 있는 소스·설정·asset·backend·Android 소스**를 포함하도록 구성했습니다.

## 보안 처리

역사 버전 중 5개의 `backend/.env`에서 OpenAI 형식의 실제 비밀키 패턴이 발견되었습니다. 값 자체는 이 저장소에 기록하지 않았으며 공개용 패키지에서도 모두 제외했습니다. `.env.example`은 안전한 예시 파일로 유지합니다.

## 파일

- `RELEASES.md` — GitHub Release 스타일의 24개 버전 변경 기록
- `VERSION_INDEX.csv` — 수정시각, 원본 폴더명, pubspec 버전, 파일 수, 변경량을 정렬한 표
- `SECURITY.md` — 공개 패키징 시 제외한 민감 파일 기록
- `releases/` — 버전별 개별 노트

## 전체 소스 아카이브

ChatGPT 작업 결과물에는 두 가지 형태를 준비했습니다.

1. 버전별 독립 `SOURCE.zip` 24개
2. 24개 버전 전체 소스를 한 번에 담은 solid 압축본

현재 연결된 GitHub 도구는 **GitHub Release 생성/바이너리 asset 업로드 액션을 제공하지 않기 때문에**, 이 브랜치에는 우선 버전 인덱스와 Release 노트를 저장합니다. 실제 소스 ZIP은 동일한 버전명·SHA-256 기준으로 별도 업로드할 수 있도록 준비되어 있습니다.
