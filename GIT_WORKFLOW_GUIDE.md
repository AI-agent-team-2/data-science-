# Очень простой гайд по Git

## 1) Скачать проект
```powershell
git clone https://github.com/AI-agent-team-2/data-science-.git san_bot
cd .\san_bot
```

## 2) Перейти в вашу ветку
```powershell
git checkout feature/integration-misha
git pull origin feature/integration-misha
```

## 3) Подтянуть свежий main
```powershell
git fetch origin main
git merge origin/main
```

## 4) Коммит и пуш
```powershell
git status -sb
git add .
git commit -m "Короткое понятное сообщение"
git push origin feature/integration-misha
```

## 5) Если конфликт
```powershell
git status
```
Уберите в файлах маркеры `<<<<<<<`, `=======`, `>>>>>>>`, затем:
```powershell
git add .
git commit
git push origin feature/integration-misha
```

## 6) Главное правило
- Всегда работайте в `feature/integration-misha`, не в `main`.

## 7) Как слить изменения в main

Рекомендуется через Pull Request:
- Запушьте ветку `feature/integration-misha`.
- На GitHub откройте PR: `feature/integration-misha` -> `main`.
- После проверки нажмите `Merge`.

Локально (если нужно руками):
```powershell
git checkout main
git pull origin main
git merge feature/integration-misha
git push origin main
```
