## 포트확인
sudo lsof -i TCP:5000
## 포트종료
kill PID
## 포트 일괄종료
sudo fuser -k 5000/tcp