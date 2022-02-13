# WSL2 install & Terminal customize

1. WSL2 설치
[https://velog.io/@full_accel/Windows-WSL2로-우분투-설치하기](https://velog.io/@full_accel/Windows-WSL2%EB%A1%9C-%EC%9A%B0%EB%B6%84%ED%88%AC-%EC%84%A4%EC%B9%98%ED%95%98%EA%B8%B0)

2. WSL 에서 jetbrains toolbox 를 Cli 로 실행하기
[https://www.jetbrains.com/help/clion/opening-files-from-command-line.html](https://www.jetbrains.com/help/clion/opening-files-from-command-line.html)
    
    
    1. 실행파일 위치를 찾는다
    
    ![Untitled](WSL2%20install%20&%20Terminal%20customize%20e3905cfa0f4843dcae25627df9102222/Untitled.png)
    
    1. 나는 toolbox 를 쓰고있으므로 다음 세팅까지 해준다
    
    ![Untitled](WSL2%20install%20&%20Terminal%20customize%20e3905cfa0f4843dcae25627df9102222/Untitled%201.png)
    
    1. 실행하려는 파일을 링크해준다
    (WSL 에서 리눅스 커널을쓰므로 리눅스 명령어를 쓴다)
    2. 다음 명령어를 실행(권한문제뜨면 명령어 앞에 sudo 붙여준다)
    
    ```bash
    ln -s /mnt/c/Users/user/AppData/local/JetBrains/Toolbox/apps/CLion/ch-0/213.6777.58/bin/clion64.exe /usr/local/bin/clion
    ```
    
    ※ ln -s [링크하려는 프로그램] /usr/local/bin
    
    ※ 리눅스에서 /usr/local/bin 에 있는 링크는 임의의 경로에서 실행할 수 있다
    
    기타
    
    나머지 editor 들도 링크해준다
    
    ```bash
    ln -s /mnt/c/Users/user/AppData/Local/JetBrains/Toolbox/apps/PyCharm-P/ch-1/213.6777.50/bin/pycharm64.exe /usr/local/bin/pycharm
    ```
    
    ```bash
    ln -s /mnt/c/Users/user/AppData/Local/JetBrains/Toolbox/apps/WebStorm/ch-0/213.6777.57/bin/webstorm64.exe /usr/local/bin/webstor
    ```
    
3. 터미널에 https://github.com/ohmyzsh/ohmyzsh 설치([https://nomadcoders.co/windows-setup-for-developers](https://nomadcoders.co/windows-setup-for-developers))
4. 터미널 theme ‣ 로 변경
5. 터미널 디자인 세팅하는 파일

```bash
/mnt/c/Users/user/AppData/Local/Packages/Microsoft.WindowsTerminal_8wekyb3d8bbwe/Localstate/settings.json
```