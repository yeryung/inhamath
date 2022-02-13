# Overthewire bandit

# Level 0 → Level 1

id : bandit0

password : bandit0

```bash
ssh bandit0@bandit.labs.overthewire.org -p 2220
```

비밀번호 칠 때 원래  표기 안됨

```bash
ls
```

폴더에 있는것들 나열(숨겨진 파일은 나열 안됨)

```bash
cat readme
```

# Info

### server

- 서비스를 제공하는 컴퓨터
- google 등

### client

- 서비스를 요청하는 컴퓨터
- 본인이 사용하고 계시는 컴퓨터 등

### internet

- server 와 client 를 연결하는 전선

### IP address

- 컴퓨터들의 주소, 컴퓨터들끼리 통신을 하기 위해 필요
- ISP(internet server provider, 통신사) 가 IP를 할당해준다
- 127.0.0.1 와 같은 형태

※ IP 주소는 사용자가 기억하기 어렵기 때문에 이름을 붙이는데 이를 domain name 이라 한다

※ 127.0.0.1 는 본인 컴퓨터의 서버를 가리키는 IP 이고 domain name 은 localhost 이다

※ 어떤 컴퓨터가 서버 컴퓨터일때 그 서버의 IP 주소는 컴퓨터의 IP 주소

## Internet 을 이용해 server 와 client 가 통신하는 과정(Web)

### web browser

- client 가 server 와 통신하기 위한 프로그램
- firefox, internet explorer, chrome 등이 있다

### webserver

- server 가 client 와 통신하기 위한 프로그램
- apache, iis, nginx 가 있다

※ program  vs  process

프로그램은 file 에 작성되어있는 코드들이고 process 는 실행중인 프로그램이다

### 과정

- client 가 접속하려는 서버의 IP 주소와 원하는 정보를 web-browser 에  입력한다
- google 과 같은 검색엔진을 이용해 접속하는 경우도 위의 과정 거침
- domain name 을 입력하면 web browser 가 이를 IP 주소로 바꾼다
- server 는 client 가 원하는 정보를 찾아서 전송해준다

### port

- server 와 client 가 만나기로 한 곳
- server 의 종류를 식별하기 위해 사용한다. 즉, 특정 server는 약속된 포트가 있다

※ 특정 server 가 약속된 포트만 써야 하는 것은 아니다

※ server 와 프로세스를 동일한 의미로 서술했습니다

※ web-browser 로 server 에 접속하려면 IP 주소와 포트가 맞아야한다

## ssh

- secure shell
- A 컴퓨터에서 B 컴퓨터를 제어하기 위한 도구

### ssh-server

- server 가 client 와 통신하기 위한 프로그램
- openssh-server 등이 있다

### ssh-client

- client 가 server 와 통신하기 위한 프로그램
- openssh-client 등이 있다

※ ssh의 약속된 port 는 22번이다

※ 원격제어에는 Telnet, SSH 가 있다

※ Telnet vs ssh

ssh 가 더 안전하다

ssh

ㅇ openssh-client 를 실행시키는 프로그램

ㅇ ssh <longin_id>@<IP_address>

- ssh를 사용해서 <IP_address> 를 가진 서버에 접속

- 자동으로 22번 포트를 사용한다

ㅇ ssh -p <숫자> <longin_id>@<IP_address>

- 22번 포트를 사용하지 않을 때

※ 모든 명령어는 프로그램이다

※ <IP_address> 에 domain name 을 사용해도 무관하다

## reference

- ssh 명령어 사용법
[https://phoenixnap.com/kb/linux-ssh-commands](https://phoenixnap.com/kb/linux-ssh-commands)

# Level 1 → Level 2

id : bandit1

password :  boJ9jbbUNNfktd78OOpsqOltutMc3MY1

```bash
cat ./-
```

— 기호는 linux 명령어에서 옵션을 추가할때 사용하는 것이므로

cat - 으로 파일을 열 수 없다(Ctrl + C 누르면 명령 종료)

cat [파일의 경로]

reference

- 이름이 — 인 파일 여는 방법
[https://stackoverflow.com/questions/42187323/how-to-open-a-dashed-filename-using-terminal](https://stackoverflow.com/questions/42187323/how-to-open-a-dashed-filename-using-terminal)

# Level 2 → Level 3

id : bandit2

password : CV1DtqXWVFXTvM2F0k09SHz0YwRINYA9

```bash
cat spaces\ in\ this\ filename
```

tap 누르면 자동완성된다(cat spaces 까지 누르고 tap 누르셈)

reference

- ~

# Level 3 → Level 4

id : bandit3

password : UmHadQclWmgdLOKQ3YNgjWxGoRMb5luK

```bash
cd inhere
```

```bash
ls -al
```

옵션설명
-a 옵션 : 숨겨진 파일도 모두 보여줌(all)
-l 옵션 : 파일명을 세로방향으로 쭉 나열(원래 가로로 나열함)

```bash
cat .hidden
```

숨겨진 파일은 파일이름앞에 . 이 붙어있다 

reference

- ls 명령어 옵션
[https://www.javatpoint.com/linux-ls](https://www.javatpoint.com/linux-ls)

# Level 4 → Level 5

id : bandit4

password : pIwrPrtPN36QITSp3EQaw936yaFoFgAB

```bash
cd inhere
```

```bash
file ./-file07
```

file 명령어는 파일의 타입을 확인하는 명령어이다

text : the file contains only printing characters
data :  usually 'binary' or non-printable

```bash
cat ./-file07
```

terminal 에 출력된게 너무 많아서 복잡하니 정리좀 해주자
(간단하게 clear 을 해도된다 ~ reset 과 달리 이건 위로 scroll 하면 다시 더러운 화면이 나온다)

```bash
reset
```

reference

- file 명령어의 결과로 나오는 data, ascii 의 의미
[https://linux.die.net/man/1/file](https://linux.die.net/man/1/file)
- reset   vs   clear
[https://superuser.com/questions/122911/what-commands-can-i-use-to-reset-and-clear-my-terminal](https://superuser.com/questions/122911/what-commands-can-i-use-to-reset-and-clear-my-terminal)

# Level 5 → Level 6

id : bandit5

password : koReBOKuIDDepwhWk7jZC0RTdopnAYKh

```bash
find . -type f ! -executable -size 1033c -exec cat {} \;
```

reference

- non-executable 파일 검색하는 방법
[https://stackoverflow.com/questions/70539901/how-can-i-find-all-non-executable-files-in-a-directory-in-linux](https://stackoverflow.com/questions/70539901/how-can-i-find-all-non-executable-files-in-a-directory-in-linux)
- 특정 사이즈의 파일 찾는 방법
[https://www.ducea.com/2008/02/12/linux-tips-find-all-files-of-a-particular-size/](https://www.ducea.com/2008/02/12/linux-tips-find-all-files-of-a-particular-size/)

# Level 6 → Level 7

id : bandit6

password : DXjZPULLxYr17uwoI01bNLQbtFemEgo7

```bash
find / -user bandit7 -group bandit6 -size 33c
```

permission denied 를 지우고 싶은 욕망이 생긴다

```bash
find / -user bandit7 -group bandit6 -size 33c 2>&1 | grep -v "Permission denied"
```

```bash
cat /var/lib/dpkg/info/bandit7.password
```

reference

- find owner 와 group 으로 검색하는 방법
[https://www.cyberciti.biz/faq/how-do-i-find-all-the-files-owned-by-a-particular-user-or-group/](https://www.cyberciti.biz/faq/how-do-i-find-all-the-files-owned-by-a-particular-user-or-group/)
- find 결과로 ‘permission denied’ 안보는 방법
[https://unix.stackexchange.com/questions/42841/how-to-skip-permission-denied-errors-when-running-find-in-linux](https://unix.stackexchange.com/questions/42841/how-to-skip-permission-denied-errors-when-running-find-in-linux)
- pipe command 의미( | 이렇게 생긴거)
[https://superuser.com/questions/756158/what-does-the-linux-pipe-symbol-do](https://superuser.com/questions/756158/what-does-the-linux-pipe-symbol-do)
- 2>&1 의미
[https://stackoverflow.com/questions/818255/in-the-shell-what-does-21-mean](https://stackoverflow.com/questions/818255/in-the-shell-what-does-21-mean)
- difference between redirection and pipe ( > 요거랑 | 요거의 차이)
[https://askubuntu.com/questions/172982/what-is-the-difference-between-redirection-and-pipe#:~:text=Redirection is used to redirect,%2C e.g. ls | grep file](https://askubuntu.com/questions/172982/what-is-the-difference-between-redirection-and-pipe#:~:text=Redirection%20is%20used%20to%20redirect,%2C%20e.g.%20ls%20%7C%20grep%20file).
> 는 기존의 파일을 overwright 한다(데이터 유실된다!! = 회사에서 쓰면 퇴사 쌉가능)
데이터 추가할꺼면  > 대신 >> 쓸 것!

# Level 7 → Level 8

id : bandit7

password : HKBPTKQnIay4Fw76bEy8PVxKEDQRKTzs

```bash
ls -al
```

뭔가 계속 출력된다. 어지러우니까 출력을 멈추자 Ctrl + C

```bash
grep "millionth" data.txt
```

reference

- grep 사용법
[https://www.cyberciti.biz/faq/howto-use-grep-command-in-linux-unix/](https://www.cyberciti.biz/faq/howto-use-grep-command-in-linux-unix/)

# Level 8 → Level 9

id : bandit8

password : cvX2JJa4CFALtqS87jk27qwqGhBM9plV

```bash
sort data.txt | uniq -u
```

reference

- uniq 명령어 사용법
[https://www.geeksforgeeks.org/uniq-command-in-linux-with-examples/](https://www.geeksforgeeks.org/uniq-command-in-linux-with-examples/)

# Level 9 → Level 10

id : bandit9

password : UsvVyFSfZZWbi6wgC7dAFyFuR6jQQUhR

```bash
strings data.txt
```

reference

- strings 명령어 사용법
[https://www.thegeekstuff.com/2010/11/strings-command-examples/](https://www.thegeekstuff.com/2010/11/strings-command-examples/)

# Level 10 → Level 11

id : bnadit10

password : truKLdjsbJ5g7yyJ2X2R0o3a5HQJFuLk

```bash
base64 -d data.txt
```

reference

- base64 명령어 사용법
[https://linuxhint.com/bash_base64_encode_decode/](https://linuxhint.com/bash_base64_encode_decode/)

# Level 11 → Level 12

id : bandit11

password : IFukwKGsFW8MOq3IRFqrxE1hxTNEbUPR

```bash
cat data.txt | tr 'A-Za-z' 'N-ZA-Mn-za-m'
```

reference

- rot13 decoding
[https://askubuntu.com/questions/1085069/how-can-i-decode-a-file-where-each-letter-has-been-replaced-with-the-letter-13-l](https://askubuntu.com/questions/1085069/how-can-i-decode-a-file-where-each-letter-has-been-replaced-with-the-letter-13-l)

# Level 12 → Level 13

id :  bandit12

password : 5Te8Y4drgCRfCx8ugdwuEX8KFC6k2EUu

reference

- ~

# Level 14 → Level 15

id : 

password :

reference

- ~