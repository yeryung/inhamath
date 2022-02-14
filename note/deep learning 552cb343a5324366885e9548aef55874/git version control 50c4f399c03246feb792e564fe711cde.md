# git version control

# git 으로 할 수 있는 것

- 폴더의 변경사항에 대한 메모
- 특정 시점에서의 코드로 돌아갈 수 있다(Ctrl Z 마냥)
- 파일의 변화 과정을 세세히 볼 수 있다
- git bisect 를 통해 버그를 효과적으로 탐색 가능
[https://www.youtube.com/watch?v=-UERw--siBw](https://www.youtube.com/watch?v=-UERw--siBw)

# git 에게 관리하는 파일을 지정하는 과정

1. 원하는 파일을 특정 장소(stage area)에 모아두었다가
2. 한꺼번에 파일을 commit

# Create

```bash
git init [directory]
```

git 을 사용할 수 있는 환경세팅(.git 이라는 폴더가 생김)

```bash
git config --global user.name [name]
git config --global user.email [email]
```

사용자 정보 설정

```bash
git config --global core.editor "[path_of_editor]"
```

commit 메세지를 적을 editor 변경

reference

- 작성자 정보 기입
[https://git-scm.com/book/ko/v2/시작하기-Git-최초-설정](https://git-scm.com/book/ko/v2/%EC%8B%9C%EC%9E%91%ED%95%98%EA%B8%B0-Git-%EC%B5%9C%EC%B4%88-%EC%84%A4%EC%A0%95)

```bash
git status
```

현재 폴더의 상태(어떤 파일이 수정되었는지 등등)를 보여줌

```bash
git add [file_name]
```

file 을 stage area 에 추가

```bash
git add [director_name]
```

directory 하위 모든 파일을 stage area 에 포함

```bash
git commit -m [message]
```

stage area 에 있는 파일들의 변화를 저장한다

```bash
git commit -am [message]
```

tracking 중인 파일들을 알아서 stage area 에 넣고 commit 함
※ 새로 만든 파일(untracking) 들은 commit 안됨. 따라서 얘들은 add 명령어를 사용해야함 

```bash
git commit --amend
```

방금 commit 한 메세지 수정

# Read

```bash
git log 
```

log 를 보여준다(log 는 기록? 정도로 이해하면 될 듯)

```bash
git log -stat
```

log 의 통계를 보여줌(몇개가 없어지고 생겼는지)

```bash
git diff
```

현재 수정된 working tree 의 변화를 보여줌

```bash
git log -p
```

각 commit 의 diff 명령어 결과를 보여줌

```bash
git reflog
```

명령어 log 를 보여줌

![Untitled](git%20version%20control%2050c4f399c03246feb792e564fe711cde/Untitled.png)

왼쪽 갈색 숫자는 절대적 기준의 commit id 이고 HEAD@{?} 는 현시점 기준 상대적 commit id 이다

위 그림 기준

```bash
git reset --hard 2313136
```

혹은

```bash
git --reset HEAD@{1}
```

를 통해 방금 commit 하기 전 상황으로 되돌아갈 수 있다

※

```bash
git config --global alias.undo "reset --hard HEAD@{1}"
```

로 별칭? 을 지정하고

```bash
git undo
```

로 사용하기도 한다

reference

- git log 명령어 사용법
[https://git-scm.com/book/ko/v2/Git의-기초-커밋-히스토리-조회하기](https://git-scm.com/book/ko/v2/Git%EC%9D%98-%EA%B8%B0%EC%B4%88-%EC%BB%A4%EB%B0%8B-%ED%9E%88%EC%8A%A4%ED%86%A0%EB%A6%AC-%EC%A1%B0%ED%9A%8C%ED%95%98%EA%B8%B0)
- git diff 명령어 사용법
[https://yoongrammer.tistory.com/30](https://yoongrammer.tistory.com/30)

# Update

```bash
git checkout [log_id]
```

log_id 의 상태로 돌아감

```bash
git checkout master
```

최신 상태로 돌아감

# Delete

(사실은 git은 commit 을 지우지 않는다 → 항상 복원 가능)

```bash
git reset --hard [log_id]
```

log_id 상태로 돌아감(그러면서 그 이후에것은 다 지움)

※ default mixed

![Untitled](git%20version%20control%2050c4f399c03246feb792e564fe711cde/Untitled%201.png)

- hard : 작업한거 다 날리고 싶을 때 씀

```bash
git reset --hard HEAD
```

- soft : 지금까지 한거 보존하면서 과거로 돌아가고 싶을 때 씀

```bash
git reset --soft
```

reference [https://www.youtube.com/watch?v=A6jzRmygVXM&list=PLuHgQVnccGMAvTJlPGzizAkyqXfZ9IyY8&index=5](https://www.youtube.com/watch?v=A6jzRmygVXM&list=PLuHgQVnccGMAvTJlPGzizAkyqXfZ9IyY8&index=5)

```bash
git revert [log_id]
```

이 명령어는 이해가 잘 안됨

reference

- checkout   vs   reset   vs   revert
[https://stackoverflow.com/questions/8358035/whats-the-difference-between-git-revert-checkout-and-reset](https://stackoverflow.com/questions/8358035/whats-the-difference-between-git-revert-checkout-and-reset)

background knowledge

- 현재 working tree 의 상태는 head 가 가리키는 commit
- head 가 branch 를 가리키지 않을때 detached라고 한다
- checkout 은 head 의 포인터만을 바꾼다
- reset 은 head가 branch 를
가리키고 있을 때 : branch 의 포인터가 변경됨
가리키고 있지 않을 때 (detached) : head 의 포인터만 변경됨

![git.gif](git%20version%20control%2050c4f399c03246feb792e564fe711cde/git.gif)

reference

- head 의 의미
[https://www.youtube.com/watch?v=Eunds0Kv9gc&list=PLuHgQVnccGMAvTJlPGzizAkyqXfZ9IyY8&index=2](https://www.youtube.com/watch?v=Eunds0Kv9gc&list=PLuHgQVnccGMAvTJlPGzizAkyqXfZ9IyY8&index=2)
- reset   vs   checkout
[https://www.youtube.com/watch?v=4tJjPWnNZNw&list=PLuHgQVnccGMAvTJlPGzizAkyqXfZ9IyY8&index=3](https://www.youtube.com/watch?v=4tJjPWnNZNw&list=PLuHgQVnccGMAvTJlPGzizAkyqXfZ9IyY8&index=3)

![Untitled](git%20version%20control%2050c4f399c03246feb792e564fe711cde/Untitled%202.png)

[https://www.youtube.com/watch?v=NBb-FFB2mJk&list=PLuHgQVnccGMAvTJlPGzizAkyqXfZ9IyY8&index=8](https://www.youtube.com/watch?v=NBb-FFB2mJk&list=PLuHgQVnccGMAvTJlPGzizAkyqXfZ9IyY8&index=8)