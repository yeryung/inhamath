# git backup&collaborate by SSH

# SSH 생성

```bash
ssh-keygen -t ed25519 -C "[your_email@example.com]"
```

public key 는 노출되어도 되지만 private key 는 노출되면 안됨!!

reference

- ssh 공개키 만들기
[https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)

# ****ssh-agent 에 private key 등록****

[https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)

# ****GitHub 에 public key 등록****

![Untitled](git%20backup&collaborate%20by%20SSH%20adc9ccf6f7244ac987d114cd324369bf/Untitled.png)

[https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)

# 잘 됐나 테스트

[https://docs.github.com/en/authentication/connecting-to-github-with-ssh/testing-your-ssh-connection](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/testing-your-ssh-connection)

# ssh-agent자동 실행

[https://docs.github.com/en/authentication/connecting-to-github-with-ssh/working-with-ssh-key-passphrases](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/working-with-ssh-key-passphrases)

# git push

```bash
git remote add [remote_repository_alias] [remote_repository_url]
```

백업을 할 컴퓨터 주소(remote repository url)를 저장

※ 기본적인 remote repository 의 alias 는 origin 으로 사용한다

```bash
git remote add origin git@github.com:tinycaterpillar/tinycaterpillar.github.io.git
```

```bash
git remote -v
```

remote repository 의 alias 와 url 을 보여준다

```bash
git push --set-upstream origin master
```

추후 git push 라는 명령어는 alias 가 origin 인 remote repository 의 master branch 를 대상으로 실행

reference

- remote repository 지정방법
[https://docs.github.com/en/get-started/getting-started-with-git/about-remote-repositories](https://docs.github.com/en/get-started/getting-started-with-git/about-remote-repositories)

# git clone

```bash
git clone [remote_repository_url] [file_name]
```

remote repository 의 파일을 복사(clone)해온다

file_name 을 지정하면 폴더이름을 재지정할 수 있다

cf.

```bash
git clone git@github.com:tinycaterpillar/tinycaterpillar.github.io.git 
```

tinycaterpillar.github.io.git  라는 폴더로 복사됨

```bash
git clone git@github.com:tinycaterpillar/tinycaterpillar.github.io.git short
```

short 라는 폴더로 복사됨

# git pull

```bash
git pull
```

remote repository 의 파일의 정보를 가져오고 merge 를 진행

※

```bash
git fetch; git merge FETCH_HEAD
```

와 동일하다

# git collaborate

<aside>
🦝 pull → commit → push

</aside>